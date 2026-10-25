"""A moodsona persona as the user simulator of a langwatch/scenario test (experimental).

Install with ``pip install 'moodsona[scenario]'``. Build the agent with
:func:`user_simulator_agent` and put it in ``scenario.run(agents=[...])`` in
place of scenario's own user simulator::

    user = user_simulator_agent(load_bundled_persona("anxious-newcomer"), provider)
    result = await scenario.run(
        name="first week", description="...", agents=[MyCompanion(), user, judge]
    )
    print(user.metrics(result.messages))

How scenario's turns map onto moodsona days, and what the transcript holds, is
described in ``docs/integrations/scenario.md``. The ``scenario`` package is
imported only by :func:`user_simulator_agent`; importing this module does not
need it.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import random
import types
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from moodsona.llm.base import ChatMessage, LLMProvider, Role
from moodsona.metrics import TranscriptMetrics, compute_metrics
from moodsona.runner import Transcript, TurnRecord, day_is_over, next_active_day
from moodsona.simulator import SafetyHook, Simulator, SimulatorConfig
from moodsona.state import PersonaSeed

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile

__all__ = ["DEFAULT_TURNS_PER_DAY", "MoodsonaUser", "user_simulator_agent"]

#: Persona messages per moodsona day unless ``turns_per_day`` says otherwise.
DEFAULT_TURNS_PER_DAY = 4

_INSTALL = "install it with: pip install 'moodsona[scenario]'"

#: Name of the class :func:`user_simulator_agent` builds (an attribute of this
#: module once scenario is installed).
_AGENT_CLASS = "MoodsonaUserAgent"

#: OpenAI-style roles shown to the persona, mapped to moodsona's.
_VISIBLE_ROLES: dict[str, Role] = {"user": "user", "assistant": "assistant"}

Message = Mapping[str, Any]
"""One OpenAI-style chat message as scenario passes it (``{"role": ..., "content": ...}``)."""


def _text(content: Any) -> str:
    """The text of an OpenAI-style ``content``: a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text") for p in content if isinstance(p, Mapping) and p.get("type") == "text"
        ]
        return "\n".join(p for p in parts if isinstance(p, str))
    return ""


def _visible(messages: Sequence[Message]) -> list[ChatMessage]:
    """The user and assistant messages that carry text, as moodsona messages."""
    visible: list[ChatMessage] = []
    for message in messages:
        role = _VISIBLE_ROLES.get(str(message.get("role")))
        text = _text(message.get("content"))
        if role is not None and text.strip():
            visible.append(ChatMessage(role, text))
    return visible


def _reply_after(messages: Sequence[Message], index: int) -> str:
    """The companion's text after position ``index``, up to the next user message."""
    lines: list[str] = []
    for message in _visible(messages[index + 1 :]):
        if message.role == "user":
            break
        lines.append(message.content)
    return "\n".join(lines)


class MoodsonaUser:
    """A moodsona persona that writes the next user message whenever scenario asks.

    Create it with :func:`user_simulator_agent`, which adds scenario's
    ``AgentAdapter`` base class and the user role. This class holds the logic
    and needs no scenario installed.

    Args:
        seed: The persona.
        provider: The model that writes the persona's lines.
        turns_per_day: Maximum persona messages per moodsona day.
        rng_seed: Seed for overnight energy and typing noise.
        language: Language profile; None (the default) uses the persona's
            ``language`` (see :class:`~moodsona.simulator.Simulator`).
        safety_hook: Optional crisis detector (see :class:`~moodsona.simulator.Simulator`).
        config: Optional simulator settings.

    Attributes:
        simulator: The simulated user. Its ``state`` is the hidden state
            (mood, day notes) kept from one call to the next.
        turns_per_day: Maximum persona messages per moodsona day.
        rng_seed: Seed of the random source.

    Raises:
        ValueError: ``turns_per_day`` is below 1.
        LookupError: No language profile covers the persona's language.
    """

    def __init__(
        self,
        seed: PersonaSeed,
        provider: LLMProvider,
        *,
        turns_per_day: int = DEFAULT_TURNS_PER_DAY,
        rng_seed: int = 0,
        language: LanguageProfile | None = None,
        safety_hook: SafetyHook | None = None,
        config: SimulatorConfig | None = None,
    ) -> None:
        if turns_per_day < 1:
            raise ValueError("turns_per_day must be at least 1")
        self.simulator = Simulator(
            seed,
            provider,
            language=language,
            safety_hook=safety_hook,
            config=config,
            rng=random.Random(rng_seed),  # noqa: S311 - simulation, not cryptography
        )
        self.turns_per_day = turns_per_day
        self.rng_seed = rng_seed
        self._transcript = Transcript(seed.id, rng_seed, turns_per_day=turns_per_day)
        self._day = 0
        self._turn_in_day = 0
        #: Position in scenario's message list where today's conversation starts.
        self._day_start = 0
        #: The persona's latest turn, its reply still empty, and where its message sits.
        self._open: TurnRecord | None = None
        self._open_index = 0
        self._open_ends_day = False
        #: The scenario thread this persona talks in, once known.
        self._thread_id: str | None = None
        #: Messages the next call's list must at least hold (the persona's last one included).
        self._seen = 0

    @property
    def day(self) -> int:
        """The current moodsona day (0 before the first message)."""
        return self._day

    async def call(self, input: Any) -> str:
        """Return the persona's next visible message (scenario's entry point).

        The simulator's model calls are blocking; they run in a worker thread,
        on a copy of the messages, so the event loop stays free. Cancelling the
        call (a scenario timeout, for example) does not stop a model call that
        has already started; the thread finishes it in the background.

        Args:
            input: scenario's ``AgentInput``. Only ``input.messages``, the
                conversation so far as OpenAI-style messages, and
                ``input.thread_id`` are read.

        Returns:
            The message text; scenario adds it to the conversation as a user message.

        Raises:
            ValueError: The agent is already in another scenario run.
            moodsona.llm.LLMError: The provider failed.
        """
        messages = list(input.messages)
        return await asyncio.to_thread(self.respond, messages, thread_id=input.thread_id)

    def respond(self, messages: Sequence[Message], *, thread_id: str | None = None) -> str:
        """Return the persona's next visible message (the blocking form of :meth:`call`).

        Args:
            messages: The conversation so far as OpenAI-style messages.
            thread_id: scenario's id of the conversation. The first one seen
                binds the agent to that conversation. Without it, a message
                list shorter than the last one is taken as another conversation.

        Raises:
            ValueError: The messages belong to another conversation than the
                one this persona is in (the agent was reused for another run).
            moodsona.llm.LLMError: The provider failed.
        """
        self._check_conversation(messages, thread_id)
        transcript = self._transcript
        opened = self._open
        if opened is not None:
            if self._open_ends_day:
                # First: if the notes call fails, the turn stays open and a retry repeats it.
                today = _visible(messages[self._day_start :])
                note = self.simulator.end_day(today, day=self._day)
                if note is not None:
                    transcript.notes.append(note)
            transcript.turns.append(self._closed(opened, messages))
            self._open = None
            if self._open_ends_day:
                self._start_day(next_active_day(self.simulator, self._day, transcript.skipped_days))
                self._day_start = len(messages)
        elif self._day == 0:
            # Day 1 starts at the top, so the persona also sees an opening agent message.
            self._start_day(next_active_day(self.simulator, 0, transcript.skipped_days))
        turn_in_day = self._turn_in_day + 1
        today = _visible(messages[self._day_start :])
        turn = self.simulator.next_turn(today, day=self._day, turn_in_day=turn_in_day)
        self._turn_in_day = turn_in_day
        self._open = TurnRecord.from_turn(
            turn,
            day=self._day,
            turn_in_day=turn_in_day,
            companion_reply="",
            mood=self.simulator.state.mood,
        )
        self._open_index = len(messages)
        self._open_ends_day = day_is_over(turn, turn_in_day, self.turns_per_day)
        self._seen = len(messages) + 1
        return turn.text

    def _check_conversation(self, messages: Sequence[Message], thread_id: str | None) -> None:
        if thread_id is None:
            if len(messages) < self._seen:
                raise ValueError(
                    "the conversation is shorter than the one this persona is in; "
                    "use one agent per scenario run"
                )
        elif self._thread_id is None:
            self._thread_id = thread_id
        elif thread_id != self._thread_id:
            raise ValueError(
                f"this persona is in scenario thread {self._thread_id!r}, not {thread_id!r}; "
                "use one agent per scenario run"
            )

    def _start_day(self, day: int) -> None:
        self.simulator.start_day(day)
        self._day = day
        self._turn_in_day = 0

    def _closed(self, opened: TurnRecord, messages: Sequence[Message]) -> TurnRecord:
        """The open turn with the companion's reply from ``messages`` filled in."""
        return replace(opened, companion_reply=_reply_after(messages, self._open_index))

    def transcript(self, messages: Sequence[Message] | None = None) -> Transcript:
        """The run so far, in the form :func:`moodsona.run` returns.

        Write it with :meth:`~moodsona.runner.Transcript.to_jsonl` and read it
        with ``moodsona report``. Calling this changes nothing.

        Args:
            messages: The final conversation (for example ``result.messages``
                of scenario's result), to fill in the companion's reply to the
                persona's last message. Without it that reply is ``""``.

        Returns:
            The transcript. ``days`` is the last day with a turn or a skip.
        """
        done = self._transcript
        turns = list(done.turns)
        if self._open is not None:
            turns.append(self._closed(self._open, messages or ()))
        return replace(
            done,
            turns=turns,
            notes=list(done.notes),
            skipped_days=list(done.skipped_days),
            days=max([t.day for t in turns] + done.skipped_days, default=None),
        )

    def metrics(self, messages: Sequence[Message] | None = None) -> TranscriptMetrics:
        """The :mod:`moodsona.metrics` of :meth:`transcript`, with the simulator's language profile.

        Args:
            messages: See :meth:`transcript`.

        Returns:
            The metrics. The agreement, ask-back and wrong-language rates use
            the simulator's language profile (None when language checks are off).
        """
        return compute_metrics(self.transcript(messages), language=self.simulator.language)


def user_simulator_agent(
    seed: PersonaSeed,
    provider: LLMProvider,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    rng_seed: int = 0,
    language: LanguageProfile | None = None,
    safety_hook: SafetyHook | None = None,
    config: SimulatorConfig | None = None,
) -> MoodsonaUser:
    """A scenario ``AgentAdapter`` with the user role that plays ``seed``.

    Pass it to ``scenario.run(agents=[...])`` instead of scenario's
    ``UserSimulatorAgent``. Use one agent per scenario run: it keeps the
    persona's state from call to call.

    Args:
        seed: The persona.
        provider: The model that writes the persona's lines.
        turns_per_day: Maximum persona messages per moodsona day.
        rng_seed: Seed for overnight energy and typing noise.
        language: Language profile; None (the default) uses the persona's
            ``language`` (see :class:`~moodsona.simulator.Simulator`).
        safety_hook: Optional crisis detector (see :class:`~moodsona.simulator.Simulator`).
        config: Optional simulator settings.

    Returns:
        An instance of a subclass of both :class:`MoodsonaUser` and
        ``scenario.AgentAdapter``.

    Raises:
        ImportError: langwatch-scenario is not installed.
        ValueError: ``turns_per_day`` is below 1.
        LookupError: No language profile covers the persona's language.
    """
    return _agent_class()(
        seed,
        provider,
        turns_per_day=turns_per_day,
        rng_seed=rng_seed,
        language=language,
        safety_hook=safety_hook,
        config=config,
    )


def _agent_class() -> type[MoodsonaUser]:
    """The scenario-facing class, built once per scenario base class; imports scenario."""
    try:
        # Attribute access on the imported module rather than `from scenario import
        # ...`: mypy then needs neither scenario's package nor an ignore.
        scenario = importlib.import_module("scenario")
        base, role = scenario.AgentAdapter, scenario.AgentRole.USER
    except (ImportError, AttributeError) as exc:  # AttributeError: an unrelated `scenario`
        raise ImportError(
            f"moodsona's scenario adapter needs langwatch-scenario; {_INSTALL}"
        ) from exc
    return _build_agent_class(base, role)


@functools.cache
def _build_agent_class(base: type, role: Any) -> type[MoodsonaUser]:
    def body(namespace: dict[str, Any]) -> None:
        namespace["__module__"] = __name__
        namespace["__doc__"] = "A :class:`MoodsonaUser` that scenario runs as its user simulator."
        namespace["role"] = role

    # new_class rather than a class statement: mypy cannot check a base class
    # that is only known at run time, and this way needs no type: ignore.
    agent_class = types.new_class(_AGENT_CLASS, (MoodsonaUser, base), exec_body=body)
    return cast("type[MoodsonaUser]", agent_class)


if not TYPE_CHECKING:  # hidden from mypy, so a misspelt import still fails type checking

    def __getattr__(name: str) -> Any:
        """Resolve ``MoodsonaUserAgent`` on demand, so pickle can find the class."""
        if name == _AGENT_CLASS:
            try:
                return _agent_class()
            except ImportError as exc:
                message = f"{__name__}.{name} needs langwatch-scenario; {_INSTALL}"
                raise AttributeError(message) from exc
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
