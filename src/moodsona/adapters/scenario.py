"""A moodsona persona as the user simulator of a langwatch/scenario test (experimental).

Install with ``pip install 'moodsona[scenario]'``. Build the agent with
:func:`user_simulator_agent` and put it in ``scenario.run(agents=[...])`` in
place of scenario's own user simulator::

    user = user_simulator_agent(load_bundled_persona("anxious-newcomer"), provider)
    result = await scenario.run(
        name="first week", description="...", agents=[MyCompanion(), user, judge]
    )
    print(user.metrics(result.messages))

How scenario's conversation maps onto moodsona:

* Every call for a user message is one moodsona turn. The persona's hidden
  block (inner thought, emotion, energy, intent) stays inside the adapter;
  scenario only receives the visible message.
* A moodsona day ends after ``turns_per_day`` turns, or earlier when the
  persona ends it (``end_day`` / ``skip_tomorrow``). The next call writes the
  persona's notes about that day and starts the next one, so mood and day
  memory carry over exactly as in :func:`moodsona.run`. A day the persona
  skips has no messages and is passed over.
* The companion's reply to a turn is the text of the ``assistant`` messages
  that follow the persona's message in ``AgentInput.messages``. System, tool
  and developer messages, and assistant messages without text (a bare tool
  call), are not shown to the persona.
* scenario decides when the conversation stops (``max_turns``, a judge, a
  script), so the last day's notes are never written.
* The transcript holds the persona's turns and the replies to them, nothing
  else. User lines from a scenario script (``scenario.user("...")``) and an
  opening agent message are shown to the persona as part of the day's
  conversation but are not in the transcript or the metrics.

The ``scenario`` package is imported only by :func:`user_simulator_agent`;
importing this module does not need it.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import random
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from moodsona.llm.base import ChatMessage, LLMProvider, Role
from moodsona.metrics import TranscriptMetrics, compute_metrics
from moodsona.runner import Transcript, TurnRecord
from moodsona.simulator import SafetyHook, Simulator, SimulatorConfig, SimulatorTurn
from moodsona.state import DayNote, Mood, PersonaSeed

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


@dataclass(frozen=True, slots=True)
class _Pending:
    """The persona's latest turn, waiting for the companion's reply."""

    turn: SimulatorTurn
    day: int
    turn_in_day: int
    mood: Mood
    #: Position of the persona's message in the visible conversation.
    index: int


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


def _reply_after(history: Sequence[ChatMessage], index: int) -> str:
    """The companion's messages after position ``index``, up to the next user message."""
    lines: list[str] = []
    for message in history[index + 1 :]:
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
        self._day = 0
        self._turn_in_day = 0
        self._day_start = 0
        self._pending: _Pending | None = None
        self._turns: list[TurnRecord] = []
        self._notes: list[DayNote] = []
        self._skipped: list[int] = []
        #: Visible messages the next conversation must at least hold (ours included).
        self._seen = 0

    @property
    def day(self) -> int:
        """The current moodsona day (0 before the first message)."""
        return self._day

    async def call(self, input: Any) -> str:
        """Return the persona's next visible message (scenario's entry point).

        The simulator's model calls are blocking; they run in a worker thread
        so the event loop stays free.

        Args:
            input: scenario's ``AgentInput``. Only ``input.messages``, the
                conversation so far as OpenAI-style messages, is read.

        Returns:
            The message text; scenario adds it to the conversation as a user message.

        Raises:
            moodsona.llm.LLMError: The provider failed.
        """
        return await asyncio.to_thread(self.respond, input.messages)

    def respond(self, messages: Sequence[Message]) -> str:
        """Return the persona's next visible message (the blocking form of :meth:`call`).

        Args:
            messages: The conversation so far as OpenAI-style messages.

        Raises:
            ValueError: ``messages`` is shorter than the conversation this
                persona is already in (the agent was reused for another run).
            moodsona.llm.LLMError: The provider failed.
        """
        history = _visible(messages)
        if len(history) < self._seen:
            raise ValueError(
                "the conversation is shorter than the one this persona is in; "
                "use one agent per scenario run"
            )
        pending = self._pending
        if pending is not None:
            day_over = pending.turn.ends_day or pending.turn_in_day >= self.turns_per_day
            if day_over:
                # First: if the notes call fails, the turn stays pending and a retry repeats it.
                note = self.simulator.end_day(history[self._day_start :], day=self._day)
                if note is not None:
                    self._notes.append(note)
            self._turns.append(self._record(pending, history))
            self._pending = None
            if day_over:
                self._start_next_day(len(history))
        elif self._day == 0:
            self._start_next_day(0)  # day 1 also shows the persona an opening agent message
        turn = self.simulator.next_turn(
            history[self._day_start :], day=self._day, turn_in_day=self._turn_in_day + 1
        )
        self._turn_in_day += 1
        self._pending = _Pending(
            turn, self._day, self._turn_in_day, self.simulator.state.mood, len(history)
        )
        self._seen = len(history) + 1
        return turn.text

    def _start_next_day(self, start: int) -> None:
        self._day += 1
        while self.simulator.is_skipped(self._day):
            self._skipped.append(self._day)
            self._day += 1
        self.simulator.start_day(self._day)
        self._turn_in_day = 0
        self._day_start = start

    @staticmethod
    def _record(pending: _Pending, history: Sequence[ChatMessage]) -> TurnRecord:
        return TurnRecord.from_turn(
            pending.turn,
            day=pending.day,
            turn_in_day=pending.turn_in_day,
            companion_reply=_reply_after(history, pending.index),
            mood=pending.mood,
        )

    def transcript(self, messages: Sequence[Message] | None = None) -> Transcript:
        """The run so far, in the form :func:`moodsona.run` returns.

        Write it with :meth:`~moodsona.runner.Transcript.to_jsonl` and read it
        with ``moodsona report``. Calling this changes nothing.

        Args:
            messages: The final conversation (for example ``result.messages``
                of scenario's result), to fill in the companion's reply to the
                persona's last message. Without it that reply is ``""``.

        Returns:
            The transcript. ``days`` is the last day with a turn.
        """
        turns = list(self._turns)
        if self._pending is not None:
            turns.append(self._record(self._pending, _visible(messages or ())))
        return Transcript(
            persona_id=self.simulator.seed.id,
            rng_seed=self.rng_seed,
            turns=turns,
            notes=list(self._notes),
            skipped_days=list(self._skipped),
            days=max((t.day for t in turns), default=None),
            turns_per_day=self.turns_per_day,
        )

    def metrics(self, messages: Sequence[Message] | None = None) -> TranscriptMetrics:
        """The :mod:`moodsona.metrics` of :meth:`transcript`, with the simulator's language profile.

        Args:
            messages: See :meth:`transcript`.
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
    persona's state from call to call. The arguments are those of
    :class:`MoodsonaUser`.

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
    """The scenario-facing class, built once per ``scenario`` module; imports scenario."""
    try:
        module = importlib.import_module("scenario")
    except ImportError as exc:
        raise ImportError(
            f"moodsona's scenario adapter needs langwatch-scenario; {_INSTALL}"
        ) from exc
    return _build_agent_class(module)


@functools.cache
def _build_agent_class(module: types.ModuleType) -> type[MoodsonaUser]:
    base = getattr(module, "AgentAdapter", None)
    role = getattr(getattr(module, "AgentRole", None), "USER", None)
    if not isinstance(base, type) or role is None:
        raise ImportError(
            f"the installed 'scenario' module is not langwatch-scenario "
            f"(no AgentAdapter / AgentRole.USER); {_INSTALL}"
        )

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
            return _agent_class()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
