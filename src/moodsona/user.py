"""A simulated user with the same ``reply(history) -> str`` method as a companion.

A :class:`~moodsona.companion.Companion` has ``reply(history) -> str``;
:class:`SimulatedUser` has the same method. Whatever alternates the two (an
eval harness such as openevals, langwatch/scenario or deepeval, or a loop of
your own) runs a moodsona persona against your chatbot: the persona's message
comes back as plain text, and the hidden state (emotion, energy, stance, day
notes) stays inside and ends up in the :class:`~moodsona.runner.Transcript`.
:func:`moodsona.run` is this loop with a companion::

    user = SimulatedUser(persona, provider, turns_per_day=4, days=3)
    history = []
    while not user.done:
        history.append(ChatMessage("user", user.reply(history), day=user.day))
        history.append(ChatMessage("assistant", companion.reply(tuple(history)), day=user.day))
    transcript = user.finish(history)

Days: the first reply starts day 1. A day ends after a turn whose intent ends
it (``end_day`` / ``skip_tomorrow``) or after ``turns_per_day`` turns. The next
reply then writes that day's notes and starts the next day the persona writes
on; a day it chose to skip has no messages. :attr:`SimulatedUser.day` after a
reply is the day that message belongs to; tag the messages with it, as above,
so the companion can tell the days apart. The persona ignores the tags.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeAlias

from moodsona.llm.base import ChatMessage, LLMProvider, Role
from moodsona.metrics import TranscriptMetrics, compute_metrics
from moodsona.runner import Transcript, TurnRecord
from moodsona.simulator import SafetyHook, Simulator, SimulatorConfig, SimulatorTurn
from moodsona.state import Mood, PersonaSeed, PersonaState

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile

History: TypeAlias = Sequence[ChatMessage | Mapping[str, Any]]
"""A conversation: moodsona messages or OpenAI-style ``{"role", "content"}`` mappings."""

#: OpenAI-style roles shown to the persona.
_ROLES: dict[str, Role] = {"user": "user", "assistant": "assistant"}


def _text(content: Any) -> str:
    """The text of an OpenAI-style ``content``: a string, or the text parts of a list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text") for p in content if isinstance(p, Mapping) and p.get("type") == "text"
        ]
        return "\n".join(p for p in parts if isinstance(p, str))
    return ""


def _parse(history: Sequence[object]) -> list[ChatMessage | None]:
    """``history`` position by position: the message the persona sees there, or None.

    A :class:`ChatMessage` is shown as it is, an empty one included (as
    :func:`moodsona.run` has always shown it); only system messages are left
    out. A mapping is shown when its role is ``user`` or ``assistant`` and it
    has text; tool calls, tool results and other roles are left out.

    Raises:
        TypeError: An item is neither a ChatMessage nor a mapping.
    """
    parsed: list[ChatMessage | None] = []
    for item in history:
        if isinstance(item, ChatMessage):
            parsed.append(None if item.role == "system" else item)
        elif isinstance(item, Mapping):
            role = _ROLES.get(str(item.get("role")))
            text = _text(item.get("content"))
            parsed.append(ChatMessage(role, text) if role is not None and text.strip() else None)
        else:
            raise TypeError(
                "each history item must be a ChatMessage or a mapping with 'role' and "
                f"'content', got {type(item).__name__}"
            )
    return parsed


def _reply_after(messages: Sequence[ChatMessage | None], index: int) -> str:
    """The companion's text after position ``index``, up to the next user message."""
    lines: list[str] = []
    for message in messages[index + 1 :]:
        if message is None:
            continue
        if message.role == "user":
            break
        lines.append(message.content)
    return "\n".join(lines)


def _open_record(turn: SimulatorTurn, *, day: int, turn_in_day: int, mood: Mood) -> TurnRecord:
    """The record of ``turn``; the companion's reply is filled in once it is known."""
    return TurnRecord(
        day=day,
        turn=turn_in_day,
        hidden=turn.hidden.as_dict() if turn.hidden else None,
        visible=turn.text,
        companion_reply="",
        mood=mood,
        polluted=turn.polluted,
        problems=turn.problems,
        forced=turn.forced,
        rules_applied=turn.rules_applied,
    )


def _copy(transcript: Transcript) -> Transcript:
    """A copy whose lists the caller may change."""
    return replace(
        transcript,
        turns=list(transcript.turns),
        notes=list(transcript.notes),
        skipped_days=list(transcript.skipped_days),
    )


class SimulatedUser:
    """A moodsona persona that writes the user's next message, given the conversation.

    Use one per conversation: it keeps the persona's hidden state from one
    :meth:`reply` to the next.

    Args:
        persona: The persona.
        user_model: The model that writes the persona's lines (the same name
            :func:`~moodsona.runner.run` uses).
        turns_per_day: Maximum persona messages per simulated day.
        days: Number of simulated days; :attr:`done` turns True after the last
            one. None (the default) sets no limit.
        rng_seed: Seed for overnight energy and typing noise.
        language: Language profile; None (the default) uses the persona's
            ``language`` (see :class:`~moodsona.simulator.Simulator`).
        safety_hook: Optional crisis detector (see :class:`~moodsona.simulator.Simulator`).
        config: Optional simulator settings.

    Raises:
        ValueError: ``turns_per_day`` or ``days`` is below 1.
        LookupError: No language profile covers the persona's language (raised
            here, before any model call).
    """

    def __init__(
        self,
        persona: PersonaSeed,
        user_model: LLMProvider,
        *,
        turns_per_day: int = 6,
        days: int | None = None,
        rng_seed: int = 0,
        language: LanguageProfile | None = None,
        safety_hook: SafetyHook | None = None,
        config: SimulatorConfig | None = None,
    ) -> None:
        if turns_per_day < 1:
            raise ValueError("turns_per_day must be at least 1")
        if days is not None and days < 1:
            raise ValueError("days must be at least 1, or None for no limit")
        self._simulator = Simulator(
            persona,
            user_model,
            language=language,
            safety_hook=safety_hook,
            config=config,
            rng=random.Random(rng_seed),  # noqa: S311 - simulation, not cryptography
        )
        self._turns_per_day = turns_per_day
        self._days = days
        #: Closed turns, written notes and skipped days so far.
        self._transcript = Transcript(persona.id, rng_seed, days=days, turns_per_day=turns_per_day)
        self._turn_in_day = 0
        #: Position in the history where today's conversation starts.
        self._day_start = 0
        #: Today's visible messages as of the last reply, its own message included.
        self._seen_today: list[ChatMessage] = []
        #: The latest turn, its companion reply still empty, and its position in the history.
        self._open: TurnRecord | None = None
        self._open_at = 0
        #: The open turn ended its day.
        self._day_over = False
        #: How long the next history must at least be (the last reply included).
        self._seen = 0
        self._done = False
        self._final: Transcript | None = None

    @property
    def done(self) -> bool:
        """True once the turn that ends the last day is written, or after :meth:`finish`.

        The last day is day ``days``, or an earlier one when the persona skips
        the days after it. With ``days=None`` only :meth:`finish` sets it.
        """
        return self._done

    @property
    def day(self) -> int:
        """The current simulated day, from 1 (0 before the first reply)."""
        return self._simulator.state.current_day

    @property
    def state(self) -> PersonaState:
        """The persona's hidden state (mood, memory log), live; read it, do not change it."""
        return self._simulator.state

    def reply(self, history: History) -> str:
        """Write the simulated user's next message.

        Args:
            history: The conversation so far: ``user`` messages are this
                simulated user's earlier messages, ``assistant`` messages the
                chatbot's. :class:`~moodsona.llm.ChatMessage` items and
                OpenAI-style mappings (``content`` a string or a list of
                ``{"type": "text", "text": ...}`` parts) are accepted; system
                and tool messages are not shown to the persona. Append the
                returned text to it as a ``user`` message.

        Returns:
            The visible message; the hidden state never leaves this object.

        Raises:
            RuntimeError: The conversation is over (:attr:`done`).
            ValueError: ``history`` is shorter than the one the previous call
                saw, with its reply: it is another conversation.
            TypeError: A history item is neither a ChatMessage nor a mapping.
            moodsona.llm.LLMError: The model failed. Nothing is lost: calling
                again with the same history retries.
        """
        if self._done:
            if self._final is not None:
                raise RuntimeError("finish() was called; this conversation is over")
            raise RuntimeError(
                f"all {self._days} days are simulated; call finish(history) for the transcript"
            )
        messages = _parse(history)
        self._check(messages)
        opened = self._open
        if opened is not None:
            day_over = self._day_over
            if day_over:
                # First: if the notes call fails, the turn stays open and a retry repeats it.
                self._end_day(messages)
            self._close(opened, messages)
            if day_over:
                skipped, day = self._days_ahead()
                self._transcript.skipped_days += skipped
                self._start_day(day, at=len(messages))
        elif self.day == 0:
            # Day 1 starts at the top, so the persona also sees an opening bot message.
            self._start_day(1, at=0)
        day, turn_in_day = self.day, self._turn_in_day + 1
        today = [m for m in messages[self._day_start :] if m is not None]
        turn = self._simulator.next_turn(today, day=day, turn_in_day=turn_in_day)
        mood = self._simulator.state.mood
        self._turn_in_day = turn_in_day
        self._open = _open_record(turn, day=day, turn_in_day=turn_in_day, mood=mood)
        self._open_at = len(messages)
        self._day_over = turn.ends_day or turn_in_day >= self._turns_per_day
        self._done = (
            self._day_over and self._days is not None and self._days_ahead()[1] > self._days
        )
        self._seen_today = [*today, ChatMessage("user", turn.text)]
        self._seen = len(messages) + 1
        return turn.text

    def finish(self, history: History = ()) -> Transcript:
        """End the conversation and return its transcript.

        Fills in the chatbot's reply to the last message and writes the open
        day's notes; after that :attr:`done` is True. Calling it again returns
        an equal transcript without calling the model.

        Args:
            history: The final conversation, to read the chatbot's last reply.
                Without it (or with an empty one) that reply is ``""`` and the
                notes read the day as the last :meth:`reply` saw it.

        Returns:
            The transcript, in the form :func:`moodsona.run` returns.

        Raises:
            ValueError: A non-empty ``history`` is shorter than the one the
                last reply saw.
            TypeError: A history item is neither a ChatMessage nor a mapping.
            moodsona.llm.LLMError: The notes call failed; calling again retries.
        """
        if self._final is None:
            messages = _parse(history)
            if messages:
                self._check(messages)
            opened = self._open
            if opened is not None:
                self._end_day(messages)
                self._close(opened, messages)
            if self._done:
                # As run() does: the days after the last one that the persona skips.
                self._transcript.skipped_days += self._days_ahead()[0]
            self._final = self.transcript()
            self._done = True
        return _copy(self._final)

    def transcript(self, history: History | None = None) -> Transcript:
        """The conversation so far, as a transcript; never calls the model.

        Args:
            history: The conversation so far, to fill in the chatbot's reply
                to the last message (``""`` without it or with an empty one).

        Returns:
            The transcript. The open day's notes are not in it yet. ``days`` is
            the ``days`` argument, or with ``days=None`` the last day so far
            (None before the first reply).

        Raises:
            ValueError: A non-empty ``history`` is shorter than the one the
                last reply saw.
            TypeError: A history item is neither a ChatMessage nor a mapping.
        """
        if self._final is not None:
            return _copy(self._final)
        messages = _parse(history or ())
        if messages:
            self._check(messages)
        so_far = self._transcript
        turns = list(so_far.turns)
        if self._open is not None:
            turns.append(self._closed(self._open, messages))
        days = self._days
        if days is None:
            days = max([t.day for t in turns] + so_far.skipped_days, default=None)
        return replace(_copy(so_far), turns=turns, days=days)

    def metrics(self, history: History | None = None) -> TranscriptMetrics:
        """The :mod:`moodsona.metrics` of :meth:`transcript`; never calls the model.

        Args:
            history: See :meth:`transcript`.

        Returns:
            The metrics. The agreement, ask-back and wrong-language rates use
            the simulator's language profile (None when language checks are off).
        """
        return compute_metrics(self.transcript(history), language=self._simulator.language)

    # -- bookkeeping ----------------------------------------------------------

    def _check(self, messages: Sequence[object]) -> None:
        if len(messages) < self._seen:
            raise ValueError(
                "the history is shorter than the one this simulated user has already seen; "
                "use one SimulatedUser per conversation"
            )

    def _start_day(self, day: int, *, at: int) -> None:
        self._simulator.start_day(day)
        self._turn_in_day = 0
        self._day_start = at

    def _end_day(self, messages: Sequence[ChatMessage | None]) -> None:
        """Write the current day's notes from its visible messages."""
        if messages:
            today = [m for m in messages[self._day_start :] if m is not None]
        else:
            today = self._seen_today
        note = self._simulator.end_day(today, day=self.day)
        if note is not None:
            self._transcript.notes.append(note)

    def _closed(self, opened: TurnRecord, messages: Sequence[ChatMessage | None]) -> TurnRecord:
        """The open turn with the companion's reply from ``messages`` filled in."""
        return replace(opened, companion_reply=_reply_after(messages, self._open_at))

    def _close(self, opened: TurnRecord, messages: Sequence[ChatMessage | None]) -> None:
        self._transcript.turns.append(self._closed(opened, messages))
        self._open = None

    def _days_ahead(self) -> tuple[list[int], int]:
        """The days after today the persona skips (up to ``days``), and the next it writes on."""
        skipped: list[int] = []
        day = self.day + 1
        while (self._days is None or day <= self._days) and self._simulator.is_skipped(day):
            skipped.append(day)
            day += 1
        return skipped, day
