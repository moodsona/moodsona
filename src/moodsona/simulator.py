"""The simulated user: one turn at a time, with rails.

Each turn the model writes a hidden block (inner thought, emotion, energy,
intent, triggered reaction rules) and then the visible message. Only the
visible message is meant for the companion. Before a message is accepted it
passes these rails; a failing attempt is sent back to the model with a short
repair line:

Hard rails (the attempt is never used):
    ``hidden_state_invalid``, ``empty_output``, ``identity_claim``,
    ``task_refusal``, ``meta_content``, ``wrong_language`` (needs a language
    profile) and ``crisis_scope`` (needs a ``safety_hook``).
Soft rails (after the last attempt the best soft-failing message is used and
the turn is marked ``forced``):
    ``too_long``, ``own_repeat``, and the run-wide ceilings
    ``agreement_ceiling`` / ``ask_back_ceiling`` (need a language profile).

When no attempt passes the hard rails, the turn falls back to a neutral line,
the persona state is left unchanged, and the turn is marked ``polluted`` so
analyses can exclude it.

The language profile comes from the persona's ``language`` field unless one is
passed explicitly; ``SimulatorConfig(language_checks=False)`` turns every
language-dependent rail and the typing noise off.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from moodsona.checks import find_role_break, repeats_earlier
from moodsona.lang import get_profile
from moodsona.llm.base import ChatMessage, LLMProvider
from moodsona.messages import build_notes_messages, build_turn_messages
from moodsona.parse import HiddenBlock, ParseError, parse_day_notes, parse_turn
from moodsona.prompts import REPAIRS
from moodsona.state import (
    DAY_ENDING_INTENTS,
    MAX_FREE_STEP,
    DayNote,
    Intent,
    PersonaSeed,
    PersonaState,
    react,
)

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile

#: Stand-in for an empty model output when it is replayed in a repair round
#: (APIs such as Anthropic's reject an empty assistant message).
EMPTY_OUTPUT_PLACEHOLDER = "(empty)"

#: Rails whose last failing message may still be used ("forced") after all attempts.
SOFT_PROBLEMS = frozenset({"too_long", "own_repeat", "agreement_ceiling", "ask_back_ceiling"})

SafetyHook = Callable[[str], bool]
"""Returns True when a text must not enter the transcript (e.g. a crisis detector)."""


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """Tunable numbers of the simulator. The defaults are uncalibrated starting points."""

    max_attempts: int = 3
    notes_attempts: int = 2
    turn_temperature: float = 0.9
    notes_temperature: float = 0.7
    turn_max_tokens: int = 4000
    notes_max_tokens: int = 4000
    #: How many of today's visible messages the model sees.
    window_messages: int = 12
    max_message_chars: int = 600
    own_repeat_jaccard: float = 0.7
    own_repeat_min_words: int = 3
    #: Run-wide ceilings, read over at least ``ceiling_min_lines`` lines.
    agreement_max_share: float = 0.10
    ask_back_max_share: float = 0.10
    ceiling_min_lines: int = 10
    #: Typing noise level in [0, 1] passed to the language profile; 0 = off.
    typing_noise: float = 0.0
    max_free_step: int = MAX_FREE_STEP
    fallback_message: str = "ok"
    #: False turns off everything that needs a language profile: the
    #: ``wrong_language`` rail, the agreement / ask-back ceilings and typing noise.
    language_checks: bool = True


@dataclass(frozen=True, slots=True)
class SimulatorTurn:
    """One simulated-user turn.

    Attributes:
        text: The visible message sent to the companion (after typing noise).
        clean_text: The same message before typing noise.
        hidden: The hidden block, or None for a polluted turn.
        rules_triggered: Valid 1-based reaction-rule numbers that were applied.
        attempts: Model calls used.
        problems: The rail that rejected each failed attempt, in order.
        polluted: True when no attempt passed and the fallback line was used.
        forced: The soft rail a forced message still breaks, if any.
    """

    text: str
    clean_text: str
    hidden: HiddenBlock | None
    rules_triggered: tuple[int, ...]
    attempts: int
    problems: tuple[str, ...]
    polluted: bool = False
    forced: str | None = None

    @property
    def ends_day(self) -> bool:
        """True when the persona chose to end today's conversation."""
        return self.hidden is not None and self.hidden.intent in DAY_ENDING_INTENTS


@dataclass
class _Lines:
    """Counts of the run's accepted clean lines, for the run-wide ceilings."""

    total: int = 0
    agreements: int = 0
    ask_backs: int = 0


def _never_flags(text: str) -> bool:
    del text
    return False


def _resolve_language(
    seed: PersonaSeed, language: LanguageProfile | None, enabled: bool
) -> LanguageProfile | None:
    """The profile to use: none when disabled, else the given one or the persona's."""
    if not enabled:
        return None
    if language is not None:
        return language
    try:
        return get_profile(seed.language)
    except LookupError as exc:
        raise LookupError(
            f"persona {seed.id!r}: {exc} Pass language=... or "
            "SimulatorConfig(language_checks=False)."
        ) from exc


class Simulator:
    """A simulated user built from a persona seed.

    Args:
        seed: The persona.
        provider: The model that writes the persona's lines.
        language: Language profile for the wrong-language check, the
            agreement / ask-back ceilings and typing noise. None (the default)
            looks it up from ``seed.language`` with
            :func:`moodsona.lang.get_profile`. Ignored when
            ``config.language_checks`` is False.
        safety_hook: Called on every candidate message, inner thought and day
            note; returning True rejects it as ``crisis_scope``. This is where a
            crisis detector plugs in. The default (None) never flags.
        config: Tunable numbers.
        rng: Random source for overnight energy and typing noise.

    Attributes:
        state: The persona's current state.
        language: The language profile in use, or None when
            ``config.language_checks`` is False.

    Raises:
        LookupError: ``language`` is None, language checks are on, and no
            profile covers ``seed.language``.
    """

    def __init__(
        self,
        seed: PersonaSeed,
        provider: LLMProvider,
        *,
        language: LanguageProfile | None = None,
        safety_hook: SafetyHook | None = None,
        config: SimulatorConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.seed = seed
        self.provider = provider
        self.config = config or SimulatorConfig()
        self.language = _resolve_language(seed, language, self.config.language_checks)
        self.safety_hook: SafetyHook = safety_hook or _never_flags
        self.rng = rng or random.Random(0)  # noqa: S311 - simulation, not cryptography
        self.state = PersonaState.initial(seed)
        self._lines = _Lines()
        self._today: list[str] = []

    # -- days -------------------------------------------------------------

    def is_skipped(self, day: int) -> bool:
        """True when the persona decided the day before not to write on ``day``."""
        return day in self.state.skipped_days

    def start_day(self, day: int) -> None:
        """Begin ``day``: apply the overnight rule (from the second day on)."""
        jitter = self.seed.overnight_energy_jitter
        offset = self.rng.randint(-jitter, jitter) if jitter else 0
        self.state.begin_day(self.seed, day=day, energy_offset=offset)
        self._today = []

    def end_day(self, today: Sequence[ChatMessage], *, day: int) -> DayNote | None:
        """Ask the persona for its notes about ``day`` and store them in its memory.

        Returns:
            The stored note (``ok=False`` with no lines when every attempt
            failed), or None when the persona has no memory between days.
        """
        if self.seed.memory_days_visible == 0:
            return None
        messages = build_notes_messages(self.seed, self.state, today, day=day)
        for _ in range(self.config.notes_attempts):
            response = self.provider.complete(
                messages,
                temperature=self.config.notes_temperature,
                max_tokens=self.config.notes_max_tokens,
            )
            try:
                lines, stance = parse_day_notes(response.content)
            except ParseError:
                continue
            joined = "\n".join(lines)
            if find_role_break(joined) is not None or self.safety_hook(joined):
                continue
            note = DayNote(day, lines)
            self.state.remember(note, stance=stance)
            return note
        note = DayNote(day, (), ok=False)
        self.state.remember(note)
        return note

    # -- turns ------------------------------------------------------------

    def next_turn(
        self, today: Sequence[ChatMessage], *, day: int, turn_in_day: int
    ) -> SimulatorTurn:
        """Write the persona's next message and update its state.

        Args:
            today: Visible messages of the current day so far.
            day: Day number (call :meth:`start_day` first).
            turn_in_day: Turn number within the day, starting at 1.

        Returns:
            The turn; ``turn.text`` is what the companion should receive.
        """
        messages = build_turn_messages(
            self.seed,
            self.state,
            today,
            day=day,
            turn_in_day=turn_in_day,
            window=self.config.window_messages,
        )
        problems: list[str] = []
        soft: tuple[HiddenBlock, str, str] | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            response = self.provider.complete(
                messages,
                temperature=self.config.turn_temperature,
                max_tokens=self.config.turn_max_tokens,
            )
            checked = self._check(response.content)
            if checked is None:
                problem = "hidden_state_invalid"
            else:
                hidden, message, found = checked
                if found is None:
                    return self._accept(hidden, message, day, attempt, problems, None)
                problem = found
                if problem in SOFT_PROBLEMS:
                    soft = (hidden, message, problem)
            problems.append(problem)
            messages = [
                *messages,
                ChatMessage("assistant", response.content.strip() or EMPTY_OUTPUT_PLACEHOLDER),
                ChatMessage("user", REPAIRS[problem]),
            ]
        if soft is not None:
            hidden, message, problem = soft
            if problem == "too_long":
                message = message[: self.config.max_message_chars].rstrip()
            return self._accept(hidden, message, day, len(problems), problems, problem)
        fallback = self.config.fallback_message
        return SimulatorTurn(fallback, fallback, None, (), len(problems), tuple(problems), True)

    def _check(self, raw: str) -> tuple[HiddenBlock, str, str | None] | None:
        """Parse one output and run the rails; None when it cannot be parsed."""
        try:
            hidden, message = parse_turn(raw)
        except ParseError:
            return None
        return hidden, message, self._hard_problem(hidden, message) or self._soft_problem(message)

    def _hard_problem(self, hidden: HiddenBlock, message: str) -> str | None:
        problem = find_role_break(message)
        if problem is not None:
            return problem
        if self.language is not None and self.language.is_wrong_language(message):
            return "wrong_language"
        if self.safety_hook(message) or self.safety_hook(hidden.inner_thought):
            return "crisis_scope"
        return None

    def _soft_problem(self, message: str) -> str | None:
        cfg = self.config
        if len(message) > cfg.max_message_chars:
            return "too_long"
        if repeats_earlier(
            message,
            self._today,
            threshold=cfg.own_repeat_jaccard,
            min_words=cfg.own_repeat_min_words,
        ):
            return "own_repeat"
        if self.language is None:
            return None
        if self.language.is_agreement(message) and self._over_ceiling(
            self._lines.agreements, cfg.agreement_max_share
        ):
            return "agreement_ceiling"
        if self.language.is_ask_back(message) and self._over_ceiling(
            self._lines.ask_backs, cfg.ask_back_max_share
        ):
            return "ask_back_ceiling"
        return None

    def _over_ceiling(self, count: int, share: float) -> bool:
        """Would one more such line push its run-wide share over ``share``?"""
        lines = max(self._lines.total + 1, self.config.ceiling_min_lines)
        return (count + 1) / lines > share + 1e-9

    def _accept(
        self,
        hidden: HiddenBlock,
        message: str,
        day: int,
        attempts: int,
        problems: list[str],
        forced: str | None,
    ) -> SimulatorTurn:
        rules = self.seed.reaction_rules
        valid = tuple(n for n in hidden.rules_triggered if n <= len(rules))
        self.state.mood = react(
            self.state.mood,
            proposed_emotion=hidden.emotion,
            proposed_energy=hidden.energy,
            triggered=[rules[n - 1] for n in valid],
            max_free_step=self.config.max_free_step,
        )
        self.state.last_active_day = day
        if hidden.intent == Intent.SKIP_TOMORROW:
            self.state.skipped_days.add(day + 1)
        self._remember_line(message)
        text = message
        if self.language is not None and self.config.typing_noise > 0:
            text = self.language.add_typing_noise(message, self.rng, self.config.typing_noise)
        return SimulatorTurn(
            text or message, message, hidden, valid, attempts, tuple(problems), False, forced
        )

    def _remember_line(self, message: str) -> None:
        self._today.append(message)
        self._lines.total += 1
        if self.language is not None:
            self._lines.agreements += self.language.is_agreement(message)
            self._lines.ask_backs += self.language.is_ask_back(message)
