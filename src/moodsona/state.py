"""Persona seed and mutable persona state.

Everything numeric here is plain code, not the model: clamping to 0-100, the
reaction-rule deltas, the per-turn step limit and the overnight rule. The
functions are pure so they can be property-tested.

Who decides what:

* The model reads the companion's last message, says which reaction rules it
  triggered, and proposes new emotion and energy values (in the hidden block).
* The code applies the triggered rules. For each dimension (emotion, energy)
  that a triggered rule gives a delta for, the rule wins: the new value is the
  old value plus the sum of those deltas. For a dimension no triggered rule
  covers, the model's proposal is used, but it may move at most
  ``max_free_step`` points per turn. Stance changes only through a rule's
  ``stance_to`` (the last triggered rule wins) or through the end-of-day notes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

SCORE_MIN = 0
SCORE_MAX = 100
#: Largest change of emotion or energy in one turn that no reaction rule covers.
MAX_FREE_STEP = 20
#: Default spread of the morning energy around the baseline (+/- points).
DEFAULT_OVERNIGHT_JITTER = 10


class Stance(StrEnum):
    """Overall attitude toward the companion (the schema's ``stance`` enum)."""

    OPEN = "open"
    CURIOUS = "curious"
    GUARDED = "guarded"
    SKEPTICAL = "skeptical"
    WITHDRAWN = "withdrawn"
    HOSTILE = "hostile"


class Intent(StrEnum):
    """What the simulated user does with one message."""

    REPLY = "reply"
    SHORT_REPLY = "short_reply"
    CHANGE_TOPIC = "change_topic"
    TEST_BOUNDARY = "test_boundary"
    END_DAY = "end_day"
    SKIP_TOMORROW = "skip_tomorrow"


DAY_ENDING_INTENTS = frozenset({Intent.END_DAY, Intent.SKIP_TOMORROW})


def clamp_score(value: float) -> int:
    """Round ``value`` and clamp it to the 0-100 scale."""
    return max(SCORE_MIN, min(SCORE_MAX, round(value)))


_PRONOUN_FORMS = {
    "she/her": ("her", "her"),
    "he/him": ("him", "his"),
    "they/them": ("them", "their"),
}
DEFAULT_PRONOUNS = "they/them"


@dataclass(frozen=True, slots=True)
class Pronouns:
    """Pronouns used for the persona in prompts.

    Attributes:
        label: The schema value, e.g. ``"she/her"``.
        object: Object form ("her", "him", "them").
        possessive: Possessive form ("her", "his", "their").
    """

    label: str
    object: str
    possessive: str

    @classmethod
    def from_label(cls, label: str) -> Pronouns:
        """Build pronouns from a schema value such as ``"they/them"``.

        Raises:
            ValueError: The label is not one of the supported values.
        """
        forms = _PRONOUN_FORMS.get(label)
        if forms is None:
            raise ValueError(f"unsupported pronouns {label!r}; use one of {sorted(_PRONOUN_FORMS)}")
        return cls(label, *forms)


@dataclass(frozen=True, slots=True)
class Mood:
    """Emotion, energy and stance at one moment. Always within range.

    Attributes:
        emotion: Valence, 0 = very distressed, 50 = neutral, 100 = very positive.
        energy: Engagement, 0 = about to leave, 100 = highly engaged.
        stance: Attitude toward the companion.
    """

    emotion: int
    energy: int
    stance: Stance

    def __post_init__(self) -> None:
        for name in ("emotion", "energy"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if not SCORE_MIN <= value <= SCORE_MAX:
                raise ValueError(f"{name} must be in [0, 100], got {value}")

    @classmethod
    def clamped(cls, emotion: float, energy: float, stance: Stance) -> Mood:
        """Build a mood, rounding and clamping out-of-range values."""
        return cls(clamp_score(emotion), clamp_score(energy), stance)

    def shifted(
        self, *, emotion: float = 0, energy: float = 0, stance: Stance | None = None
    ) -> Mood:
        """Return a copy moved by the given deltas (clamped), with an optional new stance."""
        return Mood.clamped(self.emotion + emotion, self.energy + energy, stance or self.stance)


@dataclass(frozen=True, slots=True)
class ReactionRule:
    """When the companion does X, the persona reacts with Y and its state shifts.

    A delta of ``None`` means the rule says nothing about that dimension.
    """

    when: str
    then: str
    emotion_delta: int | None = None
    energy_delta: int | None = None
    stance_to: Stance | None = None


@dataclass(frozen=True, slots=True)
class PersonaSeed:
    """Everything fixed about one persona, built from a persona file.

    Attributes:
        id: Kebab-case identifier.
        display_name: Short human-readable name.
        language: Language code the persona writes in.
        hidden_goal: What the persona wants; never shown to the companion.
        reaction_rules: Ordered reaction rules (numbered from 1 in prompts).
        initial_state: Mood at the start of the first day.
        baseline: Resting mood the persona drifts back to overnight.
        memory_days_visible: How many earlier days the persona remembers.
        description: Optional catalog summary; part of the simulator's persona card.
        pronouns: Pronouns used for the persona in prompts.
        tags: Catalog tags.
        overnight_energy_jitter: Morning energy spread around the baseline.
    """

    id: str
    display_name: str
    language: str
    hidden_goal: str
    reaction_rules: tuple[ReactionRule, ...]
    initial_state: Mood
    baseline: Mood
    memory_days_visible: int
    description: str = ""
    pronouns: Pronouns = field(default_factory=lambda: Pronouns.from_label(DEFAULT_PRONOUNS))
    tags: tuple[str, ...] = ()
    overnight_energy_jitter: int = DEFAULT_OVERNIGHT_JITTER


def react(
    mood: Mood,
    *,
    proposed_emotion: int,
    proposed_energy: int,
    triggered: Sequence[ReactionRule] = (),
    max_free_step: int = MAX_FREE_STEP,
) -> Mood:
    """Apply one turn's update to ``mood`` (see the module docstring for the rule).

    Args:
        mood: The mood before this turn.
        proposed_emotion: The model's proposed emotion after the turn.
        proposed_energy: The model's proposed energy after the turn.
        triggered: Reaction rules the companion's last message triggered.
        max_free_step: Largest move allowed for a dimension no rule covers.

    Returns:
        The mood after the turn, always within range.
    """

    def free(old: int, proposed: int) -> int:
        return old + max(-max_free_step, min(max_free_step, proposed - old))

    emotion_deltas = [r.emotion_delta for r in triggered if r.emotion_delta is not None]
    energy_deltas = [r.energy_delta for r in triggered if r.energy_delta is not None]
    stances = [r.stance_to for r in triggered if r.stance_to is not None]
    if emotion_deltas:
        emotion = mood.emotion + sum(emotion_deltas)
    else:
        emotion = free(mood.emotion, proposed_emotion)
    if energy_deltas:
        energy = mood.energy + sum(energy_deltas)
    else:
        energy = free(mood.energy, proposed_energy)
    return Mood.clamped(emotion, energy, stances[-1] if stances else mood.stance)


def overnight(mood: Mood, baseline: Mood, *, energy_offset: int, remembers: bool) -> Mood:
    """The morning mood after a night.

    Emotion moves halfway back to the baseline. Energy restarts at the
    baseline plus ``energy_offset`` (drawn by the caller from its seeded RNG).
    Stance is kept when the persona remembers earlier days; otherwise it
    resets to the baseline stance.
    """
    emotion = baseline.emotion + 0.5 * (mood.emotion - baseline.emotion)
    stance = mood.stance if remembers else baseline.stance
    return Mood.clamped(emotion, baseline.energy + energy_offset, stance)


@dataclass(frozen=True, slots=True)
class DayNote:
    """The persona's own notes about one day.

    Attributes:
        day: The day the notes are about.
        lines: Short first-person lines; empty when the notes failed.
        ok: False when the model never produced usable notes for the day.
    """

    day: int
    lines: tuple[str, ...]
    ok: bool = True


@dataclass
class PersonaState:
    """What changes during a simulation: mood, day bookkeeping and memory log."""

    mood: Mood
    current_day: int = 0
    last_active_day: int = 0
    memory_log: list[DayNote] = field(default_factory=list)
    skipped_days: set[int] = field(default_factory=set)

    @classmethod
    def initial(cls, seed: PersonaSeed) -> PersonaState:
        """The state before the first day."""
        return cls(mood=seed.initial_state)

    def begin_day(self, seed: PersonaSeed, *, day: int, energy_offset: int = 0) -> None:
        """Apply the overnight rule once per day, before the day's first message.

        The first active day starts from the initial state unchanged.
        """
        if day <= self.current_day:
            return
        if self.current_day > 0:
            self.mood = overnight(
                self.mood,
                seed.baseline,
                energy_offset=energy_offset,
                remembers=seed.memory_days_visible > 0,
            )
        self.current_day = day

    def days_since_last_talk(self, day: int) -> int | None:
        """Days since the last day with a message, or None before the first one."""
        return None if self.last_active_day == 0 else max(0, day - self.last_active_day)

    def visible_memory(self, *, day: int, days_visible: int) -> list[DayNote]:
        """Notes with any lines from the ``days_visible`` days before ``day``."""
        return [
            note for note in self.memory_log if day - days_visible <= note.day < day and note.lines
        ]

    def remember(self, note: DayNote, *, stance: Stance | None = None) -> None:
        """Store a day's notes (once per day) and the stance they end with."""
        if any(existing.day == note.day for existing in self.memory_log):
            return
        self.memory_log.append(note)
        if stance is not None:
            self.mood = replace(self.mood, stance=stance)
