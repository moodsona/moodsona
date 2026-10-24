"""A minimal multi-day loop: the simulated user talks to your companion.

Per day: the persona writes, the companion replies, until the persona ends the
day (``end_day`` / ``skip_tomorrow``) or ``turns_per_day`` is reached. Then the
persona writes its private notes for the next day. A day the persona decided
to skip has no messages at all. Nothing is stored except the returned
:class:`Transcript`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from moodsona.companion import Companion
from moodsona.llm.base import ChatMessage, LLMProvider
from moodsona.simulator import SafetyHook, Simulator, SimulatorConfig
from moodsona.state import DayNote, Mood, PersonaSeed, Stance

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One exchange: the persona's message and the companion's reply.

    Attributes:
        day: Day number, from 1.
        turn: Turn number within the day, from 1.
        hidden: The hidden block as a dict (None for a polluted turn).
        visible: The message the companion received.
        companion_reply: The companion's answer.
        mood: The persona's mood after this turn.
        polluted: True when the simulator fell back to a neutral line.
        problems: Rails that rejected earlier attempts of this turn.
        forced: A soft rail the accepted message still breaks, if any.
    """

    day: int
    turn: int
    hidden: dict[str, Any] | None
    visible: str
    companion_reply: str
    mood: Mood
    polluted: bool = False
    problems: tuple[str, ...] = ()
    forced: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (one line of :meth:`Transcript.to_jsonl`)."""
        return {
            "kind": "turn",
            "day": self.day,
            "turn": self.turn,
            "hidden": self.hidden,
            "visible": self.visible,
            "companion_reply": self.companion_reply,
            "state_after": {
                "emotion": self.mood.emotion,
                "energy": self.mood.energy,
                "stance": self.mood.stance.value,
            },
            "polluted": self.polluted,
            "problems": list(self.problems),
            "forced": self.forced,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnRecord:
        """Rebuild a record from :meth:`as_dict` output.

        Raises:
            KeyError: A field is missing.
            ValueError: A field has the wrong type or an out-of-range value.
        """
        hidden = data["hidden"]
        _require(hidden is None or isinstance(hidden, dict), "hidden must be an object or null")
        if hidden is not None:
            rules = hidden.get("rules_triggered", [])
            _require(
                isinstance(rules, list) and all(map(_is_int, rules)),
                "hidden.rules_triggered must be a list of integers",
            )
        problems, forced = data["problems"], data["forced"]
        _require(isinstance(data["polluted"], bool), "polluted must be true or false")
        _require(isinstance(problems, list), "problems must be a list")
        _require(forced is None or isinstance(forced, str), "forced must be a string or null")
        state = data["state_after"]
        return cls(
            day=int(data["day"]),
            turn=int(data["turn"]),
            hidden=hidden,
            visible=str(data["visible"]),
            companion_reply=str(data["companion_reply"]),
            mood=Mood(state["emotion"], state["energy"], Stance(state["stance"])),
            polluted=data["polluted"],
            problems=tuple(problems),
            forced=forced,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """The persona's mood after one turn."""

    day: int
    turn: int
    emotion: int
    energy: int
    stance: str


@dataclass
class Transcript:
    """Everything one run produced.

    Attributes:
        persona_id: The simulated persona.
        rng_seed: Seed of the run's random source.
        turns: One record per exchange, in order.
        notes: The persona's end-of-day notes, in order.
        skipped_days: Days the persona chose not to write.
        days: Simulated days requested (None when unknown).
        turns_per_day: Maximum persona messages per day (None when unknown).
    """

    persona_id: str
    rng_seed: int
    turns: list[TurnRecord] = field(default_factory=list)
    notes: list[DayNote] = field(default_factory=list)
    skipped_days: list[int] = field(default_factory=list)
    days: int | None = None
    turns_per_day: int | None = None

    def trajectory(self) -> list[TrajectoryPoint]:
        """Emotion, energy and stance after every turn."""
        return [
            TrajectoryPoint(t.day, t.turn, t.mood.emotion, t.mood.energy, t.mood.stance.value)
            for t in self.turns
        ]

    def visible_history(self) -> list[ChatMessage]:
        """The conversation exactly as the companion saw it."""
        history: list[ChatMessage] = []
        for t in self.turns:
            history += [ChatMessage("user", t.visible), ChatMessage("assistant", t.companion_reply)]
        return history

    def to_jsonl(self, path: str | PathLike[str]) -> None:
        """Write one JSON object per line.

        Every line has a ``kind`` field. The first line is ``"run"`` (persona id,
        random seed, days, turns per day); then one ``"turn"`` line per exchange,
        one ``"day_notes"`` line per day's notes and one ``"skipped_day"`` line
        per day the persona chose not to write.
        """
        lines: list[dict[str, Any]] = [
            {
                "kind": "run",
                "persona_id": self.persona_id,
                "rng_seed": self.rng_seed,
                "days": self.days,
                "turns_per_day": self.turns_per_day,
            }
        ]
        lines += [t.as_dict() for t in self.turns]
        lines += [
            {"kind": "day_notes", "day": n.day, "notes": list(n.lines), "ok": n.ok}
            for n in self.notes
        ]
        lines += [{"kind": "skipped_day", "day": day} for day in self.skipped_days]
        text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | PathLike[str]) -> Transcript:
        """Read a transcript written by :meth:`to_jsonl`.

        Blank lines and lines of an unknown ``kind`` are skipped, so files from
        newer versions with extra line kinds still load. One file holds one run.

        Args:
            path: A JSONL file.

        Returns:
            The transcript.

        Raises:
            OSError: The file cannot be read.
            ValueError: The first line is not a ``run`` line, there is a second
                ``run`` line, or a line is not valid JSON, misses a field or has
                a field of the wrong type (the message names the line).
        """
        transcript: Transcript | None = None
        text = Path(path).read_text(encoding="utf-8")
        # Split on "\n" only: to_jsonl keeps characters such as U+2028 unescaped,
        # and str.splitlines() would break a line inside a JSON string there.
        for number, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                _require(isinstance(data, dict), "not a JSON object")
                kind = data["kind"]
                if transcript is None:
                    _require(kind == "run", f"the first line must be a 'run' line, got {kind!r}")
                    transcript = cls(
                        persona_id=str(data["persona_id"]),
                        rng_seed=int(data["rng_seed"]),
                        days=data.get("days"),
                        turns_per_day=data.get("turns_per_day"),
                    )
                elif kind == "run":
                    raise ValueError("a second 'run' line; one file holds one run")
                elif kind == "turn":
                    transcript.turns.append(TurnRecord.from_dict(data))
                elif kind == "day_notes":
                    transcript.notes.append(_note_from_dict(data))
                elif kind == "skipped_day":
                    transcript.skipped_days.append(int(data["day"]))
            except (KeyError, TypeError, ValueError) as exc:  # JSONDecodeError is a ValueError
                raise ValueError(f"{path}:{number}: {exc}") from exc
        if transcript is None:
            raise ValueError(f"{path}: no 'run' line")
        return transcript


def _require(condition: bool, problem: str) -> None:
    if not condition:
        raise ValueError(problem)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _note_from_dict(data: dict[str, Any]) -> DayNote:
    lines = data["notes"]
    _require(
        isinstance(lines, list) and all(isinstance(x, str) for x in lines),
        "notes must be a list of strings",
    )
    _require(isinstance(data["ok"], bool), "ok must be true or false")
    return DayNote(int(data["day"]), tuple(lines), data["ok"])


def run(
    seed: PersonaSeed,
    provider: LLMProvider,
    companion: Companion,
    days: int,
    turns_per_day: int,
    rng_seed: int = 0,
    *,
    language: LanguageProfile | None = None,
    safety_hook: SafetyHook | None = None,
    config: SimulatorConfig | None = None,
) -> Transcript:
    """Simulate ``days`` days of conversation between a persona and a companion.

    Args:
        seed: The persona.
        provider: The model that writes the persona's lines.
        companion: The chatbot under test.
        days: Number of simulated days.
        turns_per_day: Maximum persona messages per day.
        rng_seed: Seed for overnight energy and typing noise.
        language: Language profile; None (the default) uses the persona's
            ``language`` (see :class:`~moodsona.simulator.Simulator`).
        safety_hook: Optional crisis detector (see :class:`~moodsona.simulator.Simulator`).
        config: Optional simulator settings.

    Returns:
        The transcript with per-turn records and day notes.

    Raises:
        ValueError: ``days`` or ``turns_per_day`` is below 1.
        LookupError: No language profile covers the persona's language (raised
            before the first model call).
        moodsona.llm.LLMError: The provider failed.
    """
    if days < 1 or turns_per_day < 1:
        raise ValueError("days and turns_per_day must be at least 1")
    simulator = Simulator(
        seed,
        provider,
        language=language,
        safety_hook=safety_hook,
        config=config,
        rng=random.Random(rng_seed),  # noqa: S311 - simulation, not cryptography
    )
    transcript = Transcript(
        persona_id=seed.id, rng_seed=rng_seed, days=days, turns_per_day=turns_per_day
    )
    history: list[ChatMessage] = []
    for day in range(1, days + 1):
        if simulator.is_skipped(day):
            transcript.skipped_days.append(day)
            continue
        simulator.start_day(day)
        today: list[ChatMessage] = []
        for turn_in_day in range(1, turns_per_day + 1):
            turn = simulator.next_turn(today, day=day, turn_in_day=turn_in_day)
            user = ChatMessage("user", turn.text)
            history.append(user)
            reply = companion.reply(tuple(history))
            answer = ChatMessage("assistant", reply)
            history.append(answer)
            today += [user, answer]
            transcript.turns.append(
                TurnRecord(
                    day=day,
                    turn=turn_in_day,
                    hidden=turn.hidden.as_dict() if turn.hidden else None,
                    visible=turn.text,
                    companion_reply=reply,
                    mood=simulator.state.mood,
                    polluted=turn.polluted,
                    problems=turn.problems,
                    forced=turn.forced,
                )
            )
            if turn.ends_day:
                break
        note = simulator.end_day(today, day=day)
        if note is not None:
            transcript.notes.append(note)
    return transcript
