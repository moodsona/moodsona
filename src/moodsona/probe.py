"""Rule probes: check, rule by rule, that a persona's reaction rules can fire.

A persona's reaction rules are data: the user model reads the companion's last
message and names the rules it triggered, and the code applies their deltas.
A rule the user model never recognises does nothing, and a full simulation is
an expensive way to find that out. A probe is a short scripted situation: a
few visible messages of context, then one companion reply that does (or
deliberately does not do) what a rule describes. Running it takes one user-model
turn with the real simulator (same prompt, same rails) and reads which rules
were applied.

Probe files are YAML, one per persona, kept apart from the persona (the
persona schema stays as it is). The bundled ones live in the repository's
``probes/`` directory, ``probes/<persona id>.yaml``, and ship in the wheel as
``moodsona/probes/``::

    persona: grieving-after-loss
    probes:
      - id: silver-lining              # kebab-case, unique in the file
        note: optional, for humans
        day: 2                         # optional, default 1
        state: {emotion: 25}           # optional; overrides the initial state
        memory:                        # optional: the persona's notes of earlier days
          - day: 1
            notes: ["told it about her garden"]
        context:                       # today's visible messages before the reply
          - persona: she would have been 47 today   # the last one is the persona's
        reply: She is in a better place now.        # the companion reply under test
        fires: [1]                     # rules that must fire (1-based)
        not_fires: [3]                 # rules that must not fire

On day 1 the context may open with a companion message; on later days it
starts with the persona's, as a real run's day does.

Verdicts per rule (``k/n`` counts repeats; a repeat whose turn fell back to the
neutral line is left out): a rule with probes that expect it to fire must fire
in at least ``threshold`` of those repeats (``weak`` below, ``never`` at 0); it
may fire in at most ``1 - threshold`` of the repeats of probes that list it
under ``not_fires`` (``overfires`` above). ``no data`` means the rule has
probes, but none of their repeats produced a turn. A rule no probe expects to
fire is ``untested``: reported, but not a failure. The threshold is an exact
fraction (default 2/3, so 2 of 3 repeats pass), and a run also fails when a
probe has no valid repeat at all.

Cost: every probe repeat is one user-model call, plus up to
``SimulatorConfig.max_attempts - 1`` more when a rail rejects the output.
"""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from importlib import resources
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, TypeGuard

import yaml

from moodsona.llm.base import ChatMessage, LLMProvider, Role
from moodsona.personas import load_bundled_persona
from moodsona.simulator import SafetyHook, Simulator, SimulatorConfig
from moodsona.state import DayNote, Mood, PersonaSeed, Stance
from moodsona.validation import CONTACT_PATTERNS, UniqueKeyLoader, iter_files, looks_like_phone

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile

__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_THRESHOLD",
    "PROBE_SUFFIX",
    "Probe",
    "ProbeError",
    "ProbeReport",
    "ProbeResult",
    "ProbeSet",
    "RuleVerdict",
    "Verdict",
    "as_threshold",
    "bundled_probes_dir",
    "bundled_probes_path",
    "check_main",
    "format_coverage",
    "format_report",
    "format_result",
    "judge",
    "load_probes",
    "max_model_calls",
    "probe_once",
    "probes_from_mapping",
    "run_probes",
]

#: How often each probe is run by default; model output varies between calls.
DEFAULT_REPEATS = 3
#: Share of a rule's repeats it must fire in (and at most ``1 -`` this on ``not_fires``).
DEFAULT_THRESHOLD = Fraction(2, 3)
#: File suffix of probe files: ``<persona id>.yaml``.
PROBE_SUFFIX = ".yaml"

_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_TOP_KEYS = frozenset({"persona", "probes"})
_PROBE_KEYS = frozenset(
    {"id", "note", "day", "state", "memory", "context", "reply", "fires", "not_fires"}
)
_SPEAKERS: dict[str, Role] = {"persona": "user", "companion": "assistant"}
_STANCES = frozenset(s.value for s in Stance)
_WHEN_WIDTH = 60


class ProbeError(ValueError):
    """A probe file cannot be read or does not fit its persona.

    Attributes:
        problems: One message per problem found.
    """

    def __init__(self, source: str, problems: list[str]) -> None:
        super().__init__(f"{source}: " + "; ".join(problems))
        self.problems = problems


def as_threshold(value: Fraction | float | str) -> Fraction:
    """A threshold as an exact fraction in (0, 1].

    Args:
        value: A fraction, an int, or a string such as ``"2/3"`` or ``"0.8"``
            (read exactly); a float is read as the nearest fraction with a
            denominator of at most 1000, so ``2 / 3`` becomes 2/3.

    Raises:
        ValueError: The value cannot be read or is outside (0, 1].
    """
    if isinstance(value, float):
        fraction = Fraction(value).limit_denominator(1000)
    else:
        try:
            fraction = Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"threshold must be a fraction such as 2/3, got {value!r}") from exc
    if not 0 < fraction <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {fraction}")
    return fraction


@dataclass(frozen=True, slots=True)
class Probe:
    """One scripted situation and the rules it must (not) trigger.

    Attributes:
        id: Kebab-case name, unique in its file.
        context: Today's visible messages before the reply under test
            (``user`` = the persona, ``assistant`` = the companion); the last
            one is the persona's.
        reply: The companion reply under test.
        state: The persona's mood when the reply arrives.
        fires: Rule numbers (1-based) that must fire.
        not_fires: Rule numbers that must not fire.
        day: The simulated day of the conversation.
        memory: The persona's notes from earlier days.
        note: Free text for humans.
    """

    id: str
    context: tuple[ChatMessage, ...]
    reply: str
    state: Mood
    fires: tuple[int, ...] = ()
    not_fires: tuple[int, ...] = ()
    day: int = 1
    memory: tuple[DayNote, ...] = ()
    note: str = ""

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Today's visible messages as the persona sees them: the context, then the reply."""
        return (*self.context, ChatMessage("assistant", self.reply, day=self.day))

    @property
    def own_lines(self) -> tuple[str, ...]:
        """The persona's own messages in the context, in order."""
        return tuple(m.content for m in self.context if m.role == "user")

    @property
    def turn_in_day(self) -> int:
        """The turn the persona writes in reply: one after its last message in the context."""
        return len(self.own_lines) + 1


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """The probes of one persona.

    Attributes:
        persona_id: The persona the probes are written for.
        probes: The probes, in file order.
        source: Where they came from (a path, for messages).
    """

    persona_id: str
    probes: tuple[Probe, ...]
    source: str = "probes"

    def untested(self, rule_count: int) -> tuple[int, ...]:
        """Rule numbers (1 to ``rule_count``) that no probe expects to fire."""
        covered = {n for p in self.probes for n in p.fires}
        return tuple(n for n in range(1, rule_count + 1) if n not in covered)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The repeats of one probe.

    Attributes:
        probe: The probe.
        fired: Per repeat, the rule numbers that were applied; None for a
            repeat whose turn fell back to the neutral line (no rule data).
    """

    probe: Probe
    fired: tuple[frozenset[int] | None, ...]

    @property
    def valid(self) -> int:
        """Repeats that produced a turn."""
        return sum(f is not None for f in self.fired)

    @property
    def polluted(self) -> int:
        """Repeats that fell back to the neutral line."""
        return len(self.fired) - self.valid

    def count(self, rule: int) -> int:
        """In how many repeats ``rule`` fired."""
        return sum(f is not None and rule in f for f in self.fired)


class Verdict(StrEnum):
    """How one rule did across its probes."""

    OK = "ok"
    WEAK = "weak"
    NEVER = "never"
    NO_DATA = "no data"
    OVERFIRES = "overfires"
    UNTESTED = "untested"


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """One reaction rule across all probes.

    Attributes:
        rule: The 1-based rule number.
        when: The rule's ``when`` text.
        probes: How many probes expect it to fire.
        fired: Repeats of those probes in which it fired.
        trials: Repeats of those probes that produced a turn.
        false_probes: How many probes list it under ``not_fires``.
        false_fired: Repeats of those probes in which it fired anyway.
        false_trials: Repeats of those probes that produced a turn.
        verdict: The verdict (see the module docstring).
    """

    rule: int
    when: str
    probes: int
    fired: int
    trials: int
    false_probes: int
    false_fired: int
    false_trials: int
    verdict: Verdict

    @property
    def passed(self) -> bool:
        """True unless the rule failed (``untested`` passes; it is reported separately)."""
        return self.verdict in (Verdict.OK, Verdict.UNTESTED)


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Everything one probe run found.

    Attributes:
        persona_id: The probed persona.
        results: One result per probe, in file order.
        rules: One verdict per reaction rule, in rule order.
        repeats: Repeats per probe.
        threshold: The share of repeats a rule had to fire in.
    """

    persona_id: str
    results: tuple[ProbeResult, ...]
    rules: tuple[RuleVerdict, ...]
    repeats: int
    threshold: Fraction = DEFAULT_THRESHOLD

    @property
    def failed(self) -> tuple[RuleVerdict, ...]:
        """The rules that failed, in rule order."""
        return tuple(r for r in self.rules if not r.passed)

    @property
    def empty(self) -> tuple[str, ...]:
        """Ids of the probes none of whose repeats produced a turn."""
        return tuple(r.probe.id for r in self.results if r.valid == 0)

    @property
    def passed(self) -> bool:
        """True when no rule failed and every probe has at least one valid repeat."""
        return not self.failed and not self.empty

    @property
    def untested(self) -> tuple[int, ...]:
        """Rule numbers that no probe expects to fire."""
        return tuple(r.rule for r in self.rules if r.verdict is Verdict.UNTESTED)


# -- files ----------------------------------------------------------------------------


def bundled_probes_dir() -> Path:
    """The directory holding the bundled probe files.

    That is ``moodsona/probes/`` in an installed package, or the repository's
    top-level ``probes/`` in a source checkout.

    Raises:
        FileNotFoundError: Neither directory exists (a broken install).
    """
    packaged = resources.files("moodsona") / "probes"
    if isinstance(packaged, Path) and packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "probes"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("bundled probes directory not found")


def bundled_probes_path(persona_id: str) -> Path:
    """The bundled probe file of a persona, ``<probes dir>/<persona id>.yaml``.

    Raises:
        KeyError: There is no bundled probe file for that persona.
        FileNotFoundError: The bundled probes directory is missing (a broken install).
    """
    path = bundled_probes_dir() / f"{persona_id}{PROBE_SUFFIX}"
    if not path.is_file():
        raise KeyError(f"no probe file for persona {persona_id!r}")
    return path


def _read_yaml(path: Path) -> Any:
    """The YAML document in ``path``, read safely, rejecting duplicate keys.

    Raises:
        ProbeError: The file does not exist, cannot be read or cannot be parsed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProbeError(str(path), ["file not found"]) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ProbeError(str(path), [f"cannot read the file: {exc}"]) from exc
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ProbeError(str(path), [f"cannot parse YAML: {exc}"]) from exc


def load_probes(path: str | PathLike[str], seed: PersonaSeed) -> ProbeSet:
    """Read and check the probe file of ``seed``.

    Args:
        path: A probe YAML file.
        seed: The persona the probes are written for.

    Returns:
        The probes.

    Raises:
        ProbeError: The file cannot be read or does not fit the persona.
    """
    path = Path(path)
    return probes_from_mapping(_read_yaml(path), seed, source=str(path))


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: Any) -> str | None:
    """``value`` stripped, or None when it is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _contact_problems(name: str, text: str) -> list[str]:
    found = [
        f"{name} contains what looks like {label}"
        for label, pattern in CONTACT_PATTERNS
        if pattern.search(text)
    ]
    if looks_like_phone(text):
        found.append(f"{name} contains what looks like a phone number")
    return found


def _unknown(data: dict[Any, Any], allowed: frozenset[str], where: str = "") -> list[str]:
    extra = sorted(str(k) for k in data if not isinstance(k, str) or k not in allowed)
    return [f"unknown key(s){where}: {', '.join(extra)}"] if extra else []


def _rules(value: Any, name: str, count: int, problems: list[str]) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(map(_is_int, value)):
        problems.append(f"{name} must be a list of rule numbers")
        return ()
    for number in sorted(set(value)):
        if not 1 <= number <= count:
            problems.append(f"{name}: rule {number} does not exist (the persona has {count})")
        if value.count(number) > 1:
            problems.append(f"{name} lists rule {number} twice")
    return tuple(value)


def _state(value: Any, seed: PersonaSeed, problems: list[str]) -> Mood:
    start = seed.initial_state
    if value is None:
        return start
    if not isinstance(value, dict):
        problems.append("state must be a mapping of emotion, energy and stance")
        return start
    problems += _unknown(value, frozenset({"emotion", "energy", "stance"}), " in state")
    scores = {}
    for name in ("emotion", "energy"):
        score = value.get(name, getattr(start, name))
        if not _is_int(score) or not 0 <= score <= 100:
            problems.append(f"state.{name} must be an integer 0-100")
            score = getattr(start, name)
        scores[name] = score
    stance = value.get("stance", start.stance.value)
    if not isinstance(stance, str) or stance not in _STANCES:
        problems.append(f"state.stance must be one of {', '.join(s.value for s in Stance)}")
        stance = start.stance.value
    return Mood(scores["emotion"], scores["energy"], Stance(stance))


def _memory(value: Any, day: int, seed: PersonaSeed, problems: list[str]) -> tuple[DayNote, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        problems.append("memory must be a list of {day, notes}")
        return ()
    notes: list[DayNote] = []
    for i, item in enumerate(value):
        name = f"memory[{i}]"
        if not isinstance(item, dict):
            problems.append(f"{name} must be a mapping with day and notes")
            continue
        problems += _unknown(item, frozenset({"day", "notes"}), f" in {name}")
        when, lines = item.get("day"), item.get("notes")
        texts = [_text(line) for line in lines] if isinstance(lines, list) else []
        if not texts or None in texts:
            problems.append(f"{name}.notes must be a non-empty list of non-empty strings")
        if not _is_int(when) or when < 1:
            problems.append(f"{name}.day must be an integer, at least 1")
            continue
        if when >= day:
            problems.append(f"{name}: memory day {when} must be before the probe's day {day}")
        elif when < day - seed.memory_days_visible:
            problems.append(
                f"{name}: memory day {when} is not visible on day {day} "
                f"(memory_days_visible {seed.memory_days_visible})"
            )
        if any(note.day == when for note in notes):
            problems.append(f"{name}: memory day {when} is listed twice (one entry per day)")
        lines_ok = tuple(t for t in texts if t is not None)
        for line in lines_ok:
            problems += _contact_problems(f"{name}.notes", line)
        notes.append(DayNote(when, lines_ok))
    return tuple(notes)


def _context(value: Any, day: int, problems: list[str]) -> tuple[ChatMessage, ...]:
    if not isinstance(value, list) or not value:
        problems.append("context must be a non-empty list of {persona: ...} / {companion: ...}")
        return ()
    messages: list[ChatMessage] = []
    for i, item in enumerate(value):
        name = f"context[{i}]"
        if not isinstance(item, dict) or len(item) != 1:
            problems.append(f"{name} must have exactly one key, persona or companion")
            continue
        ((speaker, raw),) = item.items()
        text = _text(raw)
        if not isinstance(speaker, str) or speaker not in _SPEAKERS or text is None:
            problems.append(
                f"{name} must have exactly one key, persona or companion, with a non-empty text"
            )
            continue
        problems += _contact_problems(name, text)
        messages.append(ChatMessage(_SPEAKERS[speaker], text, day=day))
    if messages and messages[-1].role != "user":
        problems.append("context must end with a persona message (the reply answers it)")
    if day > 1 and messages and messages[0].role != "user":
        problems.append(
            f"on day {day} the context must start with a persona message: in a run, "
            "every day after the first starts with the persona writing"
        )
    window = SimulatorConfig().window_messages
    if len(value) + 1 > window:
        problems.append(
            f"context has {len(value)} messages; with the reply the persona sees only "
            f"the last {window}"
        )
    return tuple(messages)


def _probe(item: Any, seed: PersonaSeed, problems: list[str]) -> Probe | None:
    """Build one probe, adding its problems (without a prefix) to ``problems``."""
    if not isinstance(item, dict):
        problems.append("must be a mapping")
        return None
    problems += _unknown(item, _PROBE_KEYS)
    probe_id = item.get("id")
    if not isinstance(probe_id, str) or not _ID.fullmatch(probe_id):
        problems.append("id must be kebab-case, e.g. silver-lining")
    note = item.get("note", "")
    if not isinstance(note, str):
        problems.append("note must be a string")
        note = ""
    day = item.get("day", 1)
    if not _is_int(day) or day < 1:
        problems.append("day must be an integer, at least 1")
        day = 1
    reply = _text(item.get("reply"))
    if reply is None:
        problems.append("reply must be a non-empty string")
    count = len(seed.reaction_rules)
    fires = _rules(item.get("fires"), "fires", count, problems)
    not_fires = _rules(item.get("not_fires"), "not_fires", count, problems)
    for number in sorted(set(fires) & set(not_fires)):
        problems.append(f"rule {number} is in both fires and not_fires")
    if not fires and not not_fires:
        problems.append("needs fires or not_fires (rule numbers), or both")
    for name, text in (("reply", reply or ""), ("note", note)):
        problems += _contact_problems(name, text)
    probe = Probe(
        id=str(probe_id),
        context=_context(item.get("context"), day, problems),
        reply=reply or "",
        state=_state(item.get("state"), seed, problems),
        fires=fires,
        not_fires=not_fires,
        day=day,
        memory=_memory(item.get("memory"), day, seed, problems),
        note=note.strip(),
    )
    return None if problems else probe


def probes_from_mapping(data: Any, seed: PersonaSeed, *, source: str = "probes") -> ProbeSet:
    """Check a loaded probe document against its persona and build the probes.

    Args:
        data: The parsed YAML document.
        seed: The persona the probes are written for.
        source: Name used in error messages.

    Returns:
        The probes.

    Raises:
        ProbeError: The document does not fit the format or the persona
            (unknown rule numbers, a different persona id, ...).
    """
    if not isinstance(data, dict):
        raise ProbeError(source, ["top level must be a mapping with persona and probes"])
    problems = _unknown(data, _TOP_KEYS)
    if data.get("persona") != seed.id:
        problems.append(
            f"persona {data.get('persona')!r} must be {seed.id!r}, the persona being probed"
        )
    items = data.get("probes")
    if not isinstance(items, list) or not items:
        problems.append("probes must be a non-empty list")
        items = []
    probes: list[Probe] = []
    for index, item in enumerate(items):
        mine: list[str] = []
        probe = _probe(item, seed, mine)
        name = item.get("id") if isinstance(item, dict) else None
        label = f"probes[{index}]" + (f" ({name})" if isinstance(name, str) else "")
        problems += [f"{label}: {p}" for p in mine]
        if probe is None:
            continue
        if any(p.id == probe.id for p in probes):
            problems.append(f"{label}: id {probe.id!r} is used twice")
        probes.append(probe)
    if problems:
        raise ProbeError(source, problems)
    return ProbeSet(seed.id, tuple(probes), source)


# -- running ----------------------------------------------------------------------------


def probe_once(
    seed: PersonaSeed,
    probe: Probe,
    provider: LLMProvider,
    *,
    repeat: int = 0,
    config: SimulatorConfig | None = None,
    language: LanguageProfile | None = None,
    safety_hook: SafetyHook | None = None,
) -> frozenset[int] | None:
    """Run one probe once: one simulated-user turn after the scripted reply.

    A fresh :class:`~moodsona.simulator.Simulator` resumes the probe's day
    (:meth:`~moodsona.simulator.Simulator.resume_day`) with the probe's mood
    and memory notes, the persona's messages in the context counted as its own
    lines of the day, and no rule fired yet today. The context plus the reply
    under test is today's visible conversation; the persona then writes one
    turn through the usual prompt and rails.

    Args:
        seed: The persona.
        probe: The probe.
        provider: The user model.
        repeat: The repeat number; seeds the simulator's random source.
        config: Optional simulator settings.
        language: Optional language profile (see
            :class:`~moodsona.simulator.Simulator`).
        safety_hook: Optional crisis detector.

    Returns:
        The numbers of the rules that were applied, or None when no attempt
        passed the rails and the turn fell back to the neutral line.

    Raises:
        LookupError: No language profile covers the persona's language.
        moodsona.llm.LLMError: The provider failed.
    """
    simulator = Simulator(
        seed,
        provider,
        language=language,
        safety_hook=safety_hook,
        config=config,
        rng=random.Random(repeat),  # noqa: S311 - simulation, not cryptography
    )
    simulator.resume_day(
        probe.day, mood=probe.state, memory=probe.memory, said_today=probe.own_lines
    )
    turn = simulator.next_turn(probe.history, day=probe.day, turn_in_day=probe.turn_in_day)
    if turn.polluted:
        return None
    return frozenset(r.rule for r in turn.rules_applied)


def _verdict(
    probes: int,
    fired: int,
    trials: int,
    false_probes: int,
    false_fired: int,
    false_trials: int,
    threshold: Fraction,
) -> Verdict:
    if probes:
        if trials == 0:
            return Verdict.NO_DATA
        if fired == 0:
            return Verdict.NEVER
        if Fraction(fired, trials) < threshold:
            return Verdict.WEAK
    if false_probes:
        if false_trials == 0:
            return Verdict.NO_DATA
        if Fraction(false_fired, false_trials) > 1 - threshold:
            return Verdict.OVERFIRES
    return Verdict.OK if probes else Verdict.UNTESTED


def judge(
    seed: PersonaSeed,
    results: Sequence[ProbeResult],
    *,
    threshold: Fraction | float | str = DEFAULT_THRESHOLD,
) -> tuple[RuleVerdict, ...]:
    """One verdict per reaction rule of ``seed`` from the probe results.

    Args:
        seed: The persona.
        results: The probe results.
        threshold: The share of repeats a rule must fire in on its probes
            (see :func:`as_threshold`).

    Returns:
        The verdicts, in rule order.

    Raises:
        ValueError: ``threshold`` is outside (0, 1].
    """
    share = as_threshold(threshold)
    verdicts = []
    for number, rule in enumerate(seed.reaction_rules, start=1):
        positive = [r for r in results if number in r.probe.fires]
        negative = [r for r in results if number in r.probe.not_fires]
        counts = (
            len(positive),
            sum(r.count(number) for r in positive),
            sum(r.valid for r in positive),
            len(negative),
            sum(r.count(number) for r in negative),
            sum(r.valid for r in negative),
        )
        verdicts.append(RuleVerdict(number, rule.when, *counts, _verdict(*counts, share)))
    return tuple(verdicts)


def max_model_calls(probes: int, repeats: int, config: SimulatorConfig | None = None) -> int:
    """The most user-model calls a probe run can make (every attempt but the last fails)."""
    return probes * repeats * (config or SimulatorConfig()).max_attempts


def run_probes(
    seed: PersonaSeed,
    probes: ProbeSet,
    provider: LLMProvider,
    *,
    repeats: int = DEFAULT_REPEATS,
    threshold: Fraction | float | str = DEFAULT_THRESHOLD,
    config: SimulatorConfig | None = None,
    language: LanguageProfile | None = None,
    safety_hook: SafetyHook | None = None,
    progress: Callable[[str], object] | None = None,
) -> ProbeReport:
    """Run every probe ``repeats`` times and judge every rule.

    Args:
        seed: The persona.
        probes: Its probes (see :func:`load_probes`).
        provider: The user model.
        repeats: Runs per probe; model output varies, so one run says little.
        threshold: The share of repeats, in (0, 1], a rule must fire in on its
            probes; it may fire in at most ``1 - threshold`` on the probes that
            list it under ``not_fires``. Compared exactly (see :func:`as_threshold`).
        config: Optional simulator settings.
        language: Optional language profile.
        safety_hook: Optional crisis detector.
        progress: Called with one line per probe (:func:`format_result`).

    Returns:
        The report.

    Raises:
        ValueError: ``repeats`` is below 1, ``threshold`` is outside (0, 1],
            the probes are for another persona, or a probe does not fit in the
            simulator's window (``config.window_messages``).
        LookupError: No language profile covers the persona's language.
        moodsona.llm.LLMError: The provider failed.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    share = as_threshold(threshold)
    if probes.persona_id != seed.id:
        raise ValueError(f"these probes are for persona {probes.persona_id!r}, not {seed.id!r}")
    window = (config or SimulatorConfig()).window_messages
    too_long = [p.id for p in probes.probes if len(p.history) > window]
    if too_long:
        raise ValueError(
            f"probe(s) {', '.join(too_long)}: context and reply do not fit the simulator's "
            f"window of {window} messages"
        )
    results = []
    for probe in probes.probes:
        fired = tuple(
            probe_once(
                seed,
                probe,
                provider,
                repeat=i,
                config=config,
                language=language,
                safety_hook=safety_hook,
            )
            for i in range(repeats)
        )
        result = ProbeResult(probe, fired)
        results.append(result)
        if progress is not None:
            progress(format_result(result))
    rules = judge(seed, results, threshold=share)
    return ProbeReport(seed.id, tuple(results), rules, repeats, share)


# -- output ----------------------------------------------------------------------------


def _numbers(numbers: Sequence[int]) -> str:
    return ",".join(map(str, numbers))


def _short(text: str) -> str:
    return text if len(text) <= _WHEN_WIDTH else text[: _WHEN_WIDTH - 3].rstrip() + "..."


def format_result(result: ProbeResult) -> str:
    """One line per probe: what it expects and which rules fired in how many repeats."""
    probe = result.probe
    expects = "; ".join(
        part
        for part in (
            f"fires {_numbers(probe.fires)}" if probe.fires else "",
            f"not {_numbers(probe.not_fires)}" if probe.not_fires else "",
        )
        if part
    )
    seen = sorted({n for f in result.fired if f is not None for n in f})
    fired = ", ".join(f"rule {n} {result.count(n)}/{result.valid}" for n in seen)
    line = f"probe {probe.id} [{expects}]: {fired or f'no rule in {result.valid} repeat(s)'}"
    return line + (f", {result.polluted} polluted" if result.polluted else "")


def format_report(report: ProbeReport) -> list[str]:
    """The per-rule table and the summary lines of a probe run."""
    lines = [f"{'rule':>4}  {'probes':>6}  {'fired':>5}  {'false':>5}  {'verdict':<9}  when"]
    for r in report.rules:
        fired = f"{r.fired}/{r.trials}" if r.probes else "-"
        false = f"{r.false_fired}/{r.false_trials}" if r.false_probes else "-"
        lines.append(
            f"{r.rule:>4}  {r.probes:>6}  {fired:>5}  {false:>5}  {r.verdict.value:<9}  "
            f"{_short(r.when)}"
        )
    hit, miss = report.threshold, 1 - report.threshold
    failures = []
    if report.failed:
        failures.append(", ".join(f"rule {r.rule} ({r.verdict.value})" for r in report.failed))
    if report.empty:
        failures.append(
            f"probe(s) {', '.join(report.empty)} had no valid repeat "
            "(every turn fell back to the neutral line)"
        )
    if failures:
        lines.append(
            f"failed: {'; '.join(failures)}. A rule passes when it fires in at least {hit} "
            f"of the repeats of its probes and in at most {miss} of the repeats of probes "
            "that list it under not_fires."
        )
    elif any(r.probes for r in report.rules):
        lines.append(f"passed: every probed rule fired in at least {hit} of its repeats")
    else:
        lines.append(
            f"passed: no rule fired in more than {miss} of the repeats of probes that list it "
            "under not_fires; no probe expects a rule to fire"
        )
    if report.untested:
        lines.append(f"no probe expects rule(s) {', '.join(map(str, report.untested))} to fire")
    return lines


def format_coverage(probes: ProbeSet, seed: PersonaSeed) -> list[str]:
    """Per rule, how many probes expect it to fire and how many expect it not to."""
    lines = [
        f"{probes.source}: {len(probes.probes)} probe(s) for persona {seed.id}, 0 problem(s)",
        f"{'rule':>4}  {'fire':>4}  {'not':>4}  when",
    ]
    for number, rule in enumerate(seed.reaction_rules, start=1):
        fire = sum(number in p.fires for p in probes.probes)
        never = sum(number in p.not_fires for p in probes.probes)
        lines.append(f"{number:>4}  {fire:>4}  {never:>4}  {_short(rule.when)}")
    untested = probes.untested(len(seed.reaction_rules))
    if untested:
        lines.append(f"no probe expects rule(s) {', '.join(map(str, untested))} to fire")
    return lines


# -- offline checks ----------------------------------------------------------------------


def _check_file(path: Path) -> list[str]:
    """Problems of one probe file, checked against the bundled persona it names."""
    try:
        data = _read_yaml(path)
    except ProbeError as exc:
        return [str(exc)]
    persona_id = data.get("persona") if isinstance(data, dict) else None
    if not isinstance(persona_id, str):
        return [f"{path}: needs a persona id (persona: <id>)"]
    problems = []
    expected = f"{persona_id}{PROBE_SUFFIX}"
    if path.name != expected:
        problems.append(
            f"{path}: file name {path.name!r} must be {expected!r} (the persona id "
            f"{persona_id!r} plus {PROBE_SUFFIX})"
        )
    try:
        seed = load_bundled_persona(persona_id)
        probes_from_mapping(data, seed, source=str(path))
    except KeyError as exc:
        problems.append(f"{path}: {exc.args[0]}")
    except ProbeError as exc:
        problems += [f"{path}: {p}" for p in exc.problems]
    except ValueError as exc:  # PersonaError: the persona file itself is broken
        problems.append(f"{path}: {exc}")
    return problems


def check_main(argv: Sequence[str], *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    """Check probe files against the bundled personas they name, offline.

    Every file must parse, name a bundled persona (``persona: <id>``), be
    called ``<id>.yaml`` and fit that persona: rule numbers that exist, valid
    states and memory days, no contact data. Used by ``tools/validate_probes.py``.

    Args:
        argv: Probe files or directories; none means the bundled probes.
        out: Stream for the summary line (default: stdout).
        err: Stream for problems (default: stderr).

    Returns:
        0 when every file passes, 1 when any problem is found, 2 when a path
        does not exist or there is no probe file to check.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        targets: list[str | Path] = list(argv) or [bundled_probes_dir()]
    except FileNotFoundError as exc:
        err.write(f"{exc}\n")
        return 2
    missing = [t for t in targets if not Path(t).exists()]
    files = list(iter_files(targets))
    if missing or not files:
        for target in missing:
            err.write(f"{target}: no such file or directory\n")
        if not missing:
            err.write("no probe files to check (expected .yaml files or directories)\n")
        return 2
    problems = [p for path in files for p in _check_file(path)]
    for problem in problems:
        err.write(problem + "\n")
    out.write(f"checked {len(files)} probe file(s), {len(problems)} problem(s)\n")
    return 1 if problems else 0
