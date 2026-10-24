"""Benchmark runner: how well user models hold the bundled personas.

``moodsona bench`` runs bundled personas for a number of days with one user
model (the provider that plays the persona) against a fixed, scripted
companion, writes one JSONL transcript per run and a ``bench.md`` table of the
:mod:`moodsona.metrics` per model. Run it once per model into the same output
directory and ``bench.md`` compares them: every ``<out>/<model>/*.jsonl`` is
one row of the per-run table, and the directory name is the model label.

The companion is fixed on purpose: the bench compares user models, not
companions. Its lines (:data:`BENCH_COMPANION_LINES`) are original, deliberately
ordinary replies, some of them the habits personas are written to react to
(several questions at once, generic advice, an early referral), so the user
model has something to react to.

This is the scaffold for the user-model leaderboard on the roadmap. It has no
single score yet; read the columns side by side. With a real provider every
run costs money (see :func:`max_model_calls`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from statistics import fmean

from moodsona.companion import ScriptedCompanion
from moodsona.llm.anthropic import AnthropicProvider
from moodsona.llm.base import LLMError, LLMProvider
from moodsona.llm.fake import FakeProvider
from moodsona.llm.openai import OpenAIProvider
from moodsona.metrics import TranscriptMetrics, compute_metrics, format_number, format_share
from moodsona.personas import list_bundled_personas, load_bundled_persona
from moodsona.runner import Transcript, run
from moodsona.simulator import SimulatorConfig
from moodsona.state import PersonaSeed

__all__ = [
    "BENCH_COMPANION_LINES",
    "PROVIDERS",
    "BenchRun",
    "BenchRunError",
    "bench_companion",
    "collect_runs",
    "load_personas",
    "make_provider",
    "max_model_calls",
    "model_label",
    "run_bench",
    "run_one",
    "run_path",
    "safe_label",
    "write_report",
]

#: Provider names accepted by :func:`make_provider` and ``moodsona bench --provider``.
PROVIDERS = ("fake", "anthropic", "openai")

#: The fixed companion's replies, used in order and then from the start again.
BENCH_COMPANION_LINES = (
    "That sounds like a lot. How are you feeling about it, and what will you do next?",
    "Have you tried breaking it into small steps? A short list often helps.",
    "I'm here for you. Tell me more.",
    "It might help to talk to a professional about this.",
    "That's great to hear! Everything happens for a reason.",
    "What was the best part of your day?",
    "I think you mentioned something like this before. How did that go?",
    "Got it. Is there anything else on your mind?",
)

REPORT_NAME = "bench.md"


def bench_companion() -> ScriptedCompanion:
    """A new fixed companion that cycles through :data:`BENCH_COMPANION_LINES`."""
    return ScriptedCompanion(BENCH_COMPANION_LINES, cycle=True)


def make_provider(name: str, model: str | None) -> LLMProvider:
    """Create the user-model provider ``name`` (one of :data:`PROVIDERS`).

    Args:
        name: ``"fake"``, ``"anthropic"`` or ``"openai"``.
        model: Model id for a real provider; None uses the adapter's default.
            Ignored for ``"fake"``.

    Raises:
        ValueError: Unknown provider name.
        LLMConfigurationError: The provider's SDK is not installed.
    """
    if name == "fake":
        return FakeProvider()
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "openai":
        return OpenAIProvider(model)
    raise ValueError(f"unknown provider {name!r}; use one of {', '.join(PROVIDERS)}")


def load_personas(ids: Sequence[str] | None = None) -> list[PersonaSeed]:
    """Load bundled personas for a bench, each id once, in the order given.

    Args:
        ids: Bundled persona ids; None (or empty) means all bundled personas.

    Raises:
        KeyError: An id is not a bundled persona.
        PersonaError: A bundled persona file is invalid.
        FileNotFoundError: The bundled personas are missing (broken install).
    """
    return [load_bundled_persona(i) for i in dict.fromkeys(ids or list_bundled_personas())]


def safe_label(text: str) -> str:
    """Turn a model id into a directory name: runs of other characters become ``-``.

    Ids that differ only in those characters (``org/model`` and ``org:model``)
    get the same label, so their runs share a directory.
    """
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return label or "model"


def max_model_calls(personas: int, days: int, turns_per_day: int) -> int:
    """The most user-model calls a bench run can make (every attempt fails but the last).

    Per persona and day: up to ``max_attempts`` calls per turn plus up to
    ``notes_attempts`` calls for the end-of-day notes (default settings).
    """
    config = SimulatorConfig()
    per_day = turns_per_day * config.max_attempts + config.notes_attempts
    return personas * days * per_day


@dataclass(frozen=True, slots=True)
class BenchRun:
    """One persona run of a bench.

    Attributes:
        model: The model label (the directory the transcript is in).
        path: The transcript JSONL.
        days: Simulated days the run was given (None when the file does not say).
        turns_per_day: Maximum persona messages per day the run was given
            (None when the file does not say).
        metrics: Its metrics.
    """

    model: str
    path: Path
    days: int | None
    turns_per_day: int | None
    metrics: TranscriptMetrics

    @property
    def setting(self) -> str:
        """``"<days> x <turns per day>"``, the run length the report groups by."""
        return " x ".join("?" if n is None else str(n) for n in (self.days, self.turns_per_day))


def run_path(out_dir: str | PathLike[str], model: str, persona_id: str, rng_seed: int) -> Path:
    """The transcript path of a run: ``<out_dir>/<model>/<persona id>-seed<rng_seed>.jsonl``."""
    return Path(out_dir) / model / f"{persona_id}-seed{rng_seed}.jsonl"


def run_one(
    persona: PersonaSeed,
    provider: LLMProvider,
    *,
    model: str,
    out_dir: str | PathLike[str],
    days: int,
    turns_per_day: int,
    rng_seed: int = 0,
) -> BenchRun:
    """Run one persona against the fixed companion and write its transcript.

    The transcript goes to :func:`run_path` (an earlier file of the same run is
    replaced). The folder is created before the first model call, so a run is
    not paid for and then lost to an unwritable directory.

    Args:
        persona: The persona.
        provider: The user model.
        model: Model label, used as the directory name (see :func:`safe_label`).
        out_dir: The bench directory.
        days: Simulated days.
        turns_per_day: Maximum persona messages per day.
        rng_seed: The run's random seed.

    Raises:
        moodsona.llm.LLMError: The provider failed.
        OSError: The transcript cannot be written.
    """
    path = run_path(out_dir, model, persona.id, rng_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    transcript = run(persona, provider, bench_companion(), days, turns_per_day, rng_seed)
    transcript.to_jsonl(path)
    return _scored(model, path, transcript, persona.language)


def _scored(model: str, path: Path, transcript: Transcript, language: str) -> BenchRun:
    metrics = compute_metrics(transcript, language=language)
    return BenchRun(model, path, transcript.days, transcript.turns_per_day, metrics)


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    """Mean of the values that are not None; None when there are none."""
    present = [v for v in values if v is not None]
    return fmean(present) if present else None


_Column = tuple[str, Callable[[TranscriptMetrics], float | None], Callable[[float | None], str]]

#: (header, metric, format) for each table column.
_COLUMNS: tuple[_Column, ...] = (
    ("polluted", lambda m: m.polluted_share, format_share),
    ("forced", lambda m: m.forced_share, format_share),
    ("own repeats", lambda m: m.own_repeat_rate, format_share),
    ("agreement", lambda m: m.agreement_rate, format_share),
    ("ask-back", lambda m: m.ask_back_rate, format_share),
    ("wrong lang.", lambda m: m.wrong_language_share, format_share),
    ("rule triggers", lambda m: m.rule_trigger_rate, format_share),
    ("short msgs", lambda m: m.short_message_share, format_share),
    ("chars/msg", lambda m: m.mean_message_chars, lambda v: format_number(v, ".0f")),
    ("emotion drift/day", lambda m: m.emotion.drift_per_day, lambda v: format_number(v, "+.1f")),
    ("emotion volatility", lambda m: m.emotion.volatility, format_number),
    ("energy drift/day", lambda m: m.energy.drift_per_day, lambda v: format_number(v, "+.1f")),
    ("energy volatility", lambda m: m.energy.volatility, format_number),
    ("stance changes/day", lambda m: m.stance_changes_per_day, lambda v: format_number(v, ".2f")),
    ("days skipped", lambda m: m.days_skipped, format_number),
)


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(first: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    header = [*first, *(name for name, _, _ in _COLUMNS)]
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def collect_runs(out_dir: str | PathLike[str]) -> list[BenchRun]:
    """Read every bench run in ``<out_dir>/<model>/*.jsonl``, sorted by path.

    A missing directory has no runs. Each transcript is scored in its bundled
    persona's language (transcripts do not record it), or English for an id
    that is not bundled.

    Raises:
        ValueError: A file is not a transcript.
        OSError: A file cannot be read.
    """
    paths = sorted(Path(out_dir).glob("*/*.jsonl"))
    if not paths:
        return []
    languages = {p.id: p.language for p in load_personas()}
    runs = []
    for path in paths:
        transcript = Transcript.from_jsonl(path)
        language = languages.get(transcript.persona_id, "en")
        runs.append(_scored(path.parent.name, path, transcript, language))
    return runs


def write_report(out_dir: str | PathLike[str], runs: Iterable[BenchRun] | None = None) -> Path:
    """Write ``<out_dir>/bench.md``.

    Runs are averaged per model and run length (days x turns per day), so runs
    of different lengths are never mixed in one row.

    Args:
        out_dir: The bench directory.
        runs: The runs to report; None reads every transcript in
            ``<out_dir>/<model>/`` (see :func:`collect_runs`).

    Returns:
        The path of ``bench.md``.

    Raises:
        ValueError: No runs, or a file is not a transcript.
        OSError: A file cannot be read or written.
    """
    folder = Path(out_dir)
    runs = sorted(collect_runs(folder) if runs is None else runs, key=lambda r: r.path)
    if not runs:
        raise ValueError(f"{folder}: no bench runs (expected <model>/<persona>-seed<n>.jsonl)")
    groups: dict[tuple[str, str], list[BenchRun]] = {}
    for r in runs:
        groups.setdefault((r.model, r.setting), []).append(r)
    summary = []
    for (model, setting), mine in sorted(groups.items()):
        cells = [fmt(_mean_or_none([get(r.metrics) for r in mine])) for _, get, fmt in _COLUMNS]
        summary.append([model, setting, str(len(mine)), *cells])
    per_run = [
        [r.model, r.metrics.persona_id, r.setting, str(r.metrics.rng_seed), str(r.metrics.turns)]
        + [fmt(get(r.metrics)) for _, get, fmt in _COLUMNS]
        for r in runs
    ]
    lines = [
        "# moodsona bench",
        "",
        "User models playing moodsona personas against a fixed scripted companion. "
        "Each model row is the mean over its runs (persona x seed) of one length "
        "(days x turns per day); `n/a` means there "
        "was nothing to count. Definitions: `moodsona.metrics`. Lower is better for "
        "polluted, forced, own repeats, agreement, ask-back and wrong language; the "
        "other columns describe behaviour and are read against each persona's design. "
        "There is no single score yet.",
        "",
        *_table(["model", "days x turns", "runs"], summary),
        "",
        "## Per run",
        "",
        *_table(["model", "persona", "days x turns", "seed", "turns"], per_run),
        "",
    ]
    path = folder / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def model_label(name: str, provider: LLMProvider) -> str:
    """The model label for a provider: ``"fake"``, or ``"<name>-<model id>"`` made safe.

    Args:
        name: The provider name (one of :data:`PROVIDERS`).
        provider: The provider :func:`make_provider` returned.
    """
    if name == "fake":
        return "fake"
    return safe_label(f"{name}-{getattr(provider, 'model', 'default')}")


class BenchRunError(Exception):
    """A bench run failed; the runs before it are kept and reported.

    Attributes:
        report: The ``bench.md`` written for the finished runs; None when
            there were none.
    """

    def __init__(self, message: str, report: Path | None) -> None:
        """Create the error with its message and the report that was written."""
        super().__init__(message)
        self.report = report


def run_bench(
    personas: Sequence[PersonaSeed],
    provider: LLMProvider,
    *,
    model: str,
    out_dir: str | PathLike[str],
    days: int,
    turns_per_day: int,
    rng_seed: int = 0,
    skip_existing: bool = False,
    progress: Callable[[str], object] | None = None,
) -> Path:
    """Run every persona with one user model and write ``bench.md``.

    The results already in ``out_dir`` are read before the first model call,
    so a bad file there fails the bench before anything is paid for. The
    bench stops at the first failed run, after writing ``bench.md`` for the
    finished ones; rerun with ``skip_existing`` to resume.

    Args:
        personas: The personas to run.
        provider: The user model.
        model: Model label (see :func:`model_label`).
        out_dir: The bench directory.
        days: Simulated days per run.
        turns_per_day: Maximum persona messages per day.
        rng_seed: The random seed of every run.
        skip_existing: Keep a run already in ``out_dir`` instead of running
            it again, when it has the same days and turns per day.
        progress: Called with one line per persona.

    Returns:
        The path of ``bench.md``.

    Raises:
        BenchRunError: A run failed (the provider, the language profile, or
            writing its transcript).
        ValueError: A file in ``out_dir`` is not a transcript.
        OSError: A file cannot be read or written.
    """
    say = progress or (lambda _: None)
    runs = {r.path: r for r in collect_runs(out_dir)}
    for persona in personas:
        kept = runs.get(run_path(out_dir, model, persona.id, rng_seed)) if skip_existing else None
        if kept and (kept.days, kept.turns_per_day) == (days, turns_per_day):
            say(f"{model} / {persona.id}: kept {kept.path}")
            continue
        try:
            result = run_one(
                persona,
                provider,
                model=model,
                out_dir=out_dir,
                days=days,
                turns_per_day=turns_per_day,
                rng_seed=rng_seed,
            )
        except (LookupError, LLMError, OSError) as exc:
            report = write_report(out_dir, runs.values()) if runs else None
            raise BenchRunError(f"{model} / {persona.id}: {exc}", report) from exc
        runs[result.path] = result
        polluted = format_share(result.metrics.polluted_share)
        turns = result.metrics.turns
        say(f"{model} / {persona.id}: {turns} turn(s), {polluted} polluted, {result.path}")
    return write_report(out_dir, runs.values())
