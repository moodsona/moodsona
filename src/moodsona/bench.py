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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from moodsona.companion import ScriptedCompanion
from moodsona.llm.anthropic import AnthropicProvider
from moodsona.llm.base import LLMProvider
from moodsona.llm.fake import FakeProvider
from moodsona.llm.openai import OpenAIProvider
from moodsona.metrics import (
    TranscriptMetrics,
    compute_metrics,
    format_number,
    format_share,
    metrics_from_jsonl,
)
from moodsona.runner import run
from moodsona.simulator import SimulatorConfig
from moodsona.state import PersonaSeed

__all__ = [
    "BENCH_COMPANION_LINES",
    "PROVIDERS",
    "BenchRun",
    "bench_companion",
    "make_provider",
    "max_model_calls",
    "mean_or_none",
    "run_one",
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


def safe_label(text: str) -> str:
    """Turn a model id into a directory name: runs of other characters become ``-``."""
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
        metrics: Its metrics.
    """

    model: str
    path: Path
    metrics: TranscriptMetrics


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

    The transcript goes to ``<out_dir>/<model>/<persona id>-seed<rng_seed>.jsonl``
    (an earlier file of the same run is replaced).

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
    """
    transcript = run(persona, provider, bench_companion(), days, turns_per_day, rng_seed)
    folder = Path(out_dir) / model
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{persona.id}-seed{rng_seed}.jsonl"
    transcript.to_jsonl(path)
    return BenchRun(model, path, compute_metrics(transcript))


def mean_or_none(values: Sequence[float | None]) -> float | None:
    """Mean of the values that are not None; None when there are none."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


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
    ("days skipped", lambda m: float(m.days_skipped), format_number),
)


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(first: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    header = [*first, *(name for name, _, _ in _COLUMNS)]
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def _collect(out_dir: Path) -> list[BenchRun]:
    return [
        BenchRun(path.parent.name, path, metrics_from_jsonl(path))
        for path in sorted(out_dir.glob("*/*.jsonl"))
    ]


def write_report(out_dir: str | PathLike[str]) -> Path:
    """Write ``bench.md`` for every transcript in ``<out_dir>/<model>/``.

    Args:
        out_dir: The bench directory.

    Returns:
        The path of ``bench.md``.

    Raises:
        ValueError: No transcripts found, or a file is not a transcript.
        OSError: A file cannot be read or written.
    """
    folder = Path(out_dir)
    runs = _collect(folder)
    if not runs:
        raise ValueError(f"{folder}: no bench runs (expected <model>/<persona>-seed<n>.jsonl)")
    models = sorted({r.model for r in runs})
    summary = []
    for model in models:
        mine = [r.metrics for r in runs if r.model == model]
        cells = [fmt(mean_or_none([get(m) for m in mine])) for _, get, fmt in _COLUMNS]
        summary.append([model, str(len(mine)), *cells])
    per_run = [
        [r.model, r.metrics.persona_id, str(r.metrics.rng_seed), str(r.metrics.turns)]
        + [fmt(get(r.metrics)) for _, get, fmt in _COLUMNS]
        for r in runs
    ]
    lines = [
        "# moodsona bench",
        "",
        "User models playing moodsona personas against a fixed scripted companion. "
        "Each model row is the mean over its runs (persona x seed); `n/a` means there "
        "was nothing to count. Definitions: `moodsona.metrics`. Lower is better for "
        "polluted, forced, own repeats, agreement, ask-back and wrong language; the "
        "other columns describe behaviour and are read against each persona's design. "
        "There is no single score yet.",
        "",
        *_table(["model", "runs"], summary),
        "",
        "## Per run",
        "",
        *_table(["model", "persona", "seed", "turns"], per_run),
        "",
    ]
    path = folder / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
