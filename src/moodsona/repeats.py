"""Repeated runs: run one persona against one companion several times and summarise.

One run says little. The user model samples its words, so two runs of the same
persona against the same companion differ, whatever ``rng_seed`` is:
``rng_seed`` fixes only the overnight energy offsets and the typing noise, not
the model's output. Compare companions on several repeats each, and on the mean
emotion over the run rather than the emotion at the end:

* In a 75-run study (5 companions x 5 personas x 3 repeats) the repeat-to-repeat
  standard deviation, pooled, was 3.7 points for mean emotion and 9.1 for the
  emotion at the end of the last day.
* To detect a true 10-point difference between two companions (two-sided
  alpha 0.05, power 0.8) that takes about 4 repeats per companion with mean
  emotion, and about 14 with the end value. A 5-point difference takes about
  10 repeats with mean emotion.

So run 5 to 10 repeats per companion and report the spread::

    from moodsona.repeats import compare_runs, run_repeats, summarize_runs

    mine = run_repeats(persona, MyBot, repeats=5, user_model=provider, rng_seed=7)
    theirs = run_repeats(persona, OtherBot, repeats=5, user_model=provider, rng_seed=7)
    print(summarize_runs(mine, persona=persona).to_text())
    print(compare_runs(summarize_runs(mine), summarize_runs(theirs)))

:func:`run_repeats` gives repeat ``i`` the seed ``rng_seed + i``. With one seed
for every repeat, every repeat would get the same overnight energy and the
same typing noise, and the spread between repeats would leave that part of
the variation out. Consecutive seeds keep it in, and the seeds of a set of
repeats are still known. Use the same ``rng_seed`` for both companions you
compare, so their repeats meet the same overnight energy.

The intervals are 95 % intervals for the mean, from Student's t with a small
built-in table (degrees of freedom 1 to 30, then 40, 50, 60, 80, 100, 120;
between rows the smaller row is used, which makes an interval slightly wider,
never narrower). They assume the runs are independent and roughly normal; with
3 or fewer repeats they are wide, and that is the honest answer.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from statistics import fmean, stdev
from typing import TYPE_CHECKING, Any

from moodsona.companion import Companion
from moodsona.llm.base import LLMProvider
from moodsona.metrics import TranscriptMetrics, compute_metrics, format_number
from moodsona.runner import Transcript, run
from moodsona.simulator import SafetyHook, SimulatorConfig
from moodsona.state import PersonaSeed

if TYPE_CHECKING:
    from moodsona.lang.base import LanguageProfile

__all__ = [
    "MEASURES",
    "MeasureSummary",
    "RunsDifference",
    "RunsSummary",
    "compare_runs",
    "run_repeats",
    "summarize_runs",
    "summarize_values",
]

#: Each per-run measure and how to read it from a run's metrics.
_GETTERS: dict[str, Callable[[TranscriptMetrics], float | None]] = {
    "mean_emotion": lambda m: m.emotion.mean,
    "end_emotion": lambda m: m.end_emotion,
    "emotion_change": lambda m: m.emotion_change,
    "mean_energy": lambda m: m.energy.mean,
    "turns": lambda m: m.turns,
    "days_ended_early": lambda m: m.days_ended_early,
}
#: The per-run measures a :class:`RunsSummary` summarises, in display order.
MEASURES: tuple[str, ...] = tuple(_GETTERS)

#: Two-sided 95 % critical values of Student's t, by degrees of freedom.
_T95: tuple[tuple[int, float], ...] = (
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
    (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
    (11, 2.201), (12, 2.179), (13, 2.160), (14, 2.145), (15, 2.131),
    (16, 2.120), (17, 2.110), (18, 2.101), (19, 2.093), (20, 2.086),
    (21, 2.080), (22, 2.074), (23, 2.069), (24, 2.064), (25, 2.060),
    (26, 2.056), (27, 2.052), (28, 2.048), (29, 2.045), (30, 2.042),
    (40, 2.021), (50, 2.009), (60, 2.000), (80, 1.990), (100, 1.984),
    (120, 1.980),
)  # fmt: skip


def _t_critical(df: float) -> float:
    """The 95 % two-sided t value for ``df`` degrees of freedom, from :data:`_T95`.

    A ``df`` between two rows (or a fractional one) uses the smaller row, so
    the interval is never too narrow; above 120 the row for 120 is used.

    Raises:
        ValueError: ``df`` is below 1.
    """
    if not df >= 1:
        raise ValueError(f"df must be at least 1, got {df}")
    return next(t for rows_df, t in reversed(_T95) if rows_df <= df)


@dataclass(frozen=True, slots=True)
class MeasureSummary:
    """One measure over several runs.

    Attributes:
        n: Runs that have a value for the measure.
        mean: Mean over those runs; None when ``n`` is 0.
        sd: Sample standard deviation (``n - 1`` in the denominator); None
            when ``n`` is below 2.
        min: Smallest value; None when ``n`` is 0.
        max: Largest value; None when ``n`` is 0.
        ci95: 95 % t interval for the mean, ``(low, high)``; None when ``n``
            is below 2. With ``n = 2`` it is very wide (t = 12.7).
    """

    n: int
    mean: float | None
    sd: float | None
    min: float | None
    max: float | None
    ci95: tuple[float, float] | None


def summarize_values(values: Iterable[float]) -> MeasureSummary:
    """Summarise one measure over runs: mean, sample SD, min, max, 95 % interval.

    Args:
        values: One value per run.

    Returns:
        The summary; the fields that need more values than given are None.
    """
    data = [float(v) for v in values]
    if not data:
        return MeasureSummary(0, None, None, None, None, None)
    mean = fmean(data)
    if len(data) < 2:
        return MeasureSummary(1, mean, None, data[0], data[0], None)
    sd = stdev(data)
    half = _t_critical(len(data) - 1) * sd / math.sqrt(len(data))
    return MeasureSummary(len(data), mean, sd, min(data), max(data), (mean - half, mean + half))


@dataclass(frozen=True, slots=True)
class RunsSummary:
    """Per-run measures over a set of runs, usually repeats of one persona and companion.

    Each measure is the :mod:`moodsona.metrics` value of every run, summarised
    with :func:`summarize_values`. A run without a value for a measure (no
    messages, say) is left out of that measure only, so ``n`` can differ
    between measures.

    Attributes:
        runs: The number of runs.
        persona_ids: The personas of the runs, each once, in order of first
            appearance.
        mean_emotion: Mean emotion over each run's messages
            (``TranscriptMetrics.emotion.mean``): the primary outcome.
        end_emotion: Emotion after each run's last message.
        emotion_change: End emotion minus the persona's initial emotion;
            empty (``n`` 0) unless :func:`summarize_runs` was given the persona.
        mean_energy: Mean energy over each run's messages.
        turns: Turns per run, polluted ones included.
        days_ended_early: Days per run with fewer turns than ``turns_per_day``.
        final_stances: How many runs ended in each stance, most frequent first.
    """

    runs: int
    persona_ids: tuple[str, ...]
    mean_emotion: MeasureSummary
    end_emotion: MeasureSummary
    emotion_change: MeasureSummary
    mean_energy: MeasureSummary
    turns: MeasureSummary
    days_ended_early: MeasureSummary
    final_stances: dict[str, int]

    def measure(self, name: str) -> MeasureSummary:
        """The summary of measure ``name``, one of :data:`MEASURES`.

        Raises:
            ValueError: ``name`` is not a measure.
        """
        if name not in MEASURES:
            raise ValueError(f"unknown measure {name!r}; use one of {', '.join(MEASURES)}")
        summary: MeasureSummary = getattr(self, name)
        return summary

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (tuples become lists, so it equals its own JSON round trip)."""
        data = asdict(self)
        data["persona_ids"] = list(self.persona_ids)
        for name in MEASURES:
            ci = data[name]["ci95"]
            data[name]["ci95"] = None if ci is None else list(ci)
        return data

    def to_text(self) -> str:
        """A compact table: one line per measure, then the final stances."""
        ids = ", ".join(self.persona_ids) or "no persona"
        lines = [
            f"{self.runs} run(s) of {ids}",
            f"{'':<17}{'n':>3}{'mean':>8}{'sd':>7}{'min':>8}{'max':>8}  95% interval",
        ]
        for name in MEASURES:
            s = self.measure(name)
            ci = "n/a" if s.ci95 is None else f"{s.ci95[0]:.1f} to {s.ci95[1]:.1f}"
            lines.append(
                f"{name.replace('_', ' '):<17}{s.n:>3}{format_number(s.mean):>8}"
                f"{format_number(s.sd):>7}{format_number(s.min):>8}{format_number(s.max):>8}"
                f"  {ci}"
            )
        stances = ", ".join(f"{k} {v}" for k, v in self.final_stances.items())
        lines.append(f"{'final stance':<15}{stances or 'n/a'}")
        return "\n".join(lines)


def summarize_runs(
    transcripts: Iterable[Transcript], *, persona: PersonaSeed | None = None
) -> RunsSummary:
    """Summarise the per-run measures of several runs.

    Works on transcripts from :func:`run_repeats`, :func:`moodsona.run` or
    :meth:`Transcript.from_jsonl <moodsona.runner.Transcript.from_jsonl>`. It
    is meant for repeats of one persona against one companion; runs of
    different personas start from different states, and mixing them mixes
    that in as spread (``persona_ids`` shows what went in).

    Args:
        transcripts: The runs.
        persona: The persona of every run, for ``emotion_change``
            (transcripts do not record the starting state).

    Returns:
        The summary; with no runs every measure has ``n`` 0.

    Raises:
        ValueError: ``persona`` is given and a run simulated another persona.
    """
    runs = list(transcripts)
    # language=None: the language checks are not needed for these measures.
    metrics = [compute_metrics(t, language=None, persona=persona) for t in runs]

    def over_runs(name: str) -> MeasureSummary:
        get = _GETTERS[name]
        return summarize_values(v for m in metrics if (v := get(m)) is not None)

    stances = Counter(m.final_stance for m in metrics if m.final_stance is not None)
    return RunsSummary(
        runs=len(runs),
        persona_ids=tuple(dict.fromkeys(t.persona_id for t in runs)),
        mean_emotion=over_runs("mean_emotion"),
        end_emotion=over_runs("end_emotion"),
        emotion_change=over_runs("emotion_change"),
        mean_energy=over_runs("mean_energy"),
        turns=over_runs("turns"),
        days_ended_early=over_runs("days_ended_early"),
        final_stances=dict(stances.most_common()),
    )


@dataclass(frozen=True, slots=True)
class RunsDifference:
    """The difference in one measure between two sets of runs (``a`` minus ``b``).

    Attributes:
        measure: The measure compared, one of :data:`MEASURES`.
        n_a: Runs of ``a`` with a value for the measure.
        n_b: Runs of ``b`` with a value for the measure.
        difference: Mean of ``a`` minus mean of ``b``; None when a side has no
            value.
        ci95: Welch 95 % interval for the difference; None when a side has
            fewer than 2 values. ``(difference, difference)`` when neither
            side varies at all.
        df: Welch-Satterthwaite degrees of freedom; None without an interval
            or when neither side varies.
    """

    measure: str
    n_a: int
    n_b: int
    difference: float | None
    ci95: tuple[float, float] | None
    df: float | None


def compare_runs(
    a: RunsSummary, b: RunsSummary, *, measure: str = "mean_emotion"
) -> RunsDifference:
    """Compare two sets of runs on one measure: ``a`` minus ``b``, with a Welch interval.

    Welch's interval does not assume the two sets have the same spread. An
    interval that excludes 0 is evidence of a real difference; one that
    includes 0 is not evidence of no difference, only of too few runs to
    tell.

    How many runs it takes: with mean emotion (repeat SD about 3.7 points in
    the study described in :mod:`moodsona.repeats`), about 4 runs per side
    detect a true difference of 10 points and about 10 runs per side one of 5
    points (alpha 0.05, power 0.8); 3 runs per side detect only about 11
    points or more. The end emotion (SD about 9) needs about 14 runs per side
    for 10 points. Each side needs at least 2 runs for an interval at all.

    Args:
        a: The first set, e.g. ``summarize_runs(runs_of_my_bot)``.
        b: The second set.
        measure: The measure to compare, one of :data:`MEASURES`.

    Returns:
        The difference, its interval and degrees of freedom.

    Raises:
        ValueError: ``measure`` is not a measure.
    """
    sa, sb = a.measure(measure), b.measure(measure)
    if sa.mean is None or sb.mean is None:
        return RunsDifference(measure, sa.n, sb.n, None, None, None)
    difference = sa.mean - sb.mean
    if sa.sd is None or sb.sd is None:
        return RunsDifference(measure, sa.n, sb.n, difference, None, None)
    va, vb = sa.sd**2 / sa.n, sb.sd**2 / sb.n
    if va + vb == 0:
        return RunsDifference(measure, sa.n, sb.n, difference, (difference, difference), None)
    df = (va + vb) ** 2 / (va**2 / (sa.n - 1) + vb**2 / (sb.n - 1))
    half = _t_critical(df) * math.sqrt(va + vb)
    return RunsDifference(
        measure, sa.n, sb.n, difference, (difference - half, difference + half), df
    )


def run_repeats(
    persona: PersonaSeed,
    make_companion: Callable[[], Companion],
    *,
    repeats: int,
    user_model: LLMProvider,
    days: int = 3,
    turns_per_day: int = 6,
    rng_seed: int = 0,
    language: LanguageProfile | None = None,
    safety_hook: SafetyHook | None = None,
    config: SimulatorConfig | None = None,
    on_run: Callable[[Transcript], object] | None = None,
) -> list[Transcript]:
    """Run one persona against a fresh companion ``repeats`` times, one run after another.

    Each repeat is :func:`moodsona.run` with ``rng_seed + i`` for repeat ``i``
    (from 0) and a new companion from ``make_companion``, so a companion that
    keeps its own memory never carries one run into the next. Every repeat
    makes the model calls of a full run: with a paid provider, ``repeats``
    times the cost.

    Args:
        persona: The persona.
        make_companion: Called once per repeat for a new companion: your
            companion class (``MyBot``) or a function (``lambda: MyBot(url)``).
        repeats: Number of runs, at least 1; 5 to 10 to compare companions.
        user_model: The model that writes the persona's lines.
        days: Simulated days per run.
        turns_per_day: Maximum persona messages per day.
        rng_seed: Seed of the first repeat; repeat ``i`` uses ``rng_seed + i``.
        language: See :func:`moodsona.run`.
        safety_hook: See :func:`moodsona.run`.
        config: See :func:`moodsona.run`.
        on_run: Called with each transcript as soon as its run is done, e.g.
            to write it to JSONL, so finished runs survive a later failure.

    Returns:
        The transcripts, in order of their seeds.

    Raises:
        ValueError: ``repeats`` is below 1, or see :func:`moodsona.run`.
        TypeError: ``make_companion`` is not callable (an instance was passed
            instead of a class or function), or see :func:`moodsona.run`.
            Both raised before the first model call.
        moodsona.llm.LLMError: The user model failed; the runs before it are
            lost unless ``on_run`` kept them.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not callable(make_companion):
        raise TypeError(
            "run_repeats(): make_companion must make a new companion per run, such as "
            f"your companion class or lambda: MyBot(...); got {type(make_companion).__name__}"
        )
    transcripts: list[Transcript] = []
    for i in range(repeats):
        transcript = run(
            persona,
            make_companion(),
            user_model=user_model,
            days=days,
            turns_per_day=turns_per_day,
            rng_seed=rng_seed + i,
            language=language,
            safety_hook=safety_hook,
            config=config,
        )
        transcripts.append(transcript)
        if on_run is not None:
            on_run(transcript)
    return transcripts
