"""Transcript metrics: how well a simulated user held its persona over a run.

Pure functions over a :class:`~moodsona.runner.Transcript` (or its JSONL). They
describe the simulated user, not the companion: whether the mood moved
plausibly, whether the user kept repeating itself, slid into bare agreement,
asked every question back, or broke the language. Compare them across user
models, personas or seeds; a single run's numbers say little on their own.

What is counted over what:

* Everything except the counts (``turns``, ``active_days``, ``days_skipped``)
  and the polluted / forced shares is computed over the **messages**: the
  turns that are not polluted. A polluted turn carries the simulator's
  fallback line, not the model's, and only repeats the previous state, which
  would damp the volatility of a model that fails often.
* Text metrics read the visible message, after typing noise if
  ``SimulatorConfig.typing_noise`` was on (the simulator's rails see the text
  before the noise, so with noise on the two can differ).
* A rate is None when there is nothing to count (no turns, no messages, or no
  language profile), never a misleading 0.

The agreement, ask-back and wrong-language checks are the language profile's
heuristics (:mod:`moodsona.lang`), and the own-repeat check uses the
simulator's thresholds, so the metrics count what the rails look for.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import groupby, pairwise
from os import PathLike
from statistics import fmean, linear_regression
from typing import Any

from moodsona.checks import repeats_earlier
from moodsona.lang import get_profile
from moodsona.lang.base import LanguageProfile
from moodsona.runner import Transcript, TurnRecord
from moodsona.simulator import SimulatorConfig
from moodsona.state import Intent

#: Messages with at most this many words count as short.
SHORT_MAX_WORDS = 3
#: Key in ``intent_shares`` for messages whose intent is missing or unknown.
OTHER_INTENT = "other"


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    """Summary of one score (emotion or energy) over a run.

    Attributes:
        mean: Mean value; None without points.
        drift_per_day: Least-squares slope of the value against the day
            number, in points per day; None with fewer than two distinct days.
        volatility: Mean absolute change between consecutive points (across
            day boundaries too); None with fewer than two points.
    """

    mean: float | None
    drift_per_day: float | None
    volatility: float | None


@dataclass(frozen=True, slots=True)
class TranscriptMetrics:
    """Numbers that describe how a simulated user behaved in one run.

    Attributes:
        persona_id: The simulated persona.
        rng_seed: The run's random seed.
        turns: All turns, polluted ones included.
        messages: Turns that are not polluted (the base of the message rates).
        active_days: Days with at least one turn.
        days_skipped: Days the persona chose not to write.
        emotion: Emotion trajectory summary.
        energy: Energy trajectory summary.
        intent_shares: Share of messages per intent, every intent listed,
            plus ``"other"`` when some intent is missing or unknown (empty
            without messages).
        rule_trigger_rate: Share of messages whose hidden block reports at
            least one triggered reaction rule.
        rule_counts: How often each rule number was reported, as the model
            reported them (unchecked), sorted by rule number.
        own_repeat_rate: Share of messages that nearly repeat an earlier
            message of the same day (the simulator's ``own_repeat`` rail).
        agreement_rate: Share of messages that are a bare agreement.
        ask_back_rate: Share of messages that end by asking the question back.
        mean_message_chars: Mean length of a message in characters.
        short_message_share: Share of messages with at most
            ``short_max_words`` words.
        wrong_language_share: Share of messages in another language.
        stance_changes_per_day: Stance changes between consecutive messages,
            divided by the active days.
        polluted_share: Share of turns that fell back to the neutral line.
        forced_share: Share of turns accepted despite a soft rail.
    """

    persona_id: str
    rng_seed: int
    turns: int
    messages: int
    active_days: int
    days_skipped: int
    emotion: SeriesSummary
    energy: SeriesSummary
    intent_shares: dict[str, float]
    rule_trigger_rate: float | None
    rule_counts: dict[int, int]
    own_repeat_rate: float | None
    agreement_rate: float | None
    ask_back_rate: float | None
    mean_message_chars: float | None
    short_message_share: float | None
    wrong_language_share: float | None
    stance_changes_per_day: float | None
    polluted_share: float | None
    forced_share: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (rule numbers become string keys, as JSON requires)."""
        data = asdict(self)
        data["rule_counts"] = {str(k): v for k, v in self.rule_counts.items()}
        return data


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def summarize_series(points: list[tuple[int, int]]) -> SeriesSummary:
    """Summarise ``(day, value)`` points given in run order.

    Args:
        points: One ``(day, value)`` pair per turn, in the order of the run.

    Returns:
        Mean, drift per day (least-squares slope over the turns) and
        volatility (mean absolute change between consecutive turns).
    """
    days = [d for d, _ in points]
    values = [v for _, v in points]
    drift = linear_regression(days, values).slope if len(set(days)) >= 2 else None
    volatility = _mean([abs(b - a) for a, b in pairwise(values)])
    return SeriesSummary(_mean(values), drift, volatility)


def _own_repeats(messages: list[TurnRecord]) -> list[bool]:
    """For each message, whether it nearly repeats an earlier message of the same day."""
    config = SimulatorConfig()
    flags: list[bool] = []
    for _, day in groupby(messages, key=lambda t: t.day):
        today: list[str] = []
        for record in day:
            flags.append(
                repeats_earlier(
                    record.visible,
                    today,
                    threshold=config.own_repeat_jaccard,
                    min_words=config.own_repeat_min_words,
                )
            )
            today.append(record.visible)
    return flags


def compute_metrics(
    transcript: Transcript,
    *,
    language: LanguageProfile | str | None = "en",
    short_max_words: int = SHORT_MAX_WORDS,
) -> TranscriptMetrics:
    """Compute the metrics of one run.

    Args:
        transcript: The run.
        language: Language profile, or a language tag for
            :func:`moodsona.lang.get_profile`, for the agreement, ask-back and
            wrong-language checks. Defaults to English (v0.1 personas are
            English; transcripts do not record the language). None turns the
            three checks off; their rates are then None.
        short_max_words: Messages with at most this many words count as short.

    Returns:
        The metrics.
    """
    profile = get_profile(language) if isinstance(language, str) else language
    turns = transcript.turns
    messages = [t for t in turns if not t.polluted and t.hidden is not None]
    hidden = [t.hidden for t in messages if t.hidden is not None]  # the same, typed for mypy
    active_days = len({t.day for t in turns})

    intent_shares: dict[str, float] = {}
    if hidden:
        intents = Counter(str(h.get("intent")) for h in hidden)
        known = {i.value: intents.pop(i.value, 0) for i in Intent}
        if intents:  # missing or unknown intents (e.g. from a newer version's file)
            known[OTHER_INTENT] = intents.total()
        intent_shares = {name: count / len(hidden) for name, count in known.items()}
    reported = [list(h.get("rules_triggered") or []) for h in hidden]
    rule_counts = Counter(n for numbers in reported for n in numbers)

    texts = [t.visible for t in messages]
    agreement = ask_back = wrong_language = None
    if profile is not None:
        agreement = _share([profile.is_agreement(x) for x in texts])
        ask_back = _share([profile.is_ask_back(x) for x in texts])
        wrong_language = _share([profile.is_wrong_language(x) for x in texts])
    stance_changes = sum(a.mood.stance != b.mood.stance for a, b in pairwise(messages))
    return TranscriptMetrics(
        persona_id=transcript.persona_id,
        rng_seed=transcript.rng_seed,
        turns=len(turns),
        messages=len(messages),
        active_days=active_days,
        days_skipped=len(transcript.skipped_days),
        emotion=summarize_series([(t.day, t.mood.emotion) for t in messages]),
        energy=summarize_series([(t.day, t.mood.energy) for t in messages]),
        intent_shares=intent_shares,
        rule_trigger_rate=_share([bool(numbers) for numbers in reported]),
        rule_counts=dict(sorted(rule_counts.items())),
        own_repeat_rate=_share(_own_repeats(messages)),
        agreement_rate=agreement,
        ask_back_rate=ask_back,
        mean_message_chars=_mean([len(x) for x in texts]),
        short_message_share=_share([len(x.split()) <= short_max_words for x in texts]),
        wrong_language_share=wrong_language,
        stance_changes_per_day=stance_changes / active_days if active_days else None,
        polluted_share=_share([t.polluted for t in turns]),
        forced_share=_share([t.forced is not None for t in turns]),
    )


def metrics_from_jsonl(
    path: str | PathLike[str],
    *,
    language: LanguageProfile | str | None = "en",
    short_max_words: int = SHORT_MAX_WORDS,
) -> TranscriptMetrics:
    """Read a transcript JSONL file and compute its metrics.

    Args:
        path: A file written by :meth:`Transcript.to_jsonl`.
        language: See :func:`compute_metrics`.
        short_max_words: See :func:`compute_metrics`.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not a transcript.
    """
    transcript = Transcript.from_jsonl(path)
    return compute_metrics(transcript, language=language, short_max_words=short_max_words)
