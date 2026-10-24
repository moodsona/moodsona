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
heuristics (:mod:`moodsona.lang`), the same ones the simulator's rails use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from os import PathLike
from statistics import fmean
from typing import Any

from moodsona.checks import repeats_earlier
from moodsona.lang import get_profile
from moodsona.lang.base import LanguageProfile
from moodsona.runner import Transcript, TurnRecord
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
            reported them (unchecked).
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
    turns: int
    messages: int
    active_days: int
    days_skipped: int
    emotion: SeriesSummary
    energy: SeriesSummary
    intent_shares: dict[str, float] = field(default_factory=dict)
    rule_trigger_rate: float | None = None
    rule_counts: dict[int, int] = field(default_factory=dict)
    own_repeat_rate: float | None = None
    agreement_rate: float | None = None
    ask_back_rate: float | None = None
    mean_message_chars: float | None = None
    short_message_share: float | None = None
    wrong_language_share: float | None = None
    stance_changes_per_day: float | None = None
    polluted_share: float | None = None
    forced_share: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (rule numbers become string keys, as JSON requires)."""
        data = asdict(self)
        data["rule_counts"] = {str(k): v for k, v in sorted(self.rule_counts.items())}
        return data


def _mean(values: list[float]) -> float | None:
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
    values = [float(v) for _, v in points]
    mean = _mean(values)
    volatility = _mean([abs(b - a) for a, b in pairwise(values)])
    drift: float | None = None
    days = [float(d) for d, _ in points]
    if len(set(days)) >= 2 and mean is not None:
        mean_day = sum(days) / len(days)
        spread = sum((d - mean_day) ** 2 for d in days)
        drift = sum((d - mean_day) * (v - mean) for d, v in zip(days, values, strict=True)) / spread
    return SeriesSummary(mean, drift, volatility)


def _own_repeats(messages: list[TurnRecord]) -> list[bool]:
    """For each message, whether it nearly repeats an earlier message of the same day."""
    flags: list[bool] = []
    today: list[str] = []
    for i, record in enumerate(messages):
        if i and record.day != messages[i - 1].day:
            today = []
        flags.append(repeats_earlier(record.visible, today))
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

    intents = Counter(str(h.get("intent")) for h in hidden)
    intent_shares = {i.value: intents.pop(i.value, 0) / len(hidden) for i in Intent if hidden}
    if intents:  # missing or unknown intents (e.g. from a newer version's file)
        intent_shares[OTHER_INTENT] = intents.total() / len(hidden)
    reported = [list(h.get("rules_triggered") or []) for h in hidden]
    rule_counts = Counter(n for numbers in reported for n in numbers)

    texts = [t.visible for t in messages]
    stance_changes = sum(a.mood.stance != b.mood.stance for a, b in pairwise(messages))
    return TranscriptMetrics(
        persona_id=transcript.persona_id,
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
        agreement_rate=_share([profile.is_agreement(x) for x in texts]) if profile else None,
        ask_back_rate=_share([profile.is_ask_back(x) for x in texts]) if profile else None,
        mean_message_chars=_mean([float(len(x)) for x in texts]),
        short_message_share=_share([len(x.split()) <= short_max_words for x in texts]),
        wrong_language_share=(
            _share([profile.is_wrong_language(x) for x in texts]) if profile else None
        ),
        stance_changes_per_day=stance_changes / active_days if active_days else None,
        polluted_share=_share([t.polluted for t in turns]),
        forced_share=_share([t.forced is not None for t in turns]),
    )


def metrics_from_jsonl(path: str | PathLike[str], **options: Any) -> TranscriptMetrics:
    """Read a transcript JSONL file and compute its metrics.

    Args:
        path: A file written by :meth:`Transcript.to_jsonl`.
        **options: Passed to :func:`compute_metrics`.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not a transcript.
    """
    return compute_metrics(Transcript.from_jsonl(path), **options)
