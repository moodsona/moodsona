from __future__ import annotations

import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from moodsona.companion import Companion, EchoCompanion
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import compute_metrics
from moodsona.repeats import (
    MEASURES,
    MeasureSummary,
    RunsSummary,
    _t_critical,
    compare_runs,
    run_repeats,
    summarize_runs,
    summarize_values,
)
from moodsona.runner import Transcript, TurnRecord, run
from moodsona.state import Mood, Stance
from support import make_seed


def turn(day: int, n: int, emotion: int, *, stance: Stance = Stance.CURIOUS) -> TurnRecord:
    hidden = {"inner_thought": "hm", "emotion": emotion, "energy": 40, "intent": "reply"}
    return TurnRecord(day, n, hidden, "fine", "ok", Mood(emotion, 40, stance))


def transcript(*emotions: int, stance: Stance = Stance.CURIOUS, seed: int = 0) -> Transcript:
    """One day of turns with the given emotions; turns_per_day 3."""
    turns = [turn(1, i, e, stance=stance) for i, e in enumerate(emotions, start=1)]
    return Transcript("test-gardener", seed, turns=turns, days=1, turns_per_day=3)


# ------------------------------------------------------------ summarize_values


def test_summary_of_known_values() -> None:
    s = summarize_values([1, 2, 3, 4, 5])
    assert (s.n, s.mean, s.min, s.max) == (5, 3, 1, 5)
    assert s.sd == pytest.approx(math.sqrt(2.5))  # sample SD, n - 1 in the denominator
    assert s.ci95 is not None
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)  # t(0.975, df=4)
    assert s.ci95 == pytest.approx((3 - half, 3 + half))


def test_one_value_has_no_spread_and_no_interval() -> None:
    assert summarize_values([42]) == MeasureSummary(1, 42, None, 42, 42, None)


def test_no_values_summarise_to_none() -> None:
    assert summarize_values([]) == MeasureSummary(0, None, None, None, None, None)


def test_two_values_get_the_wide_interval_they_deserve() -> None:
    s = summarize_values([40, 44])
    assert s.sd == pytest.approx(math.sqrt(8))
    assert s.ci95 == pytest.approx((42 - 12.706 * 2, 42 + 12.706 * 2))  # t(df=1) = 12.706


def test_t_critical_values() -> None:
    assert _t_critical(1) == 12.706
    assert _t_critical(4) == 2.776
    assert _t_critical(30) == 2.042
    assert _t_critical(2.9) == _t_critical(2)  # fractional (Welch) df round down
    assert _t_critical(35) == _t_critical(30)  # between table rows: the smaller df
    assert _t_critical(10_000) == _t_critical(120)
    with pytest.raises(ValueError, match="df"):
        _t_critical(0.5)


@given(st.floats(min_value=1, max_value=500), st.floats(min_value=1, max_value=500))
def test_t_critical_shrinks_with_df_and_stays_above_the_normal(a: float, b: float) -> None:
    low, high = sorted((a, b))
    assert _t_critical(low) >= _t_critical(high) >= 1.96


scores = st.floats(min_value=0, max_value=100, allow_nan=False)


@given(st.lists(scores, min_size=1, max_size=40))
def test_summary_bounds(values: list[float]) -> None:
    s = summarize_values(values)
    assert s.n == len(values)
    assert s.mean is not None
    assert s.min is not None
    assert s.max is not None
    assert s.min - 1e-9 <= s.mean <= s.max + 1e-9
    if len(values) < 2:
        assert (s.sd, s.ci95) == (None, None)
    else:
        assert s.sd is not None
        assert s.sd >= 0
        assert s.ci95 is not None
        assert s.ci95[0] - 1e-9 <= s.mean <= s.ci95[1] + 1e-9


@given(st.lists(scores, min_size=2, max_size=40), st.floats(min_value=-50, max_value=50))
def test_shifting_values_moves_mean_and_interval_not_the_spread(
    values: list[float], c: float
) -> None:
    before, after = summarize_values(values), summarize_values([v + c for v in values])
    assert before.mean is not None
    assert after.mean is not None
    assert after.mean == pytest.approx(before.mean + c, abs=1e-6)
    assert after.sd == pytest.approx(before.sd, abs=1e-6)
    assert before.ci95 is not None
    assert after.ci95 is not None
    assert after.ci95 == pytest.approx((before.ci95[0] + c, before.ci95[1] + c), abs=1e-6)


@given(st.integers(min_value=2, max_value=30), scores)
def test_identical_values_have_zero_spread(n: int, value: float) -> None:
    s = summarize_values([value] * n)
    assert s.mean == pytest.approx(value)
    assert s.sd == 0
    assert s.ci95 == (s.mean, s.mean)


# -------------------------------------------------------------- summarize_runs


def test_summarize_runs_takes_each_measure_from_the_run_metrics() -> None:
    runs = [
        transcript(40, 44, 48),  # mean 44, end 48, 3 turns, no early end
        transcript(50, 52, stance=Stance.OPEN),  # mean 51, end 52, ended early
        transcript(30, stance=Stance.OPEN),  # mean 30, end 30, ended early
    ]
    s = summarize_runs(runs, persona=make_seed())  # initial emotion 50
    assert (s.runs, s.persona_ids) == (3, ("test-gardener",))
    assert s.mean_emotion == summarize_values([44, 51, 30])
    assert s.end_emotion == summarize_values([48, 52, 30])
    assert s.emotion_change == summarize_values([-2, 2, -20])
    assert s.mean_energy == summarize_values([40, 40, 40])
    assert s.turns == summarize_values([3, 2, 1])
    assert s.days_ended_early == summarize_values([0, 1, 1])
    assert s.final_stances == {"open": 2, "curious": 1}
    for name in MEASURES:
        assert s.measure(name) is getattr(s, name)


def test_summarize_runs_matches_compute_metrics() -> None:
    one = transcript(40, 44, 48)
    m = compute_metrics(one, persona=make_seed())
    s = summarize_runs([one], persona=make_seed())
    assert (s.mean_emotion.mean, s.end_emotion.mean) == (m.emotion.mean, m.end_emotion)
    assert (s.mean_energy.mean, s.emotion_change.mean) == (m.energy.mean, m.emotion_change)


def test_runs_without_a_value_are_left_out_of_that_measure() -> None:
    empty = Transcript("test-gardener", 1, skipped_days=[1], days=1, turns_per_day=3)
    s = summarize_runs([transcript(40, 44, 48), empty])
    assert s.runs == 2
    assert s.mean_emotion == summarize_values([44])  # the empty run has no emotion to average
    assert s.turns == summarize_values([3, 0])  # but it does have zero turns
    assert s.days_ended_early == summarize_values([0, 0])  # a skipped day is not ended early
    assert s.emotion_change == summarize_values([])  # no persona given
    assert s.final_stances == {"curious": 1}


def test_unknown_turn_limit_leaves_days_ended_early_out() -> None:
    s = summarize_runs([replace(transcript(40, 44), turns_per_day=None), transcript(40, 44)])
    assert s.days_ended_early == summarize_values([1])


def test_no_runs_summarise_to_none() -> None:
    s = summarize_runs([])
    assert (s.runs, s.persona_ids, s.final_stances) == (0, (), {})
    assert all(s.measure(name) == summarize_values([]) for name in MEASURES)


def test_summarize_runs_rejects_another_persona() -> None:
    with pytest.raises(ValueError, match="other-persona"):
        summarize_runs([transcript(40)], persona=make_seed(id="other-persona"))


def test_summarize_runs_lists_each_persona_once() -> None:
    other = replace(transcript(40), persona_id="someone-else")
    s = summarize_runs([transcript(40), other, transcript(50)])
    assert s.persona_ids == ("test-gardener", "someone-else")


def test_summarize_runs_works_on_transcripts_read_back(tmp_path: Path) -> None:
    runs = [transcript(40, 44, 48, seed=1), transcript(50, 52, seed=2)]
    paths = []
    for t in runs:
        path = tmp_path / f"run-{t.rng_seed}.jsonl"
        t.to_jsonl(path)
        paths.append(path)
    loaded = [Transcript.from_jsonl(p) for p in paths]
    assert summarize_runs(loaded) == summarize_runs(runs)


def test_runs_summary_to_dict_is_json_ready() -> None:
    s = summarize_runs([transcript(40, 44, 48), transcript(50)], persona=make_seed())
    data = s.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["mean_emotion"]["n"] == 2
    assert data["mean_emotion"]["ci95"] == list(s.mean_emotion.ci95 or ())
    assert data["persona_ids"] == ["test-gardener"]


def test_runs_summary_text() -> None:
    text = summarize_runs([transcript(40, 44, 48), transcript(50)]).to_text()
    lines = text.splitlines()
    assert lines[0] == "2 run(s) of test-gardener"
    assert lines[1].split() == ["n", "mean", "sd", "min", "max", "95%", "interval"]
    assert lines[2].split()[:4] == ["mean", "emotion", "2", "47.0"]
    assert "emotion change" in text
    assert "final stance   curious 2" in lines


def test_runs_summary_text_of_one_run_says_there_is_no_interval() -> None:
    line = summarize_runs([transcript(40, 44, 48)]).to_text().splitlines()[2]
    assert line.split() == ["mean", "emotion", "1", "44.0", "n/a", "44.0", "44.0", "n/a"]


# ----------------------------------------------------------------- compare_runs


def runs_with_means(*means: int) -> RunsSummary:
    """A summary whose runs have these mean emotions (one turn each)."""
    return summarize_runs([transcript(m) for m in means])


def test_welch_difference_by_hand() -> None:
    a, b = runs_with_means(40, 42, 44), runs_with_means(30, 35, 40)
    d = compare_runs(a, b)
    assert (d.measure, d.n_a, d.n_b, d.difference) == ("mean_emotion", 3, 3, 7)
    va, vb = 4 / 3, 25 / 3  # variance / n per group
    df = (va + vb) ** 2 / (va**2 / 2 + vb**2 / 2)
    assert d.df == pytest.approx(df)  # about 2.62: t for df 2
    half = 4.303 * math.sqrt(va + vb)
    assert d.ci95 == pytest.approx((7 - half, 7 + half))


def test_compare_another_measure() -> None:
    a = summarize_runs([transcript(40, 44, 48), transcript(40, 60)])
    b = summarize_runs([transcript(40), transcript(40)])
    d = compare_runs(a, b, measure="turns")
    assert (d.measure, d.difference) == ("turns", 1.5)


def test_compare_rejects_unknown_measures() -> None:
    with pytest.raises(ValueError, match="mean_emotion"):
        compare_runs(runs_with_means(40), runs_with_means(40), measure="happiness")


@pytest.mark.parametrize(
    ("a", "b"),
    [((40,), (30, 35)), ((40, 42), (30,)), ((40,), (30,))],
)
def test_one_run_on_a_side_gives_a_difference_without_an_interval(
    a: tuple[int, ...], b: tuple[int, ...]
) -> None:
    d = compare_runs(runs_with_means(*a), runs_with_means(*b))
    assert d.difference == pytest.approx(statistics.fmean(a) - statistics.fmean(b))
    assert (d.ci95, d.df) == (None, None)


def test_no_runs_on_a_side_gives_no_difference() -> None:
    d = compare_runs(runs_with_means(40, 42), summarize_runs([]))
    assert (d.n_a, d.n_b, d.difference, d.ci95, d.df) == (2, 0, None, None, None)


def test_no_spread_on_either_side_collapses_the_interval() -> None:
    d = compare_runs(runs_with_means(40, 40), runs_with_means(30, 30, 30))
    assert (d.difference, d.ci95, d.df) == (10, (10, 10), None)


means = st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=12)


@given(means, means)
def test_swapping_the_sides_negates_the_difference(a: list[int], b: list[int]) -> None:
    forward = compare_runs(runs_with_means(*a), runs_with_means(*b))
    backward = compare_runs(runs_with_means(*b), runs_with_means(*a))
    assert forward.difference is not None
    assert backward.difference is not None
    assert forward.difference == pytest.approx(-backward.difference)
    assert forward.ci95 is not None
    assert backward.ci95 is not None
    assert forward.ci95 == pytest.approx((-backward.ci95[1], -backward.ci95[0]))
    assert forward.ci95[0] - 1e-9 <= forward.difference <= forward.ci95[1] + 1e-9
    if forward.df is not None:  # Welch df lies between the smaller group's and the pooled df
        assert min(len(a), len(b)) - 1 - 1e-9 <= forward.df <= len(a) + len(b) - 2 + 1e-9


@given(means)
def test_a_set_compared_with_itself_differs_by_zero(a: list[int]) -> None:
    d = compare_runs(runs_with_means(*a), runs_with_means(*a))
    assert d.difference == pytest.approx(0)
    assert d.ci95 is not None
    assert d.ci95[0] <= 0 <= d.ci95[1]


# ----------------------------------------------------------------- run_repeats


def test_run_repeats_runs_with_consecutive_seeds_and_a_new_companion_each() -> None:
    made: list[EchoCompanion] = []

    def make() -> Companion:
        made.append(EchoCompanion())
        return made[-1]

    seen: list[int] = []
    runs = run_repeats(
        make_seed(),
        make,
        repeats=3,
        user_model=FakeProvider(),
        days=2,
        turns_per_day=2,
        rng_seed=5,
        on_run=lambda t: seen.append(t.rng_seed),
    )
    assert [t.rng_seed for t in runs] == seen == [5, 6, 7]
    assert len({id(c) for c in made}) == 3
    for t in runs:
        single = run(
            make_seed(),
            EchoCompanion(),
            user_model=FakeProvider(),
            days=2,
            turns_per_day=2,
            rng_seed=t.rng_seed,
        )
        assert t == single


def test_run_repeats_accepts_a_companion_class() -> None:
    runs = run_repeats(make_seed(), EchoCompanion, repeats=2, user_model=FakeProvider(), days=1)
    assert [t.rng_seed for t in runs] == [0, 1]
    assert summarize_runs(runs).runs == 2


def test_run_repeats_checks_its_arguments_before_any_model_call() -> None:
    provider = FakeProvider()
    with pytest.raises(ValueError, match="repeats"):
        run_repeats(make_seed(), EchoCompanion, repeats=0, user_model=provider)
    instance: object = EchoCompanion()
    with pytest.raises(TypeError, match="make_companion"):
        run_repeats(make_seed(), instance, repeats=2, user_model=provider)  # type: ignore[arg-type]
    assert provider.calls == []


def test_run_repeats_keeps_finished_runs_when_a_later_one_fails() -> None:
    saved: list[Transcript] = []
    calls = 0

    def make() -> Companion:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("bot is down")
        return EchoCompanion()

    with pytest.raises(RuntimeError, match="bot is down"):
        run_repeats(
            make_seed(), make, repeats=3, user_model=FakeProvider(), days=1, on_run=saved.append
        )
    assert [t.rng_seed for t in saved] == [0]
