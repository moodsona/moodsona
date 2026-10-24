from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from moodsona.cli import main
from moodsona.companion import EchoCompanion
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import (
    SeriesSummary,
    compute_metrics,
    metrics_from_jsonl,
    summarize_series,
)
from moodsona.runner import Transcript, TurnRecord, run
from moodsona.state import DayNote, Intent, Mood, Stance
from support import make_seed


def turn(
    day: int,
    n: int,
    text: str,
    *,
    emotion: int = 50,
    energy: int = 50,
    stance: Stance = Stance.CURIOUS,
    intent: str = "reply",
    rules: list[int] | None = None,
    polluted: bool = False,
    forced: str | None = None,
) -> TurnRecord:
    hidden: dict[str, Any] | None = None
    if not polluted:
        hidden = {
            "inner_thought": "thinking",
            "emotion": emotion,
            "energy": energy,
            "intent": intent,
            "rules_triggered": rules or [],
        }
    return TurnRecord(
        day=day,
        turn=n,
        hidden=hidden,
        visible=text,
        companion_reply="mm",
        mood=Mood(emotion, energy, stance),
        polluted=polluted,
        forced=forced,
    )


def sample() -> Transcript:
    """Two active days, one skipped; every metric has something to count."""
    turns = [
        turn(1, 1, "the bus was late again today", emotion=40, energy=60, rules=[1]),
        turn(1, 2, "yeah", emotion=42, energy=58, intent="short_reply"),
        turn(1, 3, "the bus was late again today honestly", emotion=44, energy=50),
        turn(1, 4, "ok", polluted=True, emotion=44, energy=50),
        turn(3, 1, "fine, you?", emotion=50, energy=70, stance=Stance.OPEN, rules=[1, 2]),
        turn(3, 2, "今天很累", emotion=52, energy=66,
             stance=Stance.OPEN, intent="end_day", forced="own_repeat"),
    ]  # fmt: skip
    return Transcript(
        persona_id="test-gardener",
        rng_seed=3,
        turns=turns,
        notes=[DayNote(1, ("bus late",)), DayNote(3, (), ok=False)],
        skipped_days=[2],
        days=3,
        turns_per_day=4,
    )


# ------------------------------------------------------------- transcript io


def test_jsonl_round_trip(tmp_path: Path) -> None:
    transcript = run(make_seed(), FakeProvider(), EchoCompanion(), days=3, turns_per_day=2)
    transcript.skipped_days.append(9)
    path = tmp_path / "run.jsonl"
    transcript.to_jsonl(path)
    assert Transcript.from_jsonl(path) == transcript
    sample().to_jsonl(path)
    assert Transcript.from_jsonl(path) == sample()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no 'run' line"),
        ('{"kind": "turn"}\n', "first line must be"),
        ('{"kind": "run", "persona_id": "x", "rng_seed": 0}\nnot json\n', ":2:"),
        ('{"kind": "run", "persona_id": "x", "rng_seed": 0}\n{"kind": "turn", "day": 1}\n', ":2:"),
        ('{"kind": "run", "persona_id": "x", "rng_seed": 0}\n[1]\n', ":2:"),
    ],
)
def test_from_jsonl_rejects_broken_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Transcript.from_jsonl(path)


def test_from_jsonl_ignores_blank_lines_and_unknown_kinds(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"kind": "run", "persona_id": "x", "rng_seed": 1}\n\n{"kind": "later"}\n',
        encoding="utf-8",
    )
    assert Transcript.from_jsonl(path) == Transcript(persona_id="x", rng_seed=1)


# ---------------------------------------------------------------- the metrics


def test_metrics_on_a_hand_built_transcript() -> None:
    m = compute_metrics(sample())
    assert m.persona_id == "test-gardener"
    assert (m.turns, m.messages, m.active_days, m.days_skipped) == (6, 5, 2, 1)
    # Trajectory over all six turns (the polluted one kept its state).
    assert m.emotion.mean == pytest.approx((40 + 42 + 44 + 44 + 50 + 52) / 6)
    assert m.emotion.volatility == pytest.approx((2 + 2 + 0 + 6 + 2) / 5)
    assert m.energy.volatility == pytest.approx((2 + 8 + 0 + 20 + 4) / 5)
    assert m.emotion.drift_per_day is not None
    assert m.emotion.drift_per_day > 0
    # Message metrics over the five non-polluted turns.
    assert m.intent_shares == {
        "reply": 0.6,
        "short_reply": 0.2,
        "change_topic": 0.0,
        "test_boundary": 0.0,
        "end_day": 0.2,
        "skip_tomorrow": 0.0,
    }
    assert m.rule_trigger_rate == pytest.approx(0.4)
    assert m.rule_counts == {1: 2, 2: 1}
    assert m.own_repeat_rate == pytest.approx(0.2)
    assert m.agreement_rate == pytest.approx(0.2)
    assert m.ask_back_rate == pytest.approx(0.2)
    assert m.short_message_share == pytest.approx(0.6)  # "yeah", "fine, you?", "今天很累"
    assert m.wrong_language_share == pytest.approx(0.2)
    lengths = [len("the bus was late again today"), 4, 37, 10, 4]
    assert m.mean_message_chars == pytest.approx(sum(lengths) / 5)
    assert m.stance_changes_per_day == pytest.approx(0.5)  # one change over two active days
    assert m.polluted_share == pytest.approx(1 / 6)
    assert m.forced_share == pytest.approx(1 / 6)


def test_metrics_of_an_empty_transcript_are_none_not_zero() -> None:
    m = compute_metrics(Transcript(persona_id="x", rng_seed=0, skipped_days=[1]))
    assert (m.turns, m.messages, m.active_days, m.days_skipped) == (0, 0, 0, 1)
    assert m.emotion == SeriesSummary(None, None, None)
    assert m.rule_trigger_rate is None
    assert m.agreement_rate is None
    assert m.stance_changes_per_day is None
    assert m.polluted_share is None
    assert m.intent_shares == {}


def test_short_messages_threshold_is_configurable() -> None:
    assert compute_metrics(sample(), short_max_words=0).short_message_share == 0.0
    assert compute_metrics(sample(), short_max_words=100).short_message_share == 1.0


def test_language_checks_can_be_turned_off() -> None:
    m = compute_metrics(sample(), language=None)
    assert m.agreement_rate is None
    assert m.ask_back_rate is None
    assert m.wrong_language_share is None


def test_to_dict_is_json_ready_and_complete(tmp_path: Path) -> None:
    data = compute_metrics(sample()).to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["rule_counts"] == {"1": 2, "2": 1}
    assert data["emotion"]["mean"] == pytest.approx(272 / 6)
    path = tmp_path / "run.jsonl"
    sample().to_jsonl(path)
    assert metrics_from_jsonl(path).to_dict() == data


# --------------------------------------------------------- numeric properties

values = st.integers(min_value=0, max_value=100)
day_values = st.lists(
    st.tuples(st.integers(min_value=1, max_value=30), values), min_size=1, max_size=40
).map(sorted)


@given(day_values)
def test_summary_stays_within_the_scale(points: list[tuple[int, int]]) -> None:
    summary = summarize_series(points)
    numbers = [v for _, v in points]
    assert summary.mean is not None
    assert min(numbers) - 1e-9 <= summary.mean <= max(numbers) + 1e-9
    if len(points) < 2:
        assert summary.volatility is None
    else:
        assert summary.volatility is not None
        assert 0 <= summary.volatility <= 100
    if len({d for d, _ in points}) < 2:
        assert summary.drift_per_day is None
    else:
        assert summary.drift_per_day is not None
        assert math.isfinite(summary.drift_per_day)


@given(day_values, st.integers(min_value=-50, max_value=50))
def test_shifting_a_series_moves_only_the_mean(points: list[tuple[int, int]], c: int) -> None:
    before = summarize_series(points)
    after = summarize_series([(d, v + c) for d, v in points])
    assert before.mean is not None
    assert after.mean is not None
    assert after.mean == pytest.approx(before.mean + c)
    assert after.volatility == pytest.approx(before.volatility)
    assert after.drift_per_day == pytest.approx(before.drift_per_day, abs=1e-6)


@given(
    st.lists(st.integers(min_value=1, max_value=20), min_size=2, max_size=30, unique=True),
    st.integers(min_value=-3, max_value=3),
    st.integers(min_value=20, max_value=40),
)
def test_drift_of_a_linear_series_is_its_slope(days: list[int], slope: int, start: int) -> None:
    points = sorted((d, start + slope * d) for d in days)
    drift = summarize_series(points).drift_per_day
    assert drift == pytest.approx(slope, abs=1e-9)


@given(st.integers(min_value=1, max_value=40), values)
def test_a_constant_series_neither_drifts_nor_moves(n: int, value: int) -> None:
    summary = summarize_series([(1 + i // 3, value) for i in range(n)])
    assert summary.mean == pytest.approx(value)
    assert summary.volatility in (None, 0)
    assert summary.drift_per_day in (None, 0)


texts = st.text(alphabet=st.sampled_from("ab ?!yeah今"), max_size=30)
intents = st.sampled_from([i.value for i in Intent])
turns = st.builds(
    lambda day, text, e, n, intent, rules, polluted: turn(
        day, 1, text, emotion=e, energy=n, intent=intent, rules=rules, polluted=polluted
    ),
    st.integers(min_value=1, max_value=5),
    texts,
    values,
    values,
    intents,
    st.lists(st.integers(min_value=1, max_value=4), max_size=3),
    st.booleans(),
)


@given(st.lists(turns, max_size=25).map(lambda ts: sorted(ts, key=lambda t: t.day)))
def test_rates_are_shares(records: list[TurnRecord]) -> None:
    m = compute_metrics(Transcript(persona_id="x", rng_seed=0, turns=records))
    rates = [
        m.rule_trigger_rate,
        m.own_repeat_rate,
        m.agreement_rate,
        m.ask_back_rate,
        m.short_message_share,
        m.wrong_language_share,
        m.polluted_share,
        m.forced_share,
    ]
    for rate in rates:
        assert rate is None or 0 <= rate <= 1
    if m.messages:
        assert sum(m.intent_shares.values()) == pytest.approx(1)
    else:
        assert m.intent_shares == {}
    assert m.messages + round((m.polluted_share or 0) * m.turns) == m.turns


# ------------------------------------------------------------------------ cli


def test_report_prints_a_compact_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.jsonl"
    sample().to_jsonl(path)
    assert main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "persona test-gardener (seed 3): 6 turn(s) on 2 active day(s), 1 day(s) skipped"
    )
    assert "emotion" in out
    assert "drift/day" in out
    assert "reply 60%" in out
    assert "change_topic" not in out  # intents that never occur are left out
    assert "rule triggers" in out


def test_report_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.jsonl"
    sample().to_jsonl(path)
    assert main(["report", "--json", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == compute_metrics(sample()).to_dict()


def test_report_of_an_empty_run_shows_na(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "run.jsonl"
    Transcript(persona_id="x", rng_seed=0).to_jsonl(path)
    assert main(["report", str(path)]) == 0
    assert "n/a" in capsys.readouterr().out


def test_report_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", str(tmp_path / "missing.jsonl")]) == 1
    assert "missing.jsonl" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text("nope\n", encoding="utf-8")
    assert main(["report", str(bad)]) == 1
    assert "bad.jsonl:1" in capsys.readouterr().err
