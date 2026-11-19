from __future__ import annotations

import json
import math
from dataclasses import replace
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

RUN = '{"kind": "run", "persona_id": "x", "rng_seed": 0}\n'


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
    transcript = run(
        make_seed(), EchoCompanion(), user_model=FakeProvider(), days=3, turns_per_day=2
    )
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
        (RUN + "not json\n", ":2:"),
        (RUN + '{"kind": "turn", "day": 1}\n', ":2:"),
        (RUN + "[1]\n", "not a JSON object"),
        (RUN + RUN, "second 'run' line"),
    ],
)
def test_from_jsonl_rejects_broken_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Transcript.from_jsonl(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hidden": [1]}, "hidden must be an object or null"),
        ({"hidden": {"intent": "reply", "rules_triggered": [1, "2"]}}, "list of integers"),
        ({"polluted": "false"}, "polluted must be true or false"),
        ({"problems": "too_long"}, "problems must be a list"),
        ({"forced": 3}, "forced must be a string or null"),
        ({"rules_applied": {"rule": 1}}, "rules_applied must be a list"),
        ({"rules_applied": [{"rule": "1", "repeat": 1}]}, "rules_applied"),
        ({"rules_applied": [{"repeat": 1}]}, "rule"),
        ({"rules_applied": [{"rule": 0, "repeat": 1}]}, "at least 1"),
        ({"rules_applied": [{"rule": 1, "repeat": 0}]}, "at least 1"),
    ],
)
def test_from_jsonl_checks_field_types(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    line = {**sample().turns[0].as_dict(), **change}
    path = tmp_path / "bad.jsonl"
    path.write_text(RUN + json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Transcript.from_jsonl(path)


@pytest.mark.parametrize(
    "notes", [{"notes": "bus late", "ok": True}, {"notes": ["bus late"], "ok": "yes"}]
)
def test_from_jsonl_checks_day_notes(tmp_path: Path, notes: dict[str, Any]) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(RUN + json.dumps({"kind": "day_notes", "day": 1, **notes}), encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        Transcript.from_jsonl(path)


def test_jsonl_round_trip_keeps_unicode_line_separators(tmp_path: Path) -> None:
    transcript = sample()
    transcript.turns[0] = replace(transcript.turns[0], companion_reply="one\u2028two\x85three")
    path = tmp_path / "run.jsonl"
    transcript.to_jsonl(path)
    assert Transcript.from_jsonl(path) == transcript


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
    # Everything below is over the five messages; the polluted turn is left out.
    assert m.emotion.mean == pytest.approx((40 + 42 + 44 + 50 + 52) / 5)
    assert m.emotion.volatility == pytest.approx((2 + 2 + 6 + 2) / 4)
    assert m.energy.volatility == pytest.approx((2 + 8 + 20 + 4) / 4)
    assert m.emotion.drift_per_day is not None
    assert m.emotion.drift_per_day > 0
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


def test_outcome_numbers_on_a_hand_built_transcript() -> None:
    m = compute_metrics(sample())
    assert m.emotion.mean == pytest.approx(228 / 5)  # mean emotion: the primary outcome
    assert (m.end_emotion, m.final_stance) == (52, "open")  # the last message, day 3
    # turns_per_day is 4: day 1 has four turns (its polluted one counts), day 3 ends after two.
    assert m.days_ended_early == 1
    assert m.emotion_change is None  # the start is not in the transcript; pass the persona


def test_emotion_change_is_measured_from_the_personas_initial_state() -> None:
    m = compute_metrics(sample(), persona=make_seed())  # initial emotion 50
    assert m.emotion_change == 2
    started_low = make_seed(initial_state=Mood(30, 60, Stance.GUARDED))
    assert compute_metrics(sample(), persona=started_low).emotion_change == 22


def test_the_persona_must_be_the_one_the_transcript_simulated() -> None:
    with pytest.raises(ValueError, match="other-persona"):
        compute_metrics(sample(), persona=make_seed(id="other-persona"))


def test_end_of_run_skips_a_trailing_polluted_turn() -> None:
    transcript = sample()
    transcript.turns.append(turn(3, 3, "ok", polluted=True, emotion=10, stance=Stance.HOSTILE))
    m = compute_metrics(transcript, persona=make_seed())
    assert (m.end_emotion, m.final_stance, m.emotion_change) == (52, "open", 2)


def test_only_polluted_turns_have_no_end_of_run() -> None:
    transcript = Transcript(
        persona_id="test-gardener", rng_seed=0, turns=[turn(1, 1, "ok", polluted=True)]
    )
    m = compute_metrics(transcript, persona=make_seed())
    assert (m.end_emotion, m.final_stance, m.emotion_change) == (None, None, None)


@pytest.mark.parametrize(
    ("turns_per_day", "expected"),
    [(None, None), (4, 1), (2, 0), (6, 2)],
)
def test_days_ended_early_needs_the_turn_limit(
    turns_per_day: int | None, expected: int | None
) -> None:
    transcript = replace(sample(), turns_per_day=turns_per_day)
    assert compute_metrics(transcript).days_ended_early == expected


def test_skipped_days_do_not_count_as_ended_early() -> None:
    transcript = Transcript(persona_id="x", rng_seed=0, skipped_days=[1, 2], turns_per_day=3)
    m = compute_metrics(transcript)
    assert (m.days_ended_early, m.days_skipped) == (0, 2)


def test_unknown_intents_are_counted_as_other() -> None:
    transcript = sample()
    transcript.turns[1] = turn(1, 2, "yeah", intent="sing_a_song")
    shares = compute_metrics(transcript).intent_shares
    assert shares["other"] == pytest.approx(0.2)
    assert sum(shares.values()) == pytest.approx(1)


def test_metrics_of_an_empty_transcript_are_none_not_zero() -> None:
    m = compute_metrics(Transcript(persona_id="x", rng_seed=0, skipped_days=[1]))
    assert (m.turns, m.messages, m.active_days, m.days_skipped) == (0, 0, 0, 1)
    assert m.emotion == SeriesSummary(None, None, None)
    assert (m.end_emotion, m.emotion_change, m.final_stance) == (None, None, None)
    assert m.days_ended_early is None  # turns_per_day is unknown
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
    assert data["emotion"]["mean"] == pytest.approx(228 / 5)
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


@given(
    st.lists(turns, max_size=25).map(lambda ts: sorted(ts, key=lambda t: t.day)),
    st.integers(min_value=1, max_value=8),
)
def test_end_of_run_is_the_last_message(records: list[TurnRecord], turns_per_day: int) -> None:
    transcript = Transcript(
        persona_id="test-gardener", rng_seed=0, turns=records, turns_per_day=turns_per_day
    )
    m = compute_metrics(transcript, persona=make_seed())
    messages = [t for t in records if not t.polluted]
    if messages:
        assert m.end_emotion == messages[-1].mood.emotion
        assert m.final_stance == messages[-1].mood.stance.value
        assert m.emotion_change == m.end_emotion - make_seed().initial_state.emotion
        assert m.emotion.mean is not None
        assert min(t.mood.emotion for t in messages) <= m.emotion.mean
    else:
        assert (m.end_emotion, m.final_stance, m.emotion_change) == (None, None, None)
    assert m.days_ended_early is not None
    assert 0 <= m.days_ended_early <= m.active_days


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
    assert "end of run     emotion 52, stance open" in out.splitlines()
    assert "ended early    1 of 2 active day(s)" in out.splitlines()


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
