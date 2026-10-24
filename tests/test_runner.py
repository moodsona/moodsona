from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from moodsona.companion import EchoCompanion, ScriptedCompanion
from moodsona.llm.base import ChatMessage
from moodsona.llm.fake import FakeProvider, plumbing_reply
from moodsona.prompts import NOTES_MARKER
from moodsona.runner import TrajectoryPoint, run
from support import block, make_seed, notes


def scheduled(messages: Sequence[ChatMessage]) -> str:
    """Plumbing output, except day 2 ends at turn 2 and day 3 skips day 4 at turn 1."""
    if messages[0].content.startswith(NOTES_MARKER):
        return plumbing_reply(messages)
    payload = json.loads(messages[1].content)
    day, turn = payload["day"], payload["turn_in_day"]
    state = payload["state"]
    intent = {(2, 2): "end_day", (3, 1): "skip_tomorrow"}.get((day, turn), "reply")
    return block(
        f"day {day} line {turn} about the weather",
        intent=intent,
        thought=f"secret thought {day}-{turn}",
        emotion=state["emotion"] + 1,
        energy=state["energy"] - 1,
    )


def test_run_carries_state_across_days_and_honours_intents() -> None:
    provider = FakeProvider(scheduled)
    companion = ScriptedCompanion(["Mm, tell me more."], cycle=True)
    transcript = run(make_seed(), provider, companion, days=5, turns_per_day=3, rng_seed=1)

    per_day = {d: sum(1 for t in transcript.turns if t.day == d) for d in range(1, 6)}
    assert per_day == {1: 3, 2: 2, 3: 1, 4: 0, 5: 3}
    assert transcript.skipped_days == [4]
    assert [n.day for n in transcript.notes] == [1, 2, 3, 5]

    # Day 2's first prompt reads day 1's notes; day 5 knows it has been away two days.
    turn_payloads = [
        json.loads(call.messages[1].content)
        for call in provider.calls
        if not call.messages[0].content.startswith(NOTES_MARKER)
    ]
    day2 = next(p for p in turn_payloads if p["day"] == 2 and p["turn_in_day"] == 1)
    assert [entry["day"] for entry in day2["memory_log"]] == [1]
    day5 = next(p for p in turn_payloads if p["day"] == 5 and p["turn_in_day"] == 1)
    assert day5["days_since_last_talk"] == 2

    # The hidden block never reaches the companion.
    seen = "\n".join(m.content for history in companion.histories for m in history)
    assert "secret thought" not in seen
    assert "<state>" not in seen
    assert all(t.hidden and "secret thought" in t.hidden["inner_thought"] for t in transcript.turns)
    assert transcript.visible_history()[0] == ChatMessage("user", "day 1 line 1 about the weather")
    # The companion sees the whole history, across days.
    assert len(companion.histories[-1]) == 2 * len(transcript.turns) - 1


def test_trajectory_and_jsonl(tmp_path: Path) -> None:
    transcript = run(make_seed(), FakeProvider(), EchoCompanion(), days=2, turns_per_day=2)
    points = transcript.trajectory()
    assert points[0] == TrajectoryPoint(1, 1, 51, 58, "curious")
    assert [(p.day, p.turn) for p in points] == [(1, 1), (1, 2), (2, 1), (2, 2)]

    out = tmp_path / "run.jsonl"
    transcript.to_jsonl(out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [line["kind"] for line in lines] == ["run"] + ["turn"] * 4 + ["day_notes"] * 2
    first = lines[1]
    assert set(first) >= {"day", "turn", "hidden", "visible", "companion_reply", "state_after"}
    assert first["companion_reply"] == "You said: " + first["visible"]
    assert first["state_after"] == {"emotion": 51, "energy": 58, "stance": "curious"}


def test_runs_are_reproducible_from_their_seed() -> None:
    seed = make_seed(overnight_energy_jitter=15)

    def once(rng_seed: int) -> list[TrajectoryPoint]:
        return run(seed, FakeProvider(), EchoCompanion(), 3, 2, rng_seed).trajectory()

    assert once(7) == once(7)
    assert any(once(7) != once(other) for other in (1, 2, 3))


def test_polluted_turns_are_recorded() -> None:
    provider = FakeProvider(["junk"] * 3 + [notes(["meh"])])
    transcript = run(make_seed(), provider, EchoCompanion(), days=1, turns_per_day=1)
    record = transcript.turns[0]
    assert record.polluted
    assert record.hidden is None
    assert record.problems == ("hidden_state_invalid",) * 3
    assert record.as_dict()["hidden"] is None


@pytest.mark.parametrize(("days", "turns"), [(0, 1), (1, 0)])
def test_run_rejects_empty_runs(days: int, turns: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run(make_seed(), FakeProvider(), EchoCompanion(), days, turns)


def test_run_rejects_an_unknown_language_before_any_model_call() -> None:
    provider = FakeProvider()
    with pytest.raises(LookupError, match="No language profile for 'xx'"):
        run(make_seed(language="xx"), provider, EchoCompanion(), 1, 1)
    assert provider.calls == []


def test_run_with_the_english_profile() -> None:
    lang = pytest.importorskip("moodsona.lang")
    profile = lang.get_profile("en")
    transcript = run(make_seed(), FakeProvider(), EchoCompanion(), 1, 3, language=profile)
    assert len(transcript.turns) == 3
    assert not any(t.polluted for t in transcript.turns)


def test_jsonl_keeps_the_run_header_and_skipped_days(tmp_path: Path) -> None:
    transcript = run(
        make_seed(), FakeProvider(scheduled), EchoCompanion(), days=5, turns_per_day=3, rng_seed=1
    )
    out = tmp_path / "run.jsonl"
    transcript.to_jsonl(out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {
        "kind": "run",
        "persona_id": "test-gardener",
        "rng_seed": 1,
        "days": 5,
        "turns_per_day": 3,
    }
    by_kind: dict[str, list[dict[str, object]]] = {}
    for line in lines:
        by_kind.setdefault(line["kind"], []).append(line)
    assert set(by_kind) == {"run", "turn", "day_notes", "skipped_day"}
    # Everything the transcript knows about the run can be read back.
    assert [line["day"] for line in by_kind["skipped_day"]] == transcript.skipped_days == [4]
    assert len(by_kind["turn"]) == len(transcript.turns)
    assert [line["day"] for line in by_kind["day_notes"]] == [n.day for n in transcript.notes]
