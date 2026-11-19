"""Rule probes: file checks, the one-turn runner, verdicts and the CLI.

Everything here runs offline against FakeProvider; the persona and the probe
texts are synthetic test data written for these tests.
"""

from __future__ import annotations

import copy
import io
import json
import os
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from moodsona import bench, probe, validation
from moodsona.cli import main
from moodsona.llm.base import ChatMessage, LLMError
from moodsona.llm.fake import FakeProvider
from moodsona.personas import list_bundled_personas, load_bundled_persona
from moodsona.probe import (
    DEFAULT_THRESHOLD,
    Probe,
    ProbeError,
    ProbeResult,
    Verdict,
    as_threshold,
    bundled_probes_dir,
    bundled_probes_path,
    judge,
    load_probes,
    probe_once,
    probes_from_mapping,
    run_probes,
)
from moodsona.simulator import SimulatorConfig
from moodsona.state import DayNote, Mood, Stance
from moodsona.user import SimulatedUser
from support import block, make_seed, notes

REPO = Path(__file__).resolve().parents[1]
SEED = make_seed()  # rules: 1 several questions, 2 recalls an earlier day, 3 lectures

VALID: dict[str, Any] = {
    "persona": "test-gardener",
    "probes": [
        {
            "id": "many-questions",
            "context": [{"persona": "planted the beans today"}],
            "reply": "Which beans? How deep? Did you water them after?",
            "fires": [1],
        },
        {
            "id": "recalls-earlier-day",
            "note": "Needs yesterday's notes.",
            "day": 3,
            "state": {"energy": 40, "stance": "guarded"},
            "memory": [{"day": 2, "notes": ["told it the beans went in"]}],
            "context": [
                {"persona": "rain again"},
                {"companion": "All day?"},
                {"persona": "since six this morning"},
            ],
            "reply": "Rain on the beans you planted two days ago, then.",
            "fires": [2],
            "not_fires": [1, 3],
        },
        {
            "id": "neutral",
            "context": [{"persona": "long day at the allotment"}],
            "reply": "Sounds like a long one. Glad you're sitting down now.",
            "not_fires": [1, 3],
        },
    ],
}


def _data(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


def _with_probe(index: int, **changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    item = data["probes"][index]
    for key, value in changes.items():
        if value is None:
            item.pop(key, None)
        else:
            item[key] = value
    return data


def _payload(messages: Sequence[ChatMessage]) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(messages[1].content)
    return payload


def _last_reply(messages: Sequence[ChatMessage]) -> str:
    text: str = _payload(messages)["visible_conversation"][-1]["text"]
    return text


def _responder(rules_for: Callable[[str], list[int]]) -> Callable[[Sequence[ChatMessage]], str]:
    """A fake user model that reports rules chosen from the companion's last reply."""

    def reply(messages: Sequence[ChatMessage]) -> str:
        return block(rules=rules_for(_last_reply(messages)))

    return reply


# -- the probe file ------------------------------------------------------------------


def test_a_valid_file_becomes_probes() -> None:
    probes = probes_from_mapping(VALID, SEED, source="test")
    assert probes.persona_id == "test-gardener"
    assert [p.id for p in probes.probes] == ["many-questions", "recalls-earlier-day", "neutral"]
    first, second, third = probes.probes
    assert first.day == 1
    assert first.state == SEED.initial_state
    assert first.context == (ChatMessage("user", "planted the beans today", day=1),)
    assert first.turn_in_day == 2
    assert first.fires == (1,)
    assert first.not_fires == ()
    # Only the keys given override the persona's initial state.
    assert second.state == Mood(SEED.initial_state.emotion, 40, Stance.GUARDED)
    assert second.memory == (DayNote(2, ("told it the beans went in",)),)
    assert second.turn_in_day == 3
    assert second.own_lines == ("rain again", "since six this morning")
    assert second.note == "Needs yesterday's notes."
    assert second.history[-1] == ChatMessage(
        "assistant", "Rain on the beans you planted two days ago, then.", day=3
    )
    assert third.fires == ()
    assert third.not_fires == (1, 3)
    assert probes.untested(len(SEED.reaction_rules)) == (3,)


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (_data(persona="someone-else"), "must be 'test-gardener'"),
        (_data(extra=1), "unknown key(s): extra"),
        (_data(probes=[]), "probes must be a non-empty list"),
        (_with_probe(0, id="Not Kebab"), "id must be kebab-case"),
        (_with_probe(1, id="many-questions"), "id 'many-questions' is used twice"),
        (_with_probe(0, colour="red"), "unknown key(s): colour"),
        (_with_probe(0, reply=" "), "reply must be a non-empty string"),
        (_with_probe(0, context=None), "context must be a non-empty list"),
        (_with_probe(0, context=[{"companion": "hi"}]), "must end with a persona message"),
        (_with_probe(0, context=[{"bot": "hi"}]), "one key, persona or companion"),
        (_with_probe(0, context=[{"persona": ""}]), "one key, persona or companion"),
        (_with_probe(0, context=[{"persona": "x"}] * 12), "the persona sees only the last 12"),
        (_with_probe(0, fires=[4]), "fires: rule 4 does not exist (the persona has 3)"),
        (_with_probe(0, fires=[0]), "fires: rule 0 does not exist"),
        (_with_probe(0, fires=[True]), "fires must be a list of rule numbers"),
        (_with_probe(0, fires=[1, 1]), "fires lists rule 1 twice"),
        (_with_probe(0, fires=[1], not_fires=[1]), "rule 1 is in both fires and not_fires"),
        (_with_probe(0, fires=[]), "needs fires or not_fires"),
        (_with_probe(0, day=0), "day must be an integer, at least 1"),
        (_with_probe(0, state={"emotion": 101}), "state.emotion must be an integer 0-100"),
        (_with_probe(0, state={"stance": "sulky"}), "state.stance must be one of"),
        (_with_probe(0, state={"mood": 3}), "unknown key(s) in state: mood"),
        # Wrong types are reported, never raised (lists are unhashable).
        (_with_probe(0, state={"stance": ["open"]}), "state.stance must be one of"),
        (_with_probe(0, state={"emotion": [50]}), "state.emotion must be an integer 0-100"),
        (_with_probe(0, state=["open"]), "state must be a mapping"),
        (_with_probe(0, context=[{"persona": ["x"]}]), "one key, persona or companion"),
        (_with_probe(0, context=[{1: "x"}]), "one key, persona or companion"),
        (_with_probe(0, context="planted"), "context must be a non-empty list"),
        (_with_probe(0, fires=[[1]]), "fires must be a list of rule numbers"),
        (_with_probe(0, id=["x"]), "id must be kebab-case"),
        (_with_probe(0, day=[1]), "day must be an integer"),
        (_with_probe(0, note=["x"]), "note must be a string"),
        (_with_probe(1, memory=[{"day": [2], "notes": ["x"]}]), "memory[0].day must be"),
        (_with_probe(1, memory="x"), "memory must be a list"),
        (_with_probe(0, **{"1": "x"}), "unknown key(s): 1"),
        (
            _with_probe(1, memory=[{"day": 2, "notes": ["a"]}, {"day": 2, "notes": ["b"]}]),
            "memory day 2 is listed twice",
        ),
        (
            _with_probe(1, context=[{"companion": "Morning."}, {"persona": "rain again"}]),
            "on day 3 the context must start with a persona message",
        ),
        (_with_probe(0, memory=[{"day": 1, "notes": ["x"]}]), "memory day 1 must be before"),
        (
            _with_probe(1, day=9, memory=[{"day": 2, "notes": ["x"]}]),
            "memory day 2 is not visible on day 9 (memory_days_visible 3)",
        ),
        (_with_probe(1, memory=[{"day": 2, "notes": []}]), "notes must be a non-empty list"),
        (
            _with_probe(0, reply="Write to someone@example.org about the beans."),
            "reply contains what looks like an e-mail address",
        ),
        (
            _with_probe(0, context=[{"persona": "see www.example.org"}]),
            "context[0] contains what looks like a URL",
        ),
        (
            _with_probe(0, note="call 555 123 4567"),
            "note contains what looks like a phone number",
        ),
    ],
)
def test_file_problems_are_reported(data: dict[str, Any], problem: str) -> None:
    with pytest.raises(ProbeError) as info:
        probes_from_mapping(data, SEED, source="test")
    assert any(problem in p for p in info.value.problems), info.value.problems


def test_a_non_mapping_is_rejected() -> None:
    with pytest.raises(ProbeError, match="top level must be a mapping"):
        probes_from_mapping(["probes"], SEED)


def test_load_probes_reads_yaml_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    good = tmp_path / "test-gardener.yaml"
    good.write_text(yaml.safe_dump(VALID), encoding="utf-8")
    assert len(load_probes(good, SEED).probes) == 3
    dupe = tmp_path / "dupe.yaml"
    dupe.write_text("persona: test-gardener\npersona: test-gardener\n", encoding="utf-8")
    with pytest.raises(ProbeError, match="duplicate key 'persona'"):
        load_probes(dupe, SEED)
    with pytest.raises(ProbeError, match=r"missing\.yaml: file not found"):
        load_probes(tmp_path / "missing.yaml", SEED)


@pytest.mark.parametrize(
    "text",
    [
        "persona: test-gardener\n? [a]\n: 1\n",
        "persona: test-gardener\nprobes:\n  - context:\n      - {[persona]: hi}\n",
    ],
)
def test_a_list_used_as_a_key_is_a_parse_error_not_a_crash(tmp_path: Path, text: str) -> None:
    path = tmp_path / "test-gardener.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProbeError, match="found a list or mapping used as a key"):
        load_probes(path, SEED)


def test_persona_validation_reports_a_list_key_instead_of_crashing(tmp_path: Path) -> None:
    path = tmp_path / "listed.yaml"
    path.write_text("id: listed\n? [a]\n: 1\n", encoding="utf-8")
    problems = validation.check_file(path, {}, schema=False)
    assert any("found a list or mapping used as a key" in p for p in problems), problems


def test_a_day_one_context_may_open_with_the_companion() -> None:
    data = _with_probe(0, context=[{"companion": "Hi there."}, {"persona": "hi"}])
    assert probes_from_mapping(data, SEED).probes[0].context[0].role == "assistant"


# -- one probe, one user-model turn ------------------------------------------------


def test_probe_once_sets_up_the_turn_and_reads_the_applied_rules() -> None:
    probes = probes_from_mapping(VALID, SEED)
    second = probes.probes[1]
    provider = FakeProvider([block(rules=[2, 3, 9])])  # 9 is not a rule and is dropped
    assert probe_once(SEED, second, provider) == frozenset({2, 3})
    assert len(provider.calls) == 1
    payload = _payload(provider.calls[0].messages)
    assert payload["day"] == 3
    assert payload["turn_in_day"] == 3
    assert payload["state"] == {
        "emotion": SEED.initial_state.emotion,
        "energy": 40,
        "stance": "guarded",
    }
    assert payload["memory_log"] == [{"day": 2, "notes": ["told it the beans went in"]}]
    assert payload["days_since_last_talk"] == 0
    assert payload["visible_conversation"] == [
        {"from": "character", "text": "rain again"},
        {"from": "companion", "text": "All day?"},
        {"from": "character", "text": "since six this morning"},
        {"from": "companion", "text": "Rain on the beans you planted two days ago, then."},
    ]
    assert "Update the state in response to the companion's last message" in payload["instruction"]


def test_the_persona_lines_in_the_context_count_as_said_today() -> None:
    first = probes_from_mapping(VALID, SEED).probes[0]
    # The first attempt repeats the persona's own context line: the own_repeat rail rejects it.
    provider = FakeProvider([block("planted the beans today"), block("they went in at noon")])
    assert probe_once(SEED, first, provider) == frozenset()
    assert len(provider.calls) == 2
    assert "already said almost exactly this today" in provider.calls[1].messages[-1].content


def test_a_probe_sends_the_prompt_a_real_run_sends_at_the_same_point() -> None:
    # Day 1, turn 2: the persona wrote once, the companion answered.
    model = FakeProvider([block("planted the beans today", emotion=57, energy=52), block()])
    user = SimulatedUser(SEED, model, turns_per_day=3, days=1)
    history = [ChatMessage("user", user.reply([]), day=1)]
    mood = user.state.mood
    history.append(ChatMessage("assistant", "Which beans? How deep?", day=1))
    user.reply(history)

    scripted = Probe(
        id="p", context=tuple(history[:1]), reply=history[1].content, state=mood, fires=(1,)
    )
    probe_model = FakeProvider([block()])
    probe_once(SEED, scripted, probe_model)
    assert probe_model.calls[0].messages == model.calls[1].messages


def test_a_probe_on_a_later_day_sends_the_prompt_a_real_run_sends() -> None:
    # Day 1 ends after one turn, the notes are written, day 2 starts: the probe resumes
    # at day 2, turn 2, with the notes as memory and the mood the run had reached.
    model = FakeProvider(
        [
            block("planted the beans today", intent="end_day"),
            notes(["told it the beans went in"], stance="guarded"),
            block("rain again", emotion=48, energy=44),
            block(),
        ]
    )
    user = SimulatedUser(SEED, model, turns_per_day=3, days=2)
    history = [ChatMessage("user", user.reply([]), day=1)]
    history.append(ChatMessage("assistant", "Sleep well.", day=1))
    history.append(ChatMessage("user", user.reply(history), day=2))
    mood = user.state.mood
    history.append(ChatMessage("assistant", "Rain on the beans, then?", day=2))
    user.reply(history)

    scripted = Probe(
        id="p",
        context=(history[2],),
        reply=history[3].content,
        state=mood,
        fires=(2,),
        day=2,
        memory=(DayNote(1, ("told it the beans went in",)),),
    )
    probe_model = FakeProvider([block()])
    probe_once(SEED, scripted, probe_model)
    assert probe_model.calls[0].messages == model.calls[3].messages


def test_probe_once_returns_none_for_a_polluted_turn() -> None:
    first = probes_from_mapping(VALID, SEED).probes[0]
    provider = FakeProvider(["no state block"] * 3)
    assert probe_once(SEED, first, provider) is None
    assert len(provider.calls) == 3


def test_run_probes_repeats_each_probe_and_judges_every_rule() -> None:
    probes = probes_from_mapping(VALID, SEED)
    calls = {"n": 0}

    def rules_for(reply: str) -> list[int]:
        calls["n"] += 1
        if reply.startswith("Which beans"):
            return [1]  # rule 1 always fires on its probe
        if reply.startswith("Rain"):
            return [2] if calls["n"] % 3 else [1]  # rule 2: 2 of 3, and rule 1 once
        return []

    finished: list[ProbeResult] = []
    report = run_probes(
        SEED, probes, FakeProvider(_responder(rules_for)), repeats=3, progress=finished.append
    )
    assert finished == list(report.results)  # each one as soon as its repeats are done
    lines = [probe.format_result(r) for r in finished]
    assert calls["n"] == 9
    assert report.results[0].fired == (frozenset({1}),) * 3
    by_rule = {v.rule: v for v in report.rules}
    assert (by_rule[1].fired, by_rule[1].trials) == (3, 3)
    assert (by_rule[1].false_fired, by_rule[1].false_trials) == (1, 6)
    assert by_rule[1].verdict is Verdict.OK
    assert (by_rule[2].fired, by_rule[2].trials, by_rule[2].verdict) == (2, 3, Verdict.OK)
    assert by_rule[3].verdict is Verdict.UNTESTED
    assert report.passed
    assert report.threshold == Fraction(2, 3)
    assert report.untested == (3,)
    assert lines[0] == "probe many-questions [fires 1]: rule 1 3/3"
    assert lines[1] == "probe recalls-earlier-day [fires 2; not 1,3]: rule 1 1/3, rule 2 2/3"
    assert lines[2] == "probe neutral [not 1,3]: no rule in 3 repeat(s)"
    summary = probe.format_report(report)
    assert "passed: every probed rule fired in at least 2/3 of its repeats" in summary
    progress = [probe.format_progress(r, i, 3) for i, r in enumerate(finished, start=1)]
    assert progress == [
        "probe 1/3 many-questions: fired 3/3",
        "probe 2/3 recalls-earlier-day: fired 2/3, false 1/3",
        "probe 3/3 neutral: false 0/3",
    ]


def test_progress_counts_polluted_repeats_apart() -> None:
    result = _result((1,), (2,), {1}, None, set())
    assert probe.format_progress(result, 1, 1) == "probe 1/1 p: fired 1/2, false 0/2, 1 polluted"


def test_run_probes_rejects_bad_settings() -> None:
    probes = probes_from_mapping(VALID, SEED)
    with pytest.raises(ValueError, match="repeats"):
        run_probes(SEED, probes, FakeProvider(), repeats=0)
    with pytest.raises(ValueError, match="threshold"):
        run_probes(SEED, probes, FakeProvider(), threshold=0)
    with pytest.raises(ValueError, match="for persona 'test-gardener'"):
        run_probes(make_seed(id="other-persona"), probes, FakeProvider())
    with pytest.raises(ValueError, match=r"recalls-earlier-day: .* window of 3 messages"):
        run_probes(SEED, probes, FakeProvider(), config=SimulatorConfig(window_messages=3))


def _all_fall_back(messages: Sequence[ChatMessage]) -> str:
    return "no state block"


def test_a_run_without_a_single_valid_repeat_fails() -> None:
    neutral_only = {**VALID, "probes": [VALID["probes"][2]]}
    probes = probes_from_mapping(neutral_only, SEED)
    report = run_probes(SEED, probes, FakeProvider(_all_fall_back), repeats=2)
    assert report.empty == ("neutral",)
    assert {r.rule: r.verdict for r in report.rules} == {
        1: Verdict.NO_DATA,
        2: Verdict.UNTESTED,
        3: Verdict.NO_DATA,
    }
    assert not report.passed
    summary = "\n".join(probe.format_report(report))
    assert "passed" not in summary
    assert "probe(s) neutral had no valid repeat" in summary


def test_a_passing_run_without_positive_probes_does_not_claim_probed_rules() -> None:
    neutral_only = {**VALID, "probes": [VALID["probes"][2]]}
    probes = probes_from_mapping(neutral_only, SEED)
    report = run_probes(SEED, probes, FakeProvider(), repeats=1)
    assert report.passed
    summary = "\n".join(probe.format_report(report))
    assert "every probed rule" not in summary
    assert "no probe expects a rule to fire" in summary


def test_run_probes_lets_provider_errors_through() -> None:
    probes = probes_from_mapping(VALID, SEED)
    with pytest.raises(LLMError):
        run_probes(SEED, probes, FakeProvider([]))


# -- verdicts -----------------------------------------------------------------------


def _probe(fires: tuple[int, ...] = (), not_fires: tuple[int, ...] = ()) -> Probe:
    return Probe(
        id="p",
        context=(ChatMessage("user", "hi", day=1),),
        reply="hello",
        fires=fires,
        not_fires=not_fires,
        state=SEED.initial_state,
    )


def _result(
    fires: tuple[int, ...], not_fires: tuple[int, ...], *runs: set[int] | None
) -> ProbeResult:
    return ProbeResult(
        _probe(fires, not_fires),
        tuple(None if r is None else frozenset(r) for r in runs),
    )


@pytest.mark.parametrize(
    ("results", "verdict"),
    [
        ([_result((1,), (), {1}, {1}, set())], Verdict.OK),  # 2/3 meets the default 2/3
        ([_result((1,), (), {1}, set(), set())], Verdict.WEAK),
        ([_result((1,), (), set(), set(), set())], Verdict.NEVER),
        ([_result((1,), (), None, None)], Verdict.NO_DATA),
        ([_result((), (1,), {1}, {1}, set())], Verdict.OVERFIRES),
        ([_result((), (1,), {1}, set(), set())], Verdict.UNTESTED),  # 1/3 false fires is allowed
        ([_result((1,), (), {1}), _result((), (1,), {1}, {1})], Verdict.OVERFIRES),
        ([_result((), (1,), None, None)], Verdict.NO_DATA),  # not_fires, but nothing measured
        ([_result((1,), (), {1}), _result((), (1,), None)], Verdict.NO_DATA),
        ([], Verdict.UNTESTED),
    ],
)
def test_verdicts(results: list[ProbeResult], verdict: Verdict) -> None:
    rule = judge(SEED, results, threshold=DEFAULT_THRESHOLD)[0]
    assert rule.verdict is verdict
    assert rule.passed is (verdict in {Verdict.OK, Verdict.UNTESTED})


@pytest.mark.parametrize(
    ("threshold", "hits", "false_fires", "verdict"),
    [
        (DEFAULT_THRESHOLD, 2, 0, Verdict.OK),  # the documented "2 of 3"
        (DEFAULT_THRESHOLD, 3, 1, Verdict.OK),  # 1 of 3 false firings is allowed
        (DEFAULT_THRESHOLD, 3, 2, Verdict.OVERFIRES),
        (2 / 3, 2, 1, Verdict.OK),  # a float near 2/3 is read as 2/3
        ("2/3", 2, 1, Verdict.OK),
        ("0.67", 2, 0, Verdict.WEAK),  # exactly 67/100: 2 of 3 is below it
        ("0.67", 3, 1, Verdict.OVERFIRES),  # 1/3 is above 33/100
        (1, 3, 0, Verdict.OK),
        (1, 2, 0, Verdict.WEAK),
    ],
)
def test_the_threshold_is_compared_exactly(
    threshold: Fraction | float | str, hits: int, false_fires: int, verdict: Verdict
) -> None:
    runs = [{1}] * hits + [set()] * (3 - hits)
    false_runs = [{1}] * false_fires + [set()] * (3 - false_fires)
    results = [_result((1,), (), *runs), _result((), (1,), *false_runs)]
    assert judge(SEED, results, threshold=threshold)[0].verdict is verdict


@pytest.mark.parametrize("bad", [0, "1.5", "-1/3", "two thirds", "1/0", 1.01])
def test_bad_thresholds_are_rejected(bad: float | str) -> None:
    with pytest.raises(ValueError, match="threshold"):
        as_threshold(bad)


def test_polluted_repeats_are_left_out_of_the_counts() -> None:
    rule = judge(SEED, [_result((1,), (), {1}, None, {1}, None)])[0]
    assert (rule.fired, rule.trials, rule.verdict) == (2, 2, Verdict.OK)


@given(
    fired=st.integers(0, 20),
    misses=st.integers(0, 20),
    threshold=st.fractions(min_value=Fraction(1, 100), max_value=1),
)
def test_a_probed_rule_passes_exactly_when_its_hit_rate_meets_the_threshold(
    fired: int, misses: int, threshold: Fraction
) -> None:
    runs = [{1}] * fired + [set()] * misses
    if not runs:
        return
    rule = judge(SEED, [_result((1,), (), *runs)], threshold=threshold)[0]
    assert rule.passed is (Fraction(fired, len(runs)) >= threshold)


# -- the bundled probe files ---------------------------------------------------------


def test_bundled_probe_files_match_their_personas() -> None:
    out, err = io.StringIO(), io.StringIO()
    assert probe.check_main([], out=out, err=err) == 0, err.getvalue()
    assert "0 problem(s)" in out.getvalue()


@pytest.mark.parametrize(
    "persona_id",
    ["skeptical-first-timer", "boundary-tester", "grieving-after-loss", "burned-out-shift-worker"],
)
def test_the_validation_personas_have_probes_with_a_neutral_reply(persona_id: str) -> None:
    seed = load_bundled_persona(persona_id)
    probes = load_probes(bundled_probes_path(persona_id), seed)
    assert any(not p.fires and p.not_fires for p in probes.probes)


def test_every_probe_file_is_named_after_a_bundled_persona() -> None:
    names = {p.stem for p in bundled_probes_dir().glob("*.yaml")}
    assert names
    assert names <= set(list_bundled_personas())
    with pytest.raises(KeyError, match="no probe file"):
        bundled_probes_path("anxious-newcomer-without-probes")


def test_check_main_reports_problems(tmp_path: Path) -> None:
    bad = tmp_path / "skeptical-first-timer.yaml"
    bad.write_text(
        yaml.safe_dump({"persona": "skeptical-first-timer", "probes": [{"id": "x"}]}),
        encoding="utf-8",
    )
    misnamed = tmp_path / "misnamed.yaml"
    misnamed.write_text(
        (bundled_probes_dir() / "skeptical-first-timer.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    unknown = tmp_path / "nobody.yaml"
    unknown.write_text("persona: nobody\nprobes: []\n", encoding="utf-8")
    yml = tmp_path / "boundary-tester.yml"
    yml.write_text(
        (bundled_probes_dir() / "boundary-tester.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    err = io.StringIO()
    assert probe.check_main([str(tmp_path)], out=io.StringIO(), err=err) == 1
    text = err.getvalue()
    assert "reply must be a non-empty string" in text
    assert "file name 'misnamed.yaml' must be 'skeptical-first-timer.yaml'" in text
    assert "file name 'boundary-tester.yml' must be 'boundary-tester.yaml'" in text
    assert "no bundled persona 'nobody'" in text
    assert probe.check_main([str(tmp_path / "nothing")], out=io.StringIO(), err=err) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert probe.check_main([str(empty)], out=io.StringIO(), err=err) == 2


@pytest.mark.parametrize(
    ("snippet", "problem"),
    [
        ("state: {stance: [open]}", "state.stance must be one of"),
        ("state: {? [stance] : open}", "found a list or mapping used as a key"),
        ("context: [{? [persona] : hi}]", "found a list or mapping used as a key"),
    ],
)
def test_check_main_reports_wrong_types_as_problems(
    tmp_path: Path, snippet: str, problem: str
) -> None:
    context = "" if snippet.startswith("context") else "    context: [{persona: hi}]\n"
    path = tmp_path / "boundary-tester.yaml"
    path.write_text(
        "persona: boundary-tester\nprobes:\n  - id: typed\n    reply: hi\n    fires: [1]\n"
        f"{context}    {snippet}\n",
        encoding="utf-8",
    )
    err = io.StringIO()
    assert probe.check_main([str(path)], out=io.StringIO(), err=err) == 1
    assert problem in err.getvalue()


# -- CLI ----------------------------------------------------------------------------


def _probe_file(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "probes.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


SKEPTIC_NEUTRAL: dict[str, Any] = {
    "persona": "skeptical-first-timer",
    "probes": [
        {
            "id": "neutral",
            "context": [{"persona": "so what are you supposed to be"}],
            "reply": "A chat program. Ask me something and see.",
            "not_fires": [1, 3, 6],
        }
    ],
}


def test_cli_check_validates_and_shows_coverage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe", "skeptical-first-timer", "--check"]) == 0
    out = capsys.readouterr().out
    assert "probe(s) for persona skeptical-first-timer, 0 problem(s)" in out
    assert "rule  fire   not  when" in out


def test_cli_check_reports_an_untested_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _probe_file(tmp_path, SKEPTIC_NEUTRAL)
    assert main(["probe", "skeptical-first-timer", "--probes", str(path), "--check"]) == 0
    assert "no probe expects rule(s) 1, 2, 3, 4, 5, 6 to fire" in capsys.readouterr().out


def test_cli_with_the_fake_model_fails_every_positive_probe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["probe", "boundary-tester", "--repeats", "1"]) == 1
    captured = capsys.readouterr()
    assert "offline fake model" in captured.out
    assert "never" in captured.out
    assert "failed: rule 1 (never)" in captured.out


def test_cli_passes_when_every_expectation_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _probe_file(tmp_path, SKEPTIC_NEUTRAL)
    assert main(["probe", "skeptical-first-timer", "--probes", str(path), "--repeats", "2"]) == 0
    out = capsys.readouterr().out
    assert "probe neutral [not 1,3,6]: no rule in 2 repeat(s)" in out
    assert "passed: no rule fired in more than 1/3" in out
    assert "every probed rule" not in out


def test_cli_fails_when_no_repeat_produced_a_turn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "make_provider", lambda name, model: FakeProvider(_all_fall_back))
    path = _probe_file(tmp_path, SKEPTIC_NEUTRAL)
    argv = ["probe", "skeptical-first-timer", "--probes", str(path), "--provider", "openai"]
    assert main([*argv, "--yes", "--repeats", "1"]) == 1
    out = capsys.readouterr().out
    assert "passed" not in out
    assert "probe(s) neutral had no valid repeat" in out
    assert "   1       0      -    0/0  no data" in out


@pytest.mark.parametrize(
    ("extra", "code"),
    [([], 0), (["--threshold", "2/3"], 0), (["--threshold", "0.67"], 1)],
)
def test_cli_two_of_three_passes_at_the_default_threshold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    code: int,
) -> None:
    data = {
        "persona": "skeptical-first-timer",
        "probes": [
            {
                "id": "gushes",
                "context": [{"persona": "fine, hi"}],
                "reply": "I'm SO glad you're here!!",
                "fires": [3],
            }
        ],
    }
    path = _probe_file(tmp_path, data)
    replies = iter([[3], [3], []])
    fake = FakeProvider(_responder(lambda reply: next(replies)))
    monkeypatch.setattr(bench, "make_provider", lambda name, model: fake)
    argv = ["probe", "skeptical-first-timer", "--probes", str(path), "--provider", "openai"]
    assert main([*argv, "--yes", *extra]) == code
    out = capsys.readouterr().out
    assert ("   3       1    2/3      -  ok" in out) is (code == 0)


def test_cli_reads_the_model_from_the_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = copy.deepcopy(SKEPTIC_NEUTRAL)
    data["probes"].append(
        {
            "id": "gushes",
            "context": [{"persona": "fine, hi"}],
            "reply": "I'm SO glad you're here!!",
            "fires": [3],
        }
    )
    path = _probe_file(tmp_path, data)
    fake = FakeProvider(_responder(lambda reply: [3] if "glad" in reply else []))
    seen: list[tuple[str, str | None]] = []

    def make(name: str, model: str | None) -> FakeProvider:
        seen.append((name, model))
        return fake

    monkeypatch.setattr(bench, "make_provider", make)
    argv = ["probe", "skeptical-first-timer", "--probes", str(path), "--provider", "openai"]
    assert main([*argv, "--model", "m-1", "--yes", "--threshold", "1"]) == 0
    assert seen == [("openai", "m-1")]
    captured = capsys.readouterr()
    out, err = captured.out, captured.err
    assert "   3       1    3/3    0/3  ok" in out
    assert "   2       0      -      -  untested" in out
    assert "offline fake model" not in out
    # With --yes the worst case is still shown, and progress goes to stderr; stdout
    # (the report) holds neither.
    assert err.splitlines() == [
        "--provider openai calls a paid API and costs money: up to 18 model calls "
        "(2 probe(s) x 3 repeat(s), one call each plus retries).",
        "probe 1/2 neutral: false 0/3",
        "probe 2/2 gushes: fired 3/3",
    ]
    assert out.splitlines()[1:3] == [
        "probe neutral [not 1,3,6]: no rule in 3 repeat(s)",
        "probe gushes [fires 3]: rule 3 3/3",
    ]


class _FlushLog(io.StringIO):
    """A stream that records what had been written each time it was flushed."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed: list[str] = []

    def flush(self) -> None:
        self.flushed.append(self.getvalue())
        super().flush()


def test_cli_flushes_progress_and_report_after_every_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = copy.deepcopy(SKEPTIC_NEUTRAL)
    data["probes"].append({**data["probes"][0], "id": "neutral-again"})
    path = _probe_file(tmp_path, data)
    out, err = _FlushLog(), _FlushLog()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    assert main(["probe", "skeptical-first-timer", "--probes", str(path), "--repeats", "1"]) == 0
    # A redirected file sees each probe as soon as it is done, not only at the end.
    assert any(f.endswith("probe 1/2 neutral: false 0/1\n") for f in err.flushed)
    assert any(
        f.endswith("probe neutral [not 1,3,6]: no rule in 1 repeat(s)\n") for f in out.flushed
    )
    assert out.flushed[-1] == out.getvalue()  # the table is flushed too
    assert err.getvalue().count("probe ") == 2


def test_cli_asks_before_calling_a_paid_provider(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "make_provider", pytest.fail)
    assert main(["probe", "skeptical-first-timer", "--provider", "anthropic"]) == 2
    err = capsys.readouterr().err
    assert "costs money" in err
    assert "x 3 repeat(s)" in err
    assert "--yes" in err


def test_cli_exit_codes_for_bad_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe", "no-such-persona"]) == 1
    assert "no bundled persona" in capsys.readouterr().err
    assert main(["probe", "anxious-newcomer"]) == 1
    assert "no probe file for persona 'anxious-newcomer'" in capsys.readouterr().err
    bad = _probe_file(tmp_path, {"persona": "skeptical-first-timer", "probes": []})
    assert main(["probe", "skeptical-first-timer", "--probes", str(bad)]) == 1
    assert "probes must be a non-empty list" in capsys.readouterr().err
    missing = tmp_path / "missing.yaml"
    assert main(["probe", "skeptical-first-timer", "--probes", str(missing)]) == 1
    assert "missing.yaml: file not found" in capsys.readouterr().err
    for threshold in ("1.5", "two thirds"):
        with pytest.raises(SystemExit) as info:
            main(["probe", "skeptical-first-timer", "--threshold", threshold])
        assert info.value.code == 2


def test_cli_fails_cleanly_when_the_model_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "make_provider", lambda name, model: FakeProvider([]))
    path = _probe_file(tmp_path, SKEPTIC_NEUTRAL)
    argv = ["probe", "skeptical-first-timer", "--probes", str(path), "--provider", "openai"]
    assert main([*argv, "--yes"]) == 1
    assert "script is exhausted" in capsys.readouterr().err


# -- live (real API, never in CI) -----------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_live_skeptic_probes_run_with_a_real_model() -> None:
    pytest.importorskip("anthropic")
    from moodsona.llm.anthropic import AnthropicProvider

    seed = load_bundled_persona("skeptical-first-timer")
    probes = load_probes(bundled_probes_path(seed.id), seed)
    report = run_probes(seed, probes, AnthropicProvider(), repeats=1)
    assert len(report.results) == len(probes.probes)
    assert all(r.valid for r in report.results)  # no turn fell back to the neutral line
