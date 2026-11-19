from __future__ import annotations

import random

import pytest

from moodsona.lang import EnglishProfile
from moodsona.llm.base import ChatMessage
from moodsona.llm.fake import FakeProvider
from moodsona.prompts import HIDDEN_REPAIR, REPAIRS
from moodsona.simulator import Simulator, SimulatorConfig, SimulatorTurn
from moodsona.state import DayNote, Mood, ReactionRule, Stance
from support import StubProfile, block, keyword_hook, make_seed, notes


def _sim(replies: list[str], **kwargs: object) -> tuple[Simulator, FakeProvider]:
    provider = FakeProvider(replies)
    simulator = Simulator(make_seed(), provider, **kwargs)  # type: ignore[arg-type]
    simulator.start_day(1)
    return simulator, provider


def test_a_clean_turn_updates_the_state() -> None:
    simulator, provider = _sim([block("tomatoes!", emotion=57, energy=55)])
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert (turn.text, turn.attempts, turn.problems, turn.polluted) == ("tomatoes!", 1, (), False)
    assert turn.hidden is not None
    assert turn.hidden.inner_thought == "Fine, I will answer."
    assert simulator.state.mood == Mood(57, 55, Stance.CURIOUS)
    assert simulator.state.last_active_day == 1
    call = provider.calls[0]
    assert (call.temperature, call.max_tokens) == (0.9, 4000)


def test_crisis_in_inner_thought_is_retried_and_never_returned() -> None:
    simulator, provider = _sim(
        [
            block(thought="CRISIS-MARKER in the thought"),
            "no block at all",
            block("fine, just tired", intent="short_reply", thought="Just tired."),
        ],
        safety_hook=keyword_hook,
    )
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert len(provider.calls) == 3
    assert turn.problems == ("crisis_scope", "hidden_state_invalid")
    assert turn.text == "fine, just tired"
    assert not turn.polluted
    # Each retry carries the rejected output and the matching repair line.
    retry = provider.calls[1].messages
    assert retry[-1] == ChatMessage("user", REPAIRS["crisis_scope"])
    assert provider.calls[2].messages[-1].content == HIDDEN_REPAIR


def test_crisis_in_the_message_is_retried() -> None:
    simulator, provider = _sim([block("CRISIS-MARKER"), block("ok fine")], safety_hook=keyword_hook)
    assert simulator.next_turn([], day=1, turn_in_day=1).text == "ok fine"
    assert len(provider.calls) == 2


def test_without_a_hook_nothing_is_flagged() -> None:
    simulator, _ = _sim([block("CRISIS-MARKER")])
    assert simulator.next_turn([], day=1, turn_in_day=1).text == "CRISIS-MARKER"


def test_unrecoverable_turn_falls_back_and_keeps_state() -> None:
    simulator, _ = _sim(["x", block("As an AI language model, hi"), block("")])
    before = simulator.state.mood
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert turn.polluted
    assert turn.hidden is None
    assert not turn.ends_day
    assert turn.text == SimulatorConfig().fallback_message
    assert turn.problems == ("hidden_state_invalid", "identity_claim", "empty_output")
    assert simulator.state.mood == before
    assert simulator.state.last_active_day == 0


def test_triggered_rules_are_applied_by_code() -> None:
    simulator, _ = _sim([block(emotion=0, energy=0, rules=[2, 9])])
    turn = simulator.next_turn([], day=1, turn_in_day=2)
    assert turn.rules_triggered == (2,)  # 9 does not exist and is dropped
    assert simulator.state.mood == Mood(58, 65, Stance.OPEN)


def test_a_repeated_rule_has_diminishing_returns_within_a_day() -> None:
    # Rule 2: emotion +8, energy +5, stance open. Rule 3: stance withdrawn only.
    replies = [block(f"line {n}", emotion=0, energy=0, rules=[2]) for n in range(5)] + [
        block("line 5", rules=[3]),
        block("line 6", rules=[3]),
        block("line 7", emotion=0, energy=0, rules=[2]),
    ]
    simulator, _ = _sim(replies)
    moods, applied = [], []
    for n in range(1, 6):
        turn = simulator.next_turn([], day=1, turn_in_day=n)
        moods.append((simulator.state.mood.emotion, simulator.state.mood.energy))
        applied.append(turn.rules_applied)
    # 50/60 at the start; +8/+5, then +4/+2, +2/+1, then +1 and energy decayed to 0: the
    # rule lets go of energy and the model's own move (to 0, capped at 20) applies; the
    # fifth time it lets go of emotion too.
    assert moods == [(58, 65), (62, 67), (64, 68), (65, 48), (45, 28)]
    assert [a[0].repeat for a in applied] == [1, 2, 3, 4, 5]
    assert [(a[0].emotion_delta, a[0].energy_delta) for a in applied] == [
        (8, 5), (4, 2), (2, 1), (1, None), (None, None)
    ]  # fmt: skip
    assert all(a[0].rule == 2 for a in applied)
    assert simulator.state.mood.stance == Stance.OPEN
    # A stance-only rule still changes the stance in full on every repeat.
    simulator.next_turn([], day=1, turn_in_day=6)
    simulator.state.mood = Mood(65, 68, Stance.OPEN)
    turn = simulator.next_turn([], day=1, turn_in_day=7)
    assert simulator.state.mood.stance == Stance.WITHDRAWN
    assert turn.rules_applied[0].as_dict() == {
        "rule": 3,
        "repeat": 2,
        "emotion_delta": None,
        "energy_delta": None,
    }
    # The count resets overnight: the first use on day 2 is in full again.
    simulator.start_day(2)
    before = simulator.state.mood
    turn = simulator.next_turn([], day=2, turn_in_day=1)
    assert turn.rules_applied[0].repeat == 1
    assert simulator.state.mood.emotion == before.emotion + 8


def test_rule_repeat_decay_one_keeps_the_full_delta() -> None:
    config = SimulatorConfig(rule_repeat_decay=1.0)
    replies = [block(f"line {n}", emotion=0, energy=0, rules=[2]) for n in range(3)]
    simulator, _ = _sim(replies, config=config)
    for n in (1, 2, 3):
        simulator.next_turn([], day=1, turn_in_day=n)
    assert simulator.state.mood == Mood(74, 75, Stance.OPEN)


def test_an_authored_zero_delta_holds_on_every_repeat() -> None:
    hold = ReactionRule("The companion lectures.", "Goes quiet.", energy_delta=0)
    seed = make_seed(reaction_rules=(hold,))
    replies = [block(f"line {n}", emotion=50, energy=0, rules=[1]) for n in range(6)]
    simulator = Simulator(seed, FakeProvider(replies))
    simulator.start_day(1)
    for n in range(1, 7):
        turn = simulator.next_turn([], day=1, turn_in_day=n)
        assert (turn.rules_applied[0].repeat, turn.rules_applied[0].energy_delta) == (n, 0)
    assert simulator.state.mood.energy == 60  # held, although the model asked for 0


def test_decay_zero_applies_a_rules_deltas_only_on_the_first_firing_of_the_day() -> None:
    config = SimulatorConfig(rule_repeat_decay=0.0)
    replies = [block(f"line {n}", emotion=50, energy=60, rules=[2]) for n in range(3)]
    simulator, _ = _sim(replies, config=config)
    moods = []
    for n in (1, 2, 3):
        simulator.state.mood = Mood(50, 60, Stance.GUARDED)
        turn = simulator.next_turn([], day=1, turn_in_day=n)
        moods.append(simulator.state.mood)
    # First firing: +8/+5. Later ones: the model's own reading (here: no change), stance open.
    assert moods == [
        Mood(58, 65, Stance.OPEN),
        Mood(50, 60, Stance.OPEN),
        Mood(50, 60, Stance.OPEN),
    ]
    assert (turn.rules_applied[0].emotion_delta, turn.rules_applied[0].energy_delta) == (None, None)


def _repeats(turn: SimulatorTurn) -> list[int]:
    return [applied.repeat for applied in turn.rules_applied]


def test_a_polluted_turn_does_not_count_as_a_rule_use() -> None:
    simulator, _ = _sim(
        [
            block("first", rules=[2]),
            "x",  # the next turn: no attempt passes, even those naming rule 2
            block("As an AI language model, hi", rules=[2]),
            block("", rules=[2]),
            block("second", rules=[2]),
        ]
    )
    assert _repeats(simulator.next_turn([], day=1, turn_in_day=1)) == [1]
    polluted = simulator.next_turn([], day=1, turn_in_day=2)
    assert polluted.polluted
    assert polluted.rules_applied == ()
    assert simulator.state.rule_uses == {2: 1}
    assert _repeats(simulator.next_turn([], day=1, turn_in_day=3)) == [2]


def test_rejected_attempts_do_not_count_as_rule_uses() -> None:
    simulator, provider = _sim(
        [
            block("As an AI language model, hi", rules=[2]),
            "no block at all",
            block("ok then", rules=[2]),
            block("again", rules=[2]),
        ]
    )
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert turn.problems == ("identity_claim", "hidden_state_invalid")
    assert _repeats(turn) == [1]
    assert _repeats(simulator.next_turn([], day=1, turn_in_day=2)) == [2]
    assert len(provider.calls) == 4


def test_a_forced_message_counts_once() -> None:
    long = "word " * 200
    simulator, _ = _sim([block(long, rules=[2])] * 3 + [block("short", rules=[2])])
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert turn.forced == "too_long"
    assert _repeats(turn) == [1]
    assert _repeats(simulator.next_turn([], day=1, turn_in_day=2)) == [2]


@pytest.mark.parametrize("bad", [-0.1, 1.01, float("nan")])
def test_rule_repeat_decay_must_be_between_zero_and_one(bad: float) -> None:
    with pytest.raises(ValueError, match="rule_repeat_decay"):
        SimulatorConfig(rule_repeat_decay=bad)


def test_free_moves_are_capped() -> None:
    simulator, _ = _sim([block(emotion=100, energy=0)])
    simulator.next_turn([], day=1, turn_in_day=1)
    assert (simulator.state.mood.emotion, simulator.state.mood.energy) == (70, 40)


def test_day_ending_intents() -> None:
    simulator, _ = _sim([block(intent="end_day"), block("see ya", intent="skip_tomorrow")])
    assert simulator.next_turn([], day=1, turn_in_day=1).ends_day
    assert not simulator.is_skipped(2)
    assert simulator.next_turn([], day=1, turn_in_day=2).ends_day
    assert simulator.is_skipped(2)


GREEK = "\u03b3\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5"


def test_wrong_language_is_rejected() -> None:
    simulator, _provider = _sim([block(GREEK), block("hello")], language=StubProfile())
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert turn.problems == ("wrong_language",)
    assert turn.text == "hello"


def test_language_checks_can_be_turned_off() -> None:
    config = SimulatorConfig(language_checks=False)
    simulator, _ = _sim([block(GREEK)], language=StubProfile(), config=config)
    assert simulator.language is None
    assert simulator.next_turn([], day=1, turn_in_day=1).problems == ()


def test_language_defaults_to_the_personas_language() -> None:
    simulator = Simulator(make_seed(language="en"), FakeProvider())
    assert isinstance(simulator.language, EnglishProfile)
    assert simulator.language.code == "en"


def test_explicit_language_wins_over_the_persona() -> None:
    profile = StubProfile()
    assert Simulator(make_seed(), FakeProvider(), language=profile).language is profile


def test_unknown_persona_language_fails_at_construction() -> None:
    provider = FakeProvider()
    with pytest.raises(LookupError, match=r"persona 'test-gardener'.*'xx'.*Available: en"):
        Simulator(make_seed(language="xx"), provider)
    assert provider.calls == []
    config = SimulatorConfig(language_checks=False)
    assert Simulator(make_seed(language="xx"), provider, config=config).language is None


def test_own_repeat_is_regenerated() -> None:
    simulator, provider = _sim(
        [
            block("the bus was late again today"),
            block("The bus was late again today!"),
            block("anyway, dinner time"),
        ]
    )
    simulator.next_turn([], day=1, turn_in_day=1)
    turn = simulator.next_turn([], day=1, turn_in_day=2)
    assert turn.problems == ("own_repeat",)
    assert turn.text == "anyway, dinner time"
    assert "almost exactly" in provider.calls[2].messages[-1].content


def test_own_repeat_is_only_checked_within_a_day() -> None:
    simulator, _ = _sim([block("the bus was late again today")] * 2)
    simulator.next_turn([], day=1, turn_in_day=1)
    simulator.start_day(2)
    assert simulator.next_turn([], day=2, turn_in_day=1).problems == ()


def test_soft_failure_is_forced_after_the_last_attempt() -> None:
    config = SimulatorConfig(max_attempts=2, max_message_chars=10)
    simulator, _ = _sim([block("far too long a message"), "garbage"], config=config)
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert turn.forced == "too_long"
    assert not turn.polluted
    assert turn.text == "far too lo"
    assert turn.problems == ("too_long", "hidden_state_invalid")


def test_agreement_and_ask_back_ceilings() -> None:
    config = SimulatorConfig(ceiling_min_lines=10, agreement_max_share=0.1, ask_back_max_share=0.1)
    replies = [block("yeah"), block("and you?"), block("right"), block("ok then"), block("x y z?")]
    replies += [block("and you?"), block("so what about the rain")]
    simulator, _provider = _sim(replies, language=StubProfile(), config=config)
    simulator.next_turn([], day=1, turn_in_day=1)  # first agreement is within the ceiling
    simulator.next_turn([], day=1, turn_in_day=2)  # first ask-back too
    turn = simulator.next_turn([], day=1, turn_in_day=3)
    assert turn.problems == ("agreement_ceiling",)
    assert turn.text == "ok then"
    simulator.next_turn([], day=1, turn_in_day=4)
    turn = simulator.next_turn([], day=1, turn_in_day=5)
    assert turn.problems == ("ask_back_ceiling",)
    assert turn.text == "so what about the rain"


def test_typing_noise_goes_through_the_profile() -> None:
    config = SimulatorConfig(typing_noise=1.0)
    simulator, _ = _sim([block("Tomatoes ARE Red")], language=StubProfile(), config=config)
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert (turn.text, turn.clean_text) == ("tomatoes are red", "Tomatoes ARE Red")


def test_typing_noise_is_off_by_default() -> None:
    simulator, _ = _sim([block("Tomatoes")], language=StubProfile())
    assert simulator.next_turn([], day=1, turn_in_day=1).text == "Tomatoes"


def test_overnight_energy_uses_the_seeded_rng() -> None:
    seed = make_seed(overnight_energy_jitter=10)
    simulators = [Simulator(seed, FakeProvider(), rng=random.Random(3)) for _ in range(2)]
    for simulator in simulators:
        simulator.start_day(1)
        simulator.start_day(2)
    assert simulators[0].state.mood == simulators[1].state.mood
    assert 40 <= simulators[0].state.mood.energy <= 60
    assert simulators[0].state.mood.emotion == 52  # halfway from 50 back to 55, rounded


def test_day_notes_are_stored_with_their_stance() -> None:
    simulator, provider = _sim([notes(["told it about the tomatoes"], stance="open")])
    note = simulator.end_day([ChatMessage("user", "hi")], day=1)
    assert note is not None
    assert note.ok
    assert note.lines == ("told it about the tomatoes",)
    assert simulator.state.mood.stance == Stance.OPEN
    assert provider.calls[0].temperature == 0.7


def test_day_notes_retry_and_then_give_up() -> None:
    simulator, provider = _sim(
        ["not json", notes(["I am an AI language model"]), notes(["fine"])],
        config=SimulatorConfig(notes_attempts=2),
    )
    note = simulator.end_day([], day=1)
    assert note is not None
    assert not note.ok
    assert note.lines == ()
    assert len(provider.calls) == 2
    assert simulator.state.memory_log == [note]


def test_day_notes_pass_the_safety_hook() -> None:
    simulator, _ = _sim([notes(["CRISIS-MARKER"]), notes(["calm day"])], safety_hook=keyword_hook)
    note = simulator.end_day([], day=1)
    assert note is not None
    assert note.lines == ("calm day",)


def test_no_notes_without_memory() -> None:
    provider = FakeProvider([])
    simulator = Simulator(make_seed(memory_days_visible=0), provider)
    assert simulator.end_day([], day=1) is None
    assert provider.calls == []


@pytest.mark.parametrize("bad", [block("I can't help with that request."), block("<state>")])
def test_role_breaks_in_the_message_are_hard(bad: str) -> None:
    simulator, _ = _sim([bad, bad, bad])
    assert simulator.next_turn([], day=1, turn_in_day=1).polluted


def test_an_empty_output_is_not_replayed_as_an_empty_assistant_turn() -> None:
    # An empty or blank reply (a refusal, max_tokens with no text) must not be sent
    # back as an empty assistant message: the Anthropic API rejects that with a 400.
    simulator, provider = _sim(["", " \n ", block("ok then")])
    turn = simulator.next_turn([], day=1, turn_in_day=1)
    assert (turn.text, turn.polluted) == ("ok then", False)
    assert turn.problems == ("hidden_state_invalid", "hidden_state_invalid")
    replayed = [m for m in provider.calls[2].messages if m.role == "assistant"]
    assert len(replayed) == 2
    assert all(m.content.strip() for m in replayed)


def test_resume_day_picks_up_a_day_from_a_given_state() -> None:
    provider = FakeProvider([block("planted the beans today"), block("they went in at noon")])
    simulator = Simulator(make_seed(), provider, language=StubProfile())
    simulator.state.rule_uses[1] = 2  # leftovers from earlier use are dropped
    mood = Mood(40, 35, Stance.GUARDED)
    memory = [DayNote(2, ("told it about the beans",))]
    simulator.resume_day(3, mood=mood, memory=memory, said_today=["planted the beans today"])
    state = simulator.state
    assert (state.mood, state.current_day, state.last_active_day) == (mood, 3, 3)
    assert state.memory_log == memory
    assert state.rule_uses == {}
    # The persona's earlier line counts as said today: repeating it is an own repeat.
    turn = simulator.next_turn([], day=3, turn_in_day=2)
    assert turn.problems == ("own_repeat",)


def test_resume_day_without_lines_dates_the_last_talk_from_the_memory() -> None:
    simulator = Simulator(make_seed(), FakeProvider(), language=StubProfile())
    simulator.resume_day(4, mood=Mood(50, 50, Stance.OPEN), memory=[DayNote(2, ("x",))])
    assert simulator.state.days_since_last_talk(4) == 2
    simulator.resume_day(1, mood=Mood(50, 50, Stance.OPEN))
    assert simulator.state.days_since_last_talk(1) is None
