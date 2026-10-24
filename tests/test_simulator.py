from __future__ import annotations

import random

import pytest

from moodsona.lang import EnglishProfile
from moodsona.llm.base import ChatMessage
from moodsona.llm.fake import FakeProvider
from moodsona.prompts import HIDDEN_REPAIR, REPAIRS
from moodsona.simulator import Simulator, SimulatorConfig
from moodsona.state import Mood, Stance
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
