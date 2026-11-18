from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from moodsona.state import (
    MAX_FREE_STEP,
    DayNote,
    Mood,
    PersonaState,
    Pronouns,
    ReactionRule,
    Stance,
    clamp_score,
    overnight,
    react,
    repeat_delta,
    repeated_rule,
)
from support import RULES, make_seed

scores = st.integers(min_value=0, max_value=100)
any_ints = st.integers(min_value=-1000, max_value=1000)
stances = st.sampled_from(list(Stance))
deltas = st.one_of(st.none(), st.integers(min_value=-50, max_value=50))
rules = st.builds(
    ReactionRule,
    when=st.just("when x"),
    then=st.just("then y"),
    emotion_delta=deltas,
    energy_delta=deltas,
    stance_to=st.one_of(st.none(), stances),
)
moods = st.builds(Mood, scores, scores, stances)


@given(st.floats(min_value=-1e6, max_value=1e6))
def test_clamp_score_is_always_in_range(value: float) -> None:
    assert 0 <= clamp_score(value) <= 100


@given(any_ints, any_ints, stances)
def test_clamped_mood_is_always_valid(emotion: int, energy: int, stance: Stance) -> None:
    mood = Mood.clamped(emotion, energy, stance)
    assert 0 <= mood.emotion <= 100
    assert 0 <= mood.energy <= 100


@given(moods, any_ints, any_ints)
def test_shifted_stays_in_range_and_moves_toward_the_delta(mood: Mood, de: int, dn: int) -> None:
    moved = mood.shifted(emotion=de, energy=dn)
    assert 0 <= moved.emotion <= 100
    assert moved.emotion == max(0, min(100, mood.emotion + de))
    assert moved.energy == max(0, min(100, mood.energy + dn))
    assert moved.stance == mood.stance


@pytest.mark.parametrize("bad", [-1, 101, True])
def test_mood_rejects_out_of_range_values(bad: int) -> None:
    with pytest.raises((ValueError, TypeError)):
        Mood(bad, 50, Stance.OPEN)
    with pytest.raises(TypeError):
        Mood(50, 50.5, Stance.OPEN)  # type: ignore[arg-type]


@given(moods, scores, scores, st.lists(rules, max_size=4))
def test_react_always_returns_a_valid_mood(
    mood: Mood, emotion: int, energy: int, triggered: list[ReactionRule]
) -> None:
    after = react(mood, proposed_emotion=emotion, proposed_energy=energy, triggered=triggered)
    assert 0 <= after.emotion <= 100
    assert 0 <= after.energy <= 100


@given(moods, scores, scores)
def test_without_rules_the_model_moves_at_most_the_free_step(
    mood: Mood, emotion: int, energy: int
) -> None:
    after = react(mood, proposed_emotion=emotion, proposed_energy=energy)
    assert abs(after.emotion - mood.emotion) <= MAX_FREE_STEP
    assert abs(after.energy - mood.energy) <= MAX_FREE_STEP
    if abs(emotion - mood.emotion) <= MAX_FREE_STEP:
        assert after.emotion == emotion
    assert after.stance == mood.stance


@given(moods, scores, scores, st.lists(rules, min_size=1, max_size=4))
def test_rule_deltas_win_over_the_model_per_dimension(
    mood: Mood, emotion: int, energy: int, triggered: list[ReactionRule]
) -> None:
    after = react(mood, proposed_emotion=emotion, proposed_energy=energy, triggered=triggered)
    e_deltas = [r.emotion_delta for r in triggered if r.emotion_delta is not None]
    if e_deltas:
        assert after.emotion == max(0, min(100, mood.emotion + sum(e_deltas)))
    n_deltas = [r.energy_delta for r in triggered if r.energy_delta is not None]
    if n_deltas:
        assert after.energy == max(0, min(100, mood.energy + sum(n_deltas)))
    targets = [r.stance_to for r in triggered if r.stance_to is not None]
    assert after.stance == (targets[-1] if targets else mood.stance)


def test_react_example_from_a_persona_rule() -> None:
    mood = Mood(50, 60, Stance.GUARDED)
    # Rule 2 gives both deltas and a stance; the model's proposal is ignored for both.
    after = react(mood, proposed_emotion=10, proposed_energy=10, triggered=[RULES[1]])
    assert after == Mood(58, 65, Stance.OPEN)
    # Rule 1 covers energy only: emotion follows the model, capped at the free step.
    after = react(mood, proposed_emotion=95, proposed_energy=95, triggered=[RULES[0]])
    assert after == Mood(70, 50, Stance.GUARDED)


rule_deltas = st.integers(min_value=-50, max_value=50)
decays = st.floats(min_value=0.0, max_value=1.0)
repeats = st.integers(min_value=1, max_value=40)


def test_repeat_delta_halves_and_rounds_toward_zero() -> None:
    assert [repeat_delta(10, repeat=k, decay=0.5) for k in range(1, 7)] == [10, 5, 2, 1, 0, 0]
    assert [repeat_delta(-15, repeat=k, decay=0.5) for k in range(1, 5)] == [-15, -7, -3, -1]
    assert [repeat_delta(10, repeat=k, decay=0.0) for k in (1, 2, 3)] == [10, 0, 0]
    # Float noise does not cost a point: 10 * 0.7 is 7, not 6.
    assert repeat_delta(10, repeat=2, decay=0.7) == 7


@pytest.mark.parametrize(("repeat", "decay"), [(0, 0.5), (1, -0.1), (1, 1.5), (1, float("nan"))])
def test_repeat_delta_rejects_bad_arguments(repeat: int, decay: float) -> None:
    with pytest.raises(ValueError, match=r"^(repeat|decay) must be"):
        repeat_delta(10, repeat=repeat, decay=decay)


@given(rule_deltas, repeats, decays)
def test_a_repeat_never_grows_or_flips_the_delta(delta: int, repeat: int, decay: float) -> None:
    applied = repeat_delta(delta, repeat=repeat, decay=decay)
    assert abs(applied) <= abs(delta)
    assert applied == 0 or (applied > 0) == (delta > 0)
    assert abs(repeat_delta(delta, repeat=repeat + 1, decay=decay)) <= abs(applied)
    assert repeat_delta(delta, repeat=1, decay=decay) == delta


@given(rule_deltas, repeats)
def test_decay_one_is_the_old_full_delta(delta: int, repeat: int) -> None:
    assert repeat_delta(delta, repeat=repeat, decay=1.0) == delta


@given(rules, repeats, moods, scores, scores)
def test_decay_one_reacts_exactly_as_before(
    rule: ReactionRule, repeat: int, mood: Mood, emotion: int, energy: int
) -> None:
    assert repeated_rule(rule, repeat=repeat, decay=1.0) == rule
    old = react(mood, proposed_emotion=emotion, proposed_energy=energy, triggered=[rule])
    new = react(
        mood,
        proposed_emotion=emotion,
        proposed_energy=energy,
        triggered=[repeated_rule(rule, repeat=repeat, decay=1.0)],
    )
    assert new == old


@given(rule_deltas, st.floats(min_value=0.0, max_value=0.95), st.integers(1, 200))
def test_one_rule_moves_a_dimension_a_bounded_amount_per_day(
    delta: int, decay: float, times: int
) -> None:
    total = sum(repeat_delta(delta, repeat=k, decay=decay) for k in range(1, times + 1))
    # A geometric series: never more than delta / (1 - decay), however often it fires.
    assert abs(total) <= abs(delta) / (1 - decay) + 1e-6


@given(rules, decays, st.integers(1, 30), moods, scores, scores)
def test_repeating_a_rule_keeps_the_state_in_range_and_its_stance(
    rule: ReactionRule, decay: float, times: int, mood: Mood, emotion: int, energy: int
) -> None:
    for k in range(1, times + 1):
        applied = repeated_rule(rule, repeat=k, decay=decay)
        for authored, now in (
            (rule.emotion_delta, applied.emotion_delta),
            (rule.energy_delta, applied.energy_delta),
        ):
            if authored is None:
                assert now is None
            elif authored == 0:
                assert now == 0  # an authored 0 always holds the scale
            else:
                # A non-zero delta either still moves the scale or, decayed away, releases it.
                assert now is None or now != 0
                assert (now is None) == (repeat_delta(authored, repeat=k, decay=decay) == 0)
        assert applied.stance_to == rule.stance_to  # the stance change always applies in full
        mood = react(mood, proposed_emotion=emotion, proposed_energy=energy, triggered=[applied])
        assert 0 <= mood.emotion <= 100
        assert 0 <= mood.energy <= 100


def test_a_delta_decayed_to_zero_releases_the_scale_but_an_authored_zero_holds() -> None:
    drain = ReactionRule("when x", "then y", energy_delta=-10)
    hold = ReactionRule("when x", "then y", energy_delta=0)
    mood = Mood(50, 50, Stance.OPEN)
    fifth = repeated_rule(drain, repeat=5, decay=0.5)  # -10, -5, -2, -1, then 0
    assert fifth.energy_delta is None
    after = react(mood, proposed_emotion=50, proposed_energy=10, triggered=[fifth])
    assert after.energy == 50 - MAX_FREE_STEP  # the model's own move, capped
    held = repeated_rule(hold, repeat=5, decay=0.5)
    assert held.energy_delta == 0
    assert react(mood, proposed_emotion=50, proposed_energy=10, triggered=[held]).energy == 50


def test_rule_uses_are_counted_per_day_and_reset_overnight() -> None:
    seed = make_seed()
    state = PersonaState.initial(seed)
    state.begin_day(seed, day=1)
    assert [state.use_rule(2), state.use_rule(2), state.use_rule(1), state.use_rule(2)] == [
        1,
        2,
        1,
        3,
    ]
    state.begin_day(seed, day=1)  # the same day again: nothing resets
    assert state.use_rule(2) == 4
    state.begin_day(seed, day=2)
    assert [state.use_rule(2), state.use_rule(1)] == [1, 1]


@given(moods, moods, st.integers(min_value=-40, max_value=40), st.booleans())
def test_overnight_is_valid_and_moves_emotion_toward_the_baseline(
    mood: Mood, baseline: Mood, offset: int, remembers: bool
) -> None:
    morning = overnight(mood, baseline, energy_offset=offset, remembers=remembers)
    assert abs(morning.emotion - baseline.emotion) <= abs(mood.emotion - baseline.emotion)
    assert morning.energy == max(0, min(100, baseline.energy + offset))
    assert morning.stance == (mood.stance if remembers else baseline.stance)


def test_begin_day_keeps_the_first_morning_and_applies_the_night_after() -> None:
    seed = make_seed()
    state = PersonaState.initial(seed)
    state.begin_day(seed, day=1, energy_offset=7)
    assert state.mood == seed.initial_state
    state.mood = Mood(15, 5, Stance.HOSTILE)
    state.begin_day(seed, day=2, energy_offset=-4)
    assert state.mood == Mood(35, 46, Stance.HOSTILE)
    state.begin_day(seed, day=2, energy_offset=30)  # once per day
    assert state.mood.energy == 46


def test_without_memory_the_stance_resets_overnight() -> None:
    seed = make_seed(memory_days_visible=0)
    state = PersonaState.initial(seed)
    state.begin_day(seed, day=1)
    state.mood = Mood(20, 20, Stance.HOSTILE)
    state.begin_day(seed, day=3)
    assert state.mood.stance == seed.baseline.stance


def test_memory_window_and_notes_are_stored_once() -> None:
    state = PersonaState.initial(make_seed())
    assert state.days_since_last_talk(4) is None
    state.last_active_day = 2
    assert state.days_since_last_talk(5) == 3
    for day in (1, 2, 3, 4):
        state.remember(DayNote(day, (f"note {day}",)))
    state.remember(DayNote(4, ("duplicate",)), stance=Stance.HOSTILE)
    state.remember(DayNote(5, (), ok=False), stance=Stance.OPEN)
    assert [n.day for n in state.visible_memory(day=6, days_visible=3)] == [3, 4]
    assert [n.day for n in state.visible_memory(day=5, days_visible=3)] == [2, 3, 4]
    assert state.mood.stance == Stance.OPEN
    assert len(state.memory_log) == 5


def test_pronouns() -> None:
    assert Pronouns.from_label("she/her").possessive == "her"
    assert Pronouns.from_label("he/him").object == "him"
    assert Pronouns.from_label("they/them").possessive == "their"
    with pytest.raises(ValueError, match="unsupported pronouns"):
        Pronouns.from_label("xe/xem")
