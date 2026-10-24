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
