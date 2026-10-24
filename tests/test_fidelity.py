"""Tests for the experimental fidelity check.

All conversation text here is synthetic, generated from small word lists
written for these tests.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from moodsona import fidelity
from moodsona.fidelity import (
    FEATURE_NAMES,
    _fit_logistic,
    _predict,
    _scaled,
    _standardiser,
    balanced_accuracy,
    classifier_two_sample_test,
    conversation_features,
    grouped_folds,
    message_features,
)
from moodsona.llm.base import ChatMessage

Conversation = list[ChatMessage]

_TOPICS = ("the garden", "my cousin", "the night bus", "rent", "the old bike", "a quiz show")
_FILLER = ("kind of", "honestly", "anyway", "i guess", "like", "so yeah", "whatever")


def _feature(name: str, text: str, previous: str = "") -> float:
    return message_features(text, previous)[FEATURE_NAMES.index(name)]


def _conversation(user_lines: list[str], reply: str = "Tell me more.") -> Conversation:
    history: Conversation = []
    for line in user_lines:
        history += [ChatMessage("user", line), ChatMessage("assistant", reply)]
    return history


def _mixed_line(rng: random.Random) -> str:
    """Used for both sets in the null tests: each message picks its style at random."""
    topic = rng.choice(_TOPICS)
    if rng.random() < 0.5:
        return f"{rng.choice(_FILLER)} {topic} was {rng.choice(_FILLER)} fine"
    return f"What about {topic}?"


def _rambling_line(rng: random.Random) -> str:
    return " ".join(rng.choice(_FILLER + _TOPICS) for _ in range(14))


def _curt_line(rng: random.Random) -> str:
    return f"Why {rng.choice(_TOPICS)}?"


def _conversations(
    line: Callable[[random.Random], str], rng: random.Random, count: int, turns: int
) -> list[Conversation]:
    return [_conversation([line(rng) for _ in range(turns)]) for _ in range(count)]


def _speakers(
    rng: random.Random, speakers: int, conversations: int
) -> tuple[list[Conversation], list[str]]:
    """Several conversations per speaker; each speaker has a fixed style of their own.

    Both sets draw their speakers' styles from the same population, so the sets
    differ only by which speakers they happen to contain.
    """
    out: list[Conversation] = []
    ids: list[str] = []
    for n in range(speakers):
        length, asks, lower = rng.randint(1, 12), rng.random() < 0.5, rng.random() < 0.5
        for _ in range(conversations):
            lines = []
            for _ in range(3):
                text = " ".join(rng.choice(_FILLER + _TOPICS) for _ in range(length))
                lines.append((text if lower else text.capitalize()) + ("?" if asks else "."))
            out.append(_conversation(lines))
            ids.append(f"speaker-{n}")  # the same ids in both sets: they must not collide
    return out, ids


# ------------------------------------------------------------- features ----


def test_feature_names_match_the_vector() -> None:
    assert len(FEATURE_NAMES) == len(message_features("hello"))
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_length_is_log1p_of_characters() -> None:
    assert _feature("log_length", "") == 0.0
    assert _feature("log_length", "abcd") == pytest.approx(math.log1p(4))


def test_punctuation_share() -> None:
    assert _feature("punctuation_share", "ok!!") == pytest.approx(0.5)
    assert _feature("punctuation_share", "") == 0.0
    assert _feature("punctuation_share", "no punctuation here") == 0.0


def test_question_marks_are_counted() -> None:
    assert _feature("question_marks", "why? really??") == 3.0
    assert _feature("question_marks", "fine.") == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hey there", 1.0),
        ("Hey there", 0.0),
        ("...ok", 1.0),
        ("  \u2014 Sure", 0.0),
        ("123 apples", 1.0),
        ("", 0.0),
        ("?!", 0.0),
    ],
)
def test_starts_lowercase_looks_at_the_first_letter(text: str, expected: float) -> None:
    assert _feature("starts_lowercase", text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so tired \U0001f62d", 1.0),
        ("love it \u2764", 1.0),
        ("thumbs \U0001f44d", 1.0),
        ("star \u2b50", 1.0),
        ("plain text :)", 0.0),
        ("accents \u00e9\u00fc\u00e7", 0.0),
    ],
)
def test_emoji_heuristic(text: str, expected: float) -> None:
    assert _feature("has_emoji", text) == expected


def test_type_token_ratio() -> None:
    assert _feature("type_token_ratio", "the cat the dog") == pytest.approx(3 / 4)
    assert _feature("type_token_ratio", "The the THE") == pytest.approx(1 / 3)
    assert _feature("type_token_ratio", "!!!") == 0.0


def test_first_person_share_counts_singular_pronouns_and_contractions() -> None:
    # i'm, my, me, i've, myself are first person; we, you, was, sad are not.
    text = "I'm sad, my day was long. Tell me. I\u2019ve told myself we try. you?"
    tokens = 14
    assert _feature("first_person_share", text) == pytest.approx(5 / tokens)


def test_first_person_share_skips_ambiguous_spellings() -> None:
    # "ill" and "id" are ordinary words; "im" and "ive" are counted.
    assert _feature("first_person_share", "ill id") == 0.0
    assert _feature("first_person_share", "im ive") == 1.0
    # Apostrophe look-alikes from phone keyboards count like the straight one.
    assert _feature("first_person_share", "I\u02bcm I\u2018ve") == 1.0
    assert _feature("first_person_share", "") == 0.0


def test_overlap_is_jaccard_with_the_previous_companion_message() -> None:
    assert _feature("overlap_with_companion", "the cat sat", "The cat ran") == pytest.approx(2 / 4)
    assert _feature("overlap_with_companion", "the cat sat", "") == 0.0
    assert _feature("overlap_with_companion", "", "the cat") == 0.0


def test_all_word_features_share_one_word_definition() -> None:
    # "i'm" is one word everywhere: it is not the same word as "i" and "m".
    assert _feature("overlap_with_companion", "i'm ok", "i m ok") == pytest.approx(1 / 4)
    assert _feature("overlap_with_companion", "I" + chr(0x2019) + "m ok", "i'm OK") == 1.0
    assert _feature("type_token_ratio", "i'm i m") == 1.0


def test_conversation_features_pair_each_user_message_with_the_last_assistant_one() -> None:
    conversation = [
        ChatMessage("system", "red green"),
        ChatMessage("user", "red green"),
        ChatMessage("assistant", "blue sky"),
        ChatMessage("user", "blue sky"),
        ChatMessage("user", "blue"),
        ChatMessage("assistant", "late reply"),
    ]
    rows = conversation_features(conversation)
    overlap = FEATURE_NAMES.index("overlap_with_companion")
    assert [row[overlap] for row in rows] == pytest.approx([0.0, 1.0, 0.5])


@settings(max_examples=200, deadline=None)
@given(st.text(), st.text())
def test_features_are_finite_and_in_range(text: str, previous: str) -> None:
    values = dict(zip(FEATURE_NAMES, message_features(text, previous), strict=True))
    assert all(math.isfinite(v) for v in values.values())
    assert values["log_length"] >= 0.0
    assert values["question_marks"] >= 0.0
    assert values["question_marks"] == int(values["question_marks"])
    for name in ("starts_lowercase", "has_emoji"):
        assert values[name] in (0.0, 1.0)
    for name in (
        "punctuation_share",
        "type_token_ratio",
        "first_person_share",
        "overlap_with_companion",
    ):
        assert 0.0 <= values[name] <= 1.0


# --------------------------------------------------------------- folds ----


def test_grouped_folds_keep_conversations_together_and_mix_both_sets() -> None:
    groups = [0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7]
    labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    folds = grouped_folds(groups, labels, 2, seed=3)
    by_group: dict[int, set[int]] = {}
    for group, fold in zip(groups, folds, strict=True):
        by_group.setdefault(group, set()).add(fold)
    assert all(len(f) == 1 for f in by_group.values())
    for fold in (0, 1):
        assert {label for label, f in zip(labels, folds, strict=True) if f == fold} == {0, 1}


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.integers(1, 4), min_size=3, max_size=8),
    st.lists(st.integers(1, 4), min_size=3, max_size=8),
    st.integers(2, 3),
    st.integers(0, 2**32),
)
def test_grouped_folds_never_split_a_group(
    sizes_a: list[int], sizes_b: list[int], k: int, seed: int
) -> None:
    groups: list[int] = []
    labels: list[int] = []
    sized = [(0, size) for size in sizes_a] + [(1, size) for size in sizes_b]
    for gid, (label, size) in enumerate(sized):
        groups += [gid] * size
        labels += [label] * size
    folds = grouped_folds(groups, labels, k, seed=seed)
    assert all(0 <= f < k for f in folds)
    for gid in set(groups):
        assert len({f for g, f in zip(groups, folds, strict=True) if g == gid}) == 1
    for fold in range(k):
        assert {lab for lab, f in zip(labels, folds, strict=True) if f == fold} == {0, 1}


def test_grouped_folds_balance_fold_sizes_across_labels() -> None:
    groups = list(range(14))
    labels = [0] * 7 + [1] * 7
    folds = grouped_folds(groups, labels, 5, seed=0)
    sizes = [folds.count(f) for f in range(5)]
    assert max(sizes) - min(sizes) <= 1


def test_grouped_folds_shuffle_the_labels_independently() -> None:
    groups = list(range(20))
    labels = [0] * 10 + [1] * 10
    paired = []
    for seed in range(5):
        folds = grouped_folds(groups, labels, 5, seed=seed)
        paired.append(folds[:10] == folds[10:])  # group i of each label in the same fold
    assert not all(paired)


def test_grouped_folds_reject_mixed_labels_in_a_group() -> None:
    with pytest.raises(ValueError, match="group"):
        grouped_folds([0, 0, 1, 2, 3], [0, 1, 0, 1, 1], 2, seed=0)


def test_grouped_folds_need_enough_groups_per_label() -> None:
    with pytest.raises(ValueError, match="at least"):
        grouped_folds([0, 1, 2], [0, 1, 1], 2, seed=0)


# ------------------------------------------------------------ numerics ----


def test_balanced_accuracy_averages_per_class_recall() -> None:
    # Class 0: 3 of 4 right; class 1: 1 of 1 right.
    assert balanced_accuracy([0, 0, 0, 0, 1], [0, 0, 0, 1, 1]) == pytest.approx(0.875)
    # Always predicting the majority class is chance level, not 80 %.
    assert balanced_accuracy([0, 0, 0, 0, 1], [0, 0, 0, 0, 0]) == pytest.approx(0.5)


def test_balanced_accuracy_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        balanced_accuracy([], [])
    with pytest.raises(ValueError, match="length"):
        balanced_accuracy([0, 1], [0])


@given(st.lists(st.tuples(st.integers(0, 1), st.integers(0, 1)), min_size=1))
def test_balanced_accuracy_is_in_unit_range(pairs: list[tuple[int, int]]) -> None:
    truth, predicted = zip(*pairs, strict=True)
    assert 0.0 <= balanced_accuracy(truth, predicted) <= 1.0


finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.lists(finite, min_size=3, max_size=3), min_size=1, max_size=10))
def test_standardiser_never_divides_by_zero(rows: list[list[float]]) -> None:
    means, scales = _standardiser(rows)
    assert all(s > 0 and math.isfinite(s) for s in scales)
    for row in rows:
        for value, mean, scale in zip(row, means, scales, strict=True):
            assert math.isfinite((value - mean) / scale)


def test_standardiser_leaves_constant_features_centred() -> None:
    means, scales = _standardiser([[0.1, 1.0], [0.1, 3.0], [0.1, 5.0]])
    assert scales[0] == 1.0
    assert means == pytest.approx([0.1, 3.0])
    assert scales[1] == pytest.approx(math.sqrt(8 / 3))


@given(st.lists(finite, min_size=3, max_size=3), finite, st.lists(finite, min_size=3, max_size=3))
def test_predictions_are_probabilities(weights: list[float], bias: float, row: list[float]) -> None:
    assert 0.0 <= _predict(bias, weights, row) <= 1.0


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.tuples(st.lists(finite, min_size=2, max_size=2), st.integers(0, 1)),
        min_size=2,
        max_size=12,
    ).filter(lambda data: {y for _, y in data} == {0, 1})
)
def test_fitted_model_gives_probabilities(data: list[tuple[list[float], int]]) -> None:
    rows = [row for row, _ in data]
    means, scales = _standardiser(rows)
    scaled = _scaled(rows, means, scales)
    bias, weights = _fit_logistic(scaled, [y for _, y in data])
    assert math.isfinite(bias)
    assert all(math.isfinite(w) for w in weights)
    assert all(0.0 <= _predict(bias, weights, r) <= 1.0 for r in scaled)


def test_fit_separates_a_separable_set() -> None:
    rows = [[-2.0], [-1.0], [1.0], [2.0]]
    bias, weights = _fit_logistic(rows, [0, 0, 1, 1])
    assert weights[0] > 0
    assert _predict(bias, weights, [-1.5]) < 0.5 < _predict(bias, weights, [1.5])


# ------------------------------------------------------- the whole test ----


def test_same_generator_is_indistinguishable() -> None:
    rng = random.Random(11)
    reference = _conversations(_mixed_line, rng, 12, 3)
    simulated = _conversations(_mixed_line, rng, 12, 3)
    result = classifier_two_sample_test(reference, simulated, n_permutations=19, seed=1)
    assert 0.3 <= result.accuracy <= 0.7
    assert result.p_value > 0.1


def test_different_styles_are_told_apart() -> None:
    rng = random.Random(5)
    reference = _conversations(_rambling_line, rng, 10, 3)
    simulated = _conversations(_curt_line, rng, 14, 2)
    result = classifier_two_sample_test(reference, simulated, n_permutations=19, seed=1)
    assert result.accuracy >= 0.95
    assert result.p_value == pytest.approx(1 / 20)
    assert result.reference_conversations == 10
    assert result.simulated_conversations == 14
    assert result.reference_messages == 30
    assert result.simulated_messages == 28
    assert result.folds == 5
    assert result.n_permutations == 19
    # Simulated is the positive class: questions point toward it, length away from it.
    assert result.weights["question_marks"] > 0
    assert result.weights["log_length"] < 0


def test_same_seed_same_result_and_swap_flips_the_weights() -> None:
    rng = random.Random(2)
    a = _conversations(_mixed_line, rng, 8, 2) + _conversations(_rambling_line, rng, 4, 2)
    b = _conversations(_mixed_line, rng, 6, 3) + _conversations(_curt_line, rng, 6, 2)
    first = classifier_two_sample_test(a, b, folds=3, n_permutations=5, seed=4)
    again = classifier_two_sample_test(a, b, folds=3, n_permutations=5, seed=4)
    swapped = classifier_two_sample_test(b, a, folds=3, n_permutations=5, seed=4)
    assert first == again
    assert swapped.weights["log_length"] == pytest.approx(-first.weights["log_length"])


def test_speaker_groups_stop_style_leaking_across_folds() -> None:
    rng = random.Random(0)  # the leak shows for every seed tried; 0 is not special
    reference, reference_ids = _speakers(rng, 3, 5)
    simulated, simulated_ids = _speakers(rng, 3, 5)
    options: dict[str, Any] = {"folds": 3, "n_permutations": 19, "seed": 0}
    per_conversation = classifier_two_sample_test(reference, simulated, **options)
    per_speaker = classifier_two_sample_test(
        reference,
        simulated,
        reference_groups=reference_ids,
        simulated_groups=simulated_ids,
        **options,
    )
    # Same population: only the per-speaker test keeps the p-value honest.
    assert per_conversation.p_value <= 0.05
    assert per_speaker.p_value > 0.1
    assert (per_conversation.reference_groups, per_speaker.reference_groups) == (15, 3)
    assert per_speaker.simulated_groups == 3
    assert per_speaker.reference_conversations == 15


def test_permutation_ties_survive_float_rounding(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = iter([0.8, 0.1 + 0.7, 0.0])  # 0.1 + 0.7 == 0.7999999999999999
    monkeypatch.setattr(fidelity, "_cv_accuracy", lambda *_args: next(scores))
    rng = random.Random(0)
    result = classifier_two_sample_test(
        _conversations(_mixed_line, rng, 2, 2),
        _conversations(_mixed_line, rng, 2, 2),
        folds=2,
        n_permutations=2,
    )
    assert result.p_value == pytest.approx(2 / 3)


def test_default_is_100_permutations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def chance(*args: object) -> float:
        calls.append(args)
        return 0.5

    monkeypatch.setattr(fidelity, "_cv_accuracy", chance)
    rng = random.Random(0)
    result = classifier_two_sample_test(
        _conversations(_mixed_line, rng, 2, 2), _conversations(_mixed_line, rng, 2, 2), folds=2
    )
    assert result.n_permutations == 100
    assert len(calls) == 1 + 100  # the real labels, then every shuffle
    assert result.p_value == 1.0  # every shuffle ties with the observed 0.5


def test_features_can_leave_out_the_companion_overlap() -> None:
    # Same user lines; only the companion differs, and only the overlap sees it.
    rng = random.Random(4)

    def opened_by(reply: str) -> Conversation:  # the companion speaks first and every turn
        lines = [_mixed_line(rng) for _ in range(3)]
        return [ChatMessage("assistant", reply), *_conversation(lines, reply)]

    chatty = [opened_by("what about it, was it fine?") for _ in range(10)]  # shares words
    terse = [opened_by("Tell me more.") for _ in range(10)]  # shares none
    without = [n for n in FEATURE_NAMES if n != "overlap_with_companion"]
    full = classifier_two_sample_test(chatty, terse, n_permutations=9, seed=0)
    cut = classifier_two_sample_test(chatty, terse, features=without, n_permutations=9, seed=0)
    assert full.accuracy >= 0.8
    assert cut.accuracy <= 0.7
    assert list(cut.weights) == without


def test_p_value_is_in_range() -> None:
    rng = random.Random(9)
    result = classifier_two_sample_test(
        _conversations(_mixed_line, rng, 4, 2),
        _conversations(_mixed_line, rng, 4, 2),
        folds=2,
        n_permutations=3,
        seed=0,
    )
    assert 0.0 < result.p_value <= 1.0
    assert result.p_value in {k / 4 for k in range(1, 5)}


def test_result_is_json_ready_read_only_and_hashable() -> None:
    rng = random.Random(3)
    reference = _conversations(_mixed_line, rng, 3, 2)
    simulated = _conversations(_curt_line, rng, 3, 2)
    result = classifier_two_sample_test(reference, simulated, folds=2, n_permutations=2, seed=0)
    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert list(data["weights"]) == list(FEATURE_NAMES)
    assert data["accuracy"] == result.accuracy
    with pytest.raises(TypeError):
        result.weights["log_length"] = 0.0  # type: ignore[index]
    assert hash(result) == hash(result)


def test_conversations_without_user_messages_are_ignored() -> None:
    rng = random.Random(3)
    silent: Conversation = [ChatMessage("assistant", "hello?")]
    reference = [silent, *_conversations(_mixed_line, rng, 2, 2)]
    simulated = _conversations(_curt_line, rng, 2, 2)
    result = classifier_two_sample_test(reference, simulated, folds=2, n_permutations=1, seed=0)
    assert result.reference_conversations == 2
    # The silent conversation's group id goes with it.
    grouped = classifier_two_sample_test(
        reference, simulated, reference_groups=["x", "a", "b"], folds=2, n_permutations=1
    )
    assert grouped.reference_groups == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"folds": 1}, "folds"),
        ({"n_permutations": 0}, "n_permutations"),
        ({"folds": 4}, "reference"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match=message):
        classifier_two_sample_test(
            _conversations(_mixed_line, rng, 3, 2), _conversations(_mixed_line, rng, 5, 2), **kwargs
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"features": []}, "features"),
        ({"features": ["log_length", "shoe_size"]}, "shoe_size"),
        ({"reference_groups": ["a", "b"]}, "reference_groups"),
        ({"simulated_groups": ["a"] * 6}, "simulated has 1 group"),
    ],
)
def test_invalid_features_and_groups_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match=message):
        classifier_two_sample_test(
            _conversations(_mixed_line, rng, 3, 2),
            _conversations(_mixed_line, rng, 6, 2),
            folds=2,
            **kwargs,
        )


def test_a_set_without_user_messages_is_rejected() -> None:
    rng = random.Random(0)
    silent = [[ChatMessage("assistant", "anyone there?")]] * 5
    with pytest.raises(ValueError, match="simulated"):
        classifier_two_sample_test(_conversations(_mixed_line, rng, 5, 2), silent)
