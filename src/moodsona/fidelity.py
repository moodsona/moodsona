"""EXPERIMENTAL fidelity check: can a classifier tell simulated users from real ones?

This is a first cut and its API may change. It is a statistical screen, not
proof of realism. A result near chance says only that one small classifier,
looking at a handful of surface features, could not separate the two sets; it
says nothing about meaning, safety or emotional realism.

The test is a classifier two-sample test. You give it two sets of
conversations: a *reference* set (your own real, consented data; moodsona
ships none) and a *simulated* set. Each conversation is a sequence of
:class:`~moodsona.llm.base.ChatMessage` in the visible-history convention
(``"user"`` is the person, ``"assistant"`` the companion), exactly what
:meth:`moodsona.runner.Transcript.visible_history` returns. Every user message
becomes one example, described by :data:`FEATURE_NAMES`. A logistic regression
is trained to tell the sets apart and scored on held-out conversations:

- **Grouped k-fold.** All messages of one conversation are in the same fold,
  so the classifier is never tested on a conversation it partly trained on.
  Folds are stratified: each fold holds conversations from both sets.
- **Balanced accuracy.** The score is the mean of the per-set recall on
  held-out messages, so 0.5 means "no better than chance" even when the sets
  have different sizes (always guessing the larger set scores 0.5, not its
  share of the data). Training weights each set equally for the same reason.
- **Permutation p-value.** The set labels are shuffled between conversations
  (keeping each set's number of conversations) and the whole cross-validation
  is rerun ``n_permutations`` times. The p-value is
  ``(1 + #{permuted accuracy >= observed}) / (1 + n_permutations)``.

Everything is seeded: the same inputs and seed give the same result. The cost
grows with permutations x folds x messages; a few thousand messages with the
default 200 permutations take minutes in pure Python.

Example::

    from moodsona.fidelity import classifier_two_sample_test

    result = classifier_two_sample_test(my_real_conversations, simulated, seed=0)
    print(result.accuracy, result.p_value, result.weights)
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from operator import mul
from types import MappingProxyType
from typing import Any

from moodsona.checks import words
from moodsona.lang.english import _straight_apostrophes
from moodsona.llm.base import ChatMessage

#: Names of the per-message features, in the order :func:`message_features` returns them.
FEATURE_NAMES = (
    "log_length",
    "punctuation_share",
    "question_marks",
    "starts_lowercase",
    "has_emoji",
    "type_token_ratio",
    "first_person_share",
    "overlap_with_companion",
)

# A word: letters or digits, with inner apostrophes kept ("i'm", "don't").
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
# First-person singular English forms. "id" and "ill" are left out on purpose:
# they are ordinary words more often than they are "I'd" and "I'll".
_FIRST_PERSON = frozenset(
    {"i", "me", "my", "mine", "myself", "i'm", "im", "i've", "ive", "i'd", "i'll"}
)
# Code point ranges treated as emoji: Miscellaneous Symbols and Dingbats,
# Miscellaneous Symbols and Arrows (stars, squares), and the supplementary
# planes' symbol and pictograph blocks (U+1F000-U+1FAFF, which includes the
# flags). A heuristic: it misses text emoticons like ":)" and emoji elsewhere
# (watch and hourglass at U+231A-U+23FF, (c), double exclamation), and counts
# some non-emoji symbols in those blocks (check marks, ballot boxes, arrows).
_EMOJI_RANGES = ((0x2600, 0x27BF), (0x2B00, 0x2BFF), (0x1F000, 0x1FAFF))

_ITERATIONS = 200
_LEARNING_RATE = 0.5
_L2 = 0.01


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_straight_apostrophes(text).lower())


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _EMOJI_RANGES)


def message_features(text: str, previous_companion: str = "") -> tuple[float, ...]:
    """Describe one user message with the features in :data:`FEATURE_NAMES`.

    - ``log_length``: ``log1p`` of the number of characters (>= 0), so a few
      very long messages do not dominate.
    - ``punctuation_share``: characters in a Unicode punctuation category
      (``P*``) divided by all characters, 0-1 (0 for an empty message).
    - ``question_marks``: number of ``?`` characters (a count, >= 0).
    - ``starts_lowercase``: 1 when the first letter is lower case, else 0
      (also 0 when there is no letter).
    - ``has_emoji``: 1 when a character falls in a common emoji code point
      range (a heuristic), else 0.
    - ``type_token_ratio``: distinct words / words, 0-1 (0 without words).
      Short messages score high, so it partly overlaps with length.
    - ``first_person_share``: share of words that are English first-person
      singular forms (``i``, ``me``, ``my``, ``mine``, ``myself``, ``i'm``,
      ``im``, ``i've``, ``ive``, ``i'd``, ``i'll``), 0-1.
    - ``overlap_with_companion``: Jaccard index of the lower-cased word sets
      (:func:`moodsona.checks.words`, the simulator's own repeat measure) of
      the message and ``previous_companion``, 0-1 (0 when either is empty).

    For the ratio and the first-person share, words are runs of letters and
    digits with inner apostrophes kept (curly and straight alike).

    Args:
        text: The user message.
        previous_companion: The companion message just before it ("" if none).

    Returns:
        One float per name in :data:`FEATURE_NAMES`.
    """
    chars = len(text)
    tokens = _tokens(text)
    first_letter = next((c for c in text if c.isalpha()), "")
    mine, theirs = words(text), words(previous_companion)
    union = mine | theirs
    return (
        math.log1p(chars),
        sum(unicodedata.category(c).startswith("P") for c in text) / chars if chars else 0.0,
        float(text.count("?")),
        float(first_letter.islower()),
        float(any(_is_emoji(c) for c in text)),
        len(set(tokens)) / len(tokens) if tokens else 0.0,
        sum(t in _FIRST_PERSON for t in tokens) / len(tokens) if tokens else 0.0,
        len(mine & theirs) / len(union) if mine and theirs else 0.0,
    )


def conversation_features(conversation: Sequence[ChatMessage]) -> list[tuple[float, ...]]:
    """Features of every user message in one conversation, in order.

    Each user message is paired with the closest assistant message before it
    ("" when there is none). System messages are ignored.

    Args:
        conversation: Messages in the visible-history convention.

    Returns:
        One :func:`message_features` tuple per user message.
    """
    rows: list[tuple[float, ...]] = []
    previous = ""
    for message in conversation:
        if message.role == "assistant":
            previous = message.content
        elif message.role == "user":
            rows.append(message_features(message.content, previous))
    return rows


def grouped_folds(
    groups: Sequence[Hashable], labels: Sequence[int], folds: int, *, seed: int
) -> list[int]:
    """Assign items to cross-validation folds, keeping each group together.

    Groups (conversations) are shuffled within each label and dealt round-robin
    into the folds, so every fold gets at least one group of every label. The
    shuffle of one label depends only on ``seed`` and that label's number of
    groups, so renaming the labels does not move any group.

    Args:
        groups: Group id of each item; every item of a group must share a label.
        labels: Label of each item.
        folds: Number of folds.
        seed: Seed for the shuffle.

    Returns:
        The fold index (0 to ``folds - 1``) of each item.

    Raises:
        ValueError: A group has two labels, or a label has fewer groups than ``folds``.
    """
    label_of: dict[Hashable, int] = {}
    for group, label in zip(groups, labels, strict=True):
        if label_of.setdefault(group, label) != label:
            raise ValueError(f"group {group!r} has more than one label")
    fold_of: dict[Hashable, int] = {}
    for label in sorted(set(label_of.values())):
        members = [g for g, lab in label_of.items() if lab == label]
        if len(members) < folds:
            raise ValueError(f"label {label} needs at least {folds} groups, has {len(members)}")
        random.Random(seed).shuffle(members)  # noqa: S311 - statistics, not cryptography
        fold_of.update((g, i % folds) for i, g in enumerate(members))
    return [fold_of[g] for g in groups]


def balanced_accuracy(truth: Sequence[int], predicted: Sequence[int]) -> float:
    """Mean per-class recall over the classes present in ``truth``.

    Args:
        truth: True labels.
        predicted: Predicted labels, same length.

    Returns:
        A value in 0-1; 0.5 is chance level for two classes whatever their sizes.

    Raises:
        ValueError: The inputs are empty or of different lengths.
    """
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    if not truth:
        raise ValueError("balanced accuracy of an empty set is undefined")
    recalls = []
    for label in set(truth):
        hits = [p == label for t, p in zip(truth, predicted, strict=True) if t == label]
        recalls.append(sum(hits) / len(hits))
    return sum(recalls) / len(recalls)


def _standardiser(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Per-feature mean and scale; a (near) constant feature gets scale 1."""
    means, scales = [], []
    for column in zip(*rows, strict=True):
        mean = math.fsum(column) / len(column)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in column) / len(column))
        means.append(mean)
        scales.append(std if std > 1e-9 * (1.0 + abs(mean)) else 1.0)
    return means, scales


def _predict(bias: float, weights: Sequence[float], row: Sequence[float]) -> float:
    """Probability of label 1 (the logistic function, written with tanh: no overflow)."""
    return 0.5 + 0.5 * math.tanh(0.5 * (bias + sum(map(mul, weights, row))))


def _fit_logistic(
    rows: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[float, list[float]]:
    """Batch gradient descent on class-weighted log loss with L2 on the weights.

    Each class gets half of the total sample weight, matching balanced accuracy.
    A fixed number of steps of fixed size from zero: deterministic, but not
    necessarily converged (on very unequal or separable sets the weights would
    keep growing with more steps).
    """
    count = len(labels)
    positives = sum(labels)
    sample_weight = [
        count / (2 * positives) if y else count / (2 * (count - positives)) for y in labels
    ]
    columns = list(zip(*rows, strict=True))
    bias, weights = 0.0, [0.0] * len(columns)
    for _ in range(_ITERATIONS):
        errors = [
            s * (_predict(bias, weights, row) - y)
            for row, y, s in zip(rows, labels, sample_weight, strict=True)
        ]
        bias -= _LEARNING_RATE * sum(errors) / count
        weights = [
            w - _LEARNING_RATE * (sum(map(mul, errors, column)) / count + _L2 * w)
            for w, column in zip(weights, columns, strict=True)
        ]
    return bias, weights


def _scaled(
    rows: Sequence[Sequence[float]], means: Sequence[float], scales: Sequence[float]
) -> list[list[float]]:
    return [[(v - m) / s for v, m, s in zip(row, means, scales, strict=True)] for row in rows]


def _cv_accuracy(
    rows: Sequence[tuple[float, ...]],
    groups: Sequence[int],
    group_labels: Sequence[int],
    folds: int,
    seed: int,
) -> float:
    """Balanced accuracy of held-out predictions under grouped, stratified k-fold."""
    labels = [group_labels[g] for g in groups]
    fold_of = grouped_folds(groups, labels, folds, seed=seed)
    predicted = [0] * len(rows)
    for fold in range(folds):
        train = [i for i, f in enumerate(fold_of) if f != fold]
        test = [i for i, f in enumerate(fold_of) if f == fold]
        train_rows = [rows[i] for i in train]
        means, scales = _standardiser(train_rows)
        bias, weights = _fit_logistic(
            _scaled(train_rows, means, scales), [labels[i] for i in train]
        )
        for i, row in zip(test, _scaled([rows[i] for i in test], means, scales), strict=True):
            predicted[i] = int(_predict(bias, weights, row) >= 0.5)
    return balanced_accuracy(labels, predicted)


@dataclass(frozen=True, slots=True)
class FidelityResult:
    """Outcome of :func:`classifier_two_sample_test`.

    Attributes:
        accuracy: Balanced accuracy on held-out conversations (0.5 = chance).
        p_value: Permutation p-value for "the sets cannot be told apart", in (0, 1].
        n_permutations: Label shuffles used for the p-value.
        folds: Number of cross-validation folds.
        seed: The random seed.
        reference_conversations: Reference conversations with at least one user message.
        simulated_conversations: Simulated conversations with at least one user message.
        reference_messages: User messages in the reference set.
        simulated_messages: User messages in the simulated set.
        weights: Logistic regression weight per feature, fitted on all messages
            with standardised features. Positive means "more typical of the
            simulated set"; the largest absolute values show what gives the
            simulator away. The fit has a fixed step budget, so read them as
            a ranking, not as converged estimates. Read-only.
    """

    accuracy: float
    p_value: float
    n_permutations: int
    folds: int
    seed: int
    reference_conversations: int
    simulated_conversations: int
    reference_messages: int
    simulated_messages: int
    weights: Mapping[str, float] = field(hash=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form.

        Returns:
            Every field as a plain number, with ``weights`` as a plain dict.
        """
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["weights"] = dict(self.weights)
        return data


def classifier_two_sample_test(
    reference: Sequence[Sequence[ChatMessage]],
    simulated: Sequence[Sequence[ChatMessage]],
    *,
    folds: int = 5,
    n_permutations: int = 200,
    seed: int = 0,
) -> FidelityResult:
    """Test whether a classifier can tell simulated user messages from reference ones.

    EXPERIMENTAL. See the module docstring for the method and how to read it.
    Conversations without a user message are ignored.

    Args:
        reference: Real conversations you hold and may use (moodsona ships none).
        simulated: Simulated conversations, e.g. ``Transcript.visible_history()``.
        folds: Cross-validation folds (at least 2).
        n_permutations: Label shuffles for the p-value (at least 1).
        seed: Seed for the folds and the shuffles.

    Returns:
        Balanced accuracy, p-value, sizes and per-feature weights.

    Raises:
        ValueError: ``folds`` or ``n_permutations`` is too small, or a set has
            fewer conversations with user messages than ``folds``.
    """
    if folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")
    rows: list[tuple[float, ...]] = []
    groups: list[int] = []
    group_labels: list[int] = []
    for label, (name, conversations) in enumerate(
        (("reference", reference), ("simulated", simulated))
    ):
        for conversation in conversations:
            features = conversation_features(conversation)
            if features:
                rows += features
                groups += [len(group_labels)] * len(features)
                group_labels.append(label)
        used = group_labels.count(label)
        if used < folds:
            raise ValueError(
                f"{name} has {used} conversations with user messages; "
                f"folds={folds} needs at least {folds}"
            )

    observed = _cv_accuracy(rows, groups, group_labels, folds, seed)
    rng = random.Random(seed)  # noqa: S311 - statistics, not cryptography
    shuffled = list(group_labels)
    at_least_as_good = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        if _cv_accuracy(rows, groups, shuffled, folds, seed) >= observed:
            at_least_as_good += 1

    message_labels = [group_labels[g] for g in groups]
    means, scales = _standardiser(rows)
    _, weights = _fit_logistic(_scaled(rows, means, scales), message_labels)
    return FidelityResult(
        accuracy=observed,
        p_value=(1 + at_least_as_good) / (1 + n_permutations),
        n_permutations=n_permutations,
        folds=folds,
        seed=seed,
        reference_conversations=group_labels.count(0),
        simulated_conversations=group_labels.count(1),
        reference_messages=message_labels.count(0),
        simulated_messages=message_labels.count(1),
        weights=MappingProxyType(dict(zip(FEATURE_NAMES, weights, strict=True))),
    )
