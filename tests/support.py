"""Shared test helpers: a synthetic persona, hidden-block builders, a stub language profile.

The persona here is synthetic test data written for these tests.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from typing import Any

from moodsona.state import Mood, PersonaSeed, Pronouns, ReactionRule, Stance

RULES = (
    ReactionRule(
        "The companion asks several questions at once.",
        "Answers one of them, briefly.",
        energy_delta=-10,
    ),
    ReactionRule(
        "The companion remembers something from an earlier day.",
        "Warms up and shares a bit more.",
        emotion_delta=8,
        energy_delta=5,
        stance_to=Stance.OPEN,
    ),
    ReactionRule(
        "The companion lectures.",
        "Goes quiet.",
        stance_to=Stance.WITHDRAWN,
    ),
)


def make_seed(**changes: Any) -> PersonaSeed:
    """A small synthetic persona; keyword arguments override fields."""
    seed = PersonaSeed(
        id="test-gardener",
        display_name="Test gardener",
        description="Retired, keeps an allotment, talks about weather and tomatoes.",
        language="en",
        pronouns=Pronouns.from_label("he/him"),
        hidden_goal="Wants someone to notice the small wins of the week.",
        reaction_rules=RULES,
        initial_state=Mood(50, 60, Stance.CURIOUS),
        baseline=Mood(55, 50, Stance.GUARDED),
        memory_days_visible=3,
        tags=("test",),
        overnight_energy_jitter=0,
    )
    return replace(seed, **changes)


def block(
    message: str = "tomatoes are finally red",
    *,
    intent: str = "reply",
    thought: str = "Fine, I will answer.",
    emotion: Any = 55,
    energy: Any = 58,
    rules: list[Any] | None = None,
) -> str:
    """A well-formed turn output."""
    state = {
        "inner_thought": thought,
        "emotion": emotion,
        "energy": energy,
        "intent": intent,
        "rules_triggered": rules or [],
    }
    return f"<state>{json.dumps(state)}</state>\n<message>{message}</message>"


def notes(lines: list[Any], stance: str | None = "curious") -> str:
    """A well-formed day-notes output."""
    data: dict[str, Any] = {"memory": lines}
    if stance is not None:
        data["stance"] = stance
    return json.dumps(data)


def keyword_hook(text: str) -> bool:
    """A toy safety hook for tests (a real deployment plugs in a crisis detector)."""
    return "CRISIS-MARKER" in text


class StubProfile:
    """Minimal language profile for tests; flags Greek letters as the wrong language."""

    code = "en"

    def is_wrong_language(self, text: str) -> bool:
        return any("\u0370" <= ch <= "\u03ff" for ch in text)

    def is_agreement(self, text: str) -> bool:
        return text.strip().lower().rstrip(".!") in {"yeah", "right", "exactly", "true"}

    def is_ask_back(self, text: str) -> bool:
        return text.strip().lower().endswith("and you?")

    def add_typing_noise(self, text: str, rng: random.Random, level: float) -> str:
        return text.lower() if rng.random() < level else text
