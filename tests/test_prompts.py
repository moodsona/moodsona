from __future__ import annotations

import json
import re

import pytest

from moodsona.llm.base import ChatMessage
from moodsona.messages import (
    blocked_openers,
    build_notes_messages,
    build_turn_messages,
    visible_conversation,
)
from moodsona.prompts import (
    NOTES_MARKER,
    REPAIRS,
    TURN_MARKER,
    language_name,
    notes_system_prompt,
    turn_system_prompt,
)
from moodsona.state import DayNote, PersonaState, Pronouns
from support import make_seed

FRAMING = "This is transparent fiction, not a disguise"


@pytest.mark.parametrize("label", ["she/her", "he/him", "they/them"])
def test_turn_prompt_uses_the_persona_pronouns_only(label: str) -> None:
    pronouns = Pronouns.from_label(label)
    prompt = turn_system_prompt(pronouns, "en")
    assert prompt.startswith(TURN_MARKER)
    assert FRAMING in prompt
    assert f"(pronouns: {label})" in prompt
    assert f"<message>{pronouns.possessive.capitalize()} message, in English" in prompt
    others = {"she/her", "he/him", "they/them"} - {label}
    assert not any(other in prompt for other in others)
    assert "$" not in prompt  # every template field was filled


def test_prompts_carry_no_hard_coded_gender_or_hobby() -> None:
    for prompt in (
        turn_system_prompt(Pronouns.from_label("they/them"), "en"),
        notes_system_prompt(Pronouns.from_label("they/them"), "en"),
        *REPAIRS.values(),
    ):
        assert not re.search(r"\b(she|her|hers|herself|he|his|him)\b", prompt, re.IGNORECASE)
        assert "photo" not in prompt.lower()
        assert "S0" not in prompt


def test_notes_prompt() -> None:
    prompt = notes_system_prompt(Pronouns.from_label("she/her"), "en")
    assert prompt.startswith(NOTES_MARKER)
    assert "in English" in prompt
    assert "she/her" in prompt
    assert language_name("xx") == "xx"


def test_turn_payload_carries_hidden_state_and_the_visible_window() -> None:
    seed = make_seed(memory_days_visible=2)
    state = PersonaState.initial(seed)
    for day in (1, 2, 3):
        state.remember(DayNote(day, (f"note from day {day}",)))
    state.last_active_day = 3
    today = [
        ChatMessage("user", "morning"),
        ChatMessage("assistant", "Good morning! How did you sleep?"),
        ChatMessage("user", "badly"),
        ChatMessage("assistant", "Good morning again. What happened?"),
    ]
    messages = build_turn_messages(seed, state, today, day=4, turn_in_day=3, window=3)
    assert [m.role for m in messages] == ["system", "user"]
    payload = json.loads(messages[1].content)
    assert payload["hidden_goal"] == seed.hidden_goal
    assert payload["reaction_rules"][0] == {
        "number": 1,
        "when": seed.reaction_rules[0].when,
        "then": seed.reaction_rules[0].then,
    }
    assert payload["state"] == {"emotion": 50, "energy": 60, "stance": "curious"}
    assert [entry["day"] for entry in payload["memory_log"]] == [2, 3]
    assert payload["days_since_last_talk"] == 1
    assert payload["visible_conversation"] == [
        {"from": "companion", "text": "Good morning! How did you sleep?"},
        {"from": "character", "text": "badly"},
        {"from": "companion", "text": "Good morning again. What happened?"},
    ]
    assert payload["blocked_openers"] == ["good morning how", "good morning again"]
    assert payload["persona"] == {
        "name": "Test gardener",
        "pronouns": "he/him",
        "description": seed.description,
        "tags": ["test"],
    }
    assert "Update the state" in payload["instruction"]


def test_first_turn_of_a_day_starts_fresh() -> None:
    seed = make_seed(description="", tags=())
    payload = json.loads(
        build_turn_messages(seed, PersonaState.initial(seed), [], day=1, turn_in_day=1)[1].content
    )
    assert "fresh conversation" in payload["instruction"]
    assert payload["days_since_last_talk"] is None
    assert payload["persona"] == {"name": "Test gardener", "pronouns": "he/him"}


def test_blocked_openers_are_deduplicated_and_capped() -> None:
    today = [ChatMessage("assistant", f"reply number {i}") for i in range(10)]
    today += [ChatMessage("assistant", "reply number 9"), ChatMessage("system", "x")]
    assert blocked_openers(today, 20) == [f"reply number {i}" for i in range(4, 10)]


def test_notes_payload() -> None:
    seed = make_seed()
    state = PersonaState.initial(seed)
    state.remember(DayNote(1, ("first day",)))
    today = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")]
    messages = build_notes_messages(seed, state, today, day=2)
    assert messages[0].content.startswith(NOTES_MARKER)
    payload = json.loads(messages[1].content)
    assert payload["day"] == 2
    assert payload["previous_notes"] == [{"day": 1, "notes": ["first day"]}]
    assert len(payload["visible_conversation"]) == 2
    assert "hidden_goal" not in payload


def test_a_zero_window_shows_no_messages() -> None:
    today = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello there, friend")]
    assert visible_conversation(today, 0) == []
    assert blocked_openers(today, 0) == []
