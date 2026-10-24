"""Build the chat messages sent to the simulator model.

Each call is one system prompt (from :mod:`moodsona.prompts`) plus one user
message holding a JSON payload: the persona card, the hidden state, the memory
log and the visible conversation of the current day. The companion never sees
any of this; it only receives the visible messages.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from moodsona.checks import opener
from moodsona.llm.base import ChatMessage
from moodsona.prompts import notes_system_prompt, turn_system_prompt
from moodsona.state import PersonaSeed, PersonaState

#: How many companion openers are shown as "do not start like this".
MAX_BLOCKED_OPENERS = 6


def persona_card(seed: PersonaSeed) -> dict[str, Any]:
    """The persona as the simulator model sees it (no hidden goal, no rules)."""
    card: dict[str, Any] = {"name": seed.display_name, "pronouns": seed.pronouns.label}
    if seed.description:
        card["description"] = seed.description
    if seed.tags:
        card["tags"] = list(seed.tags)
    return card


def _last(today: Sequence[ChatMessage], window: int) -> Sequence[ChatMessage]:
    """The last ``window`` messages; none when ``window <= 0`` (``today[-0:]`` is all)."""
    return today[-window:] if window > 0 else ()


def visible_conversation(today: Sequence[ChatMessage], window: int) -> list[dict[str, str]]:
    """The last ``window`` visible messages of the day, labelled by speaker."""
    return [
        {"from": "character" if m.role == "user" else "companion", "text": m.content}
        for m in _last(today, window)
        if m.role != "system"
    ]


def blocked_openers(today: Sequence[ChatMessage], window: int) -> list[str]:
    """Openings of the companion's recent messages, so the persona does not mirror them."""
    found: list[str] = []
    for message in _last(today, window):
        start = opener(message.content) if message.role == "assistant" else None
        if start and start not in found:
            found.append(start)
    return found[-MAX_BLOCKED_OPENERS:]


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_turn_messages(
    seed: PersonaSeed,
    state: PersonaState,
    today: Sequence[ChatMessage],
    *,
    day: int,
    turn_in_day: int,
    window: int = 12,
) -> list[ChatMessage]:
    """Messages for one simulated-user turn.

    Args:
        seed: The persona.
        state: The persona's current state (mood and memory log).
        today: Visible messages of the current day so far (``user`` = persona).
        day: Day number, starting at 1.
        turn_in_day: Turn number within the day, starting at 1.
        window: How many of today's messages to show.

    Returns:
        A system message and a user message with the JSON payload.
    """
    memory = state.visible_memory(day=day, days_visible=seed.memory_days_visible)
    payload: dict[str, Any] = {
        "day": day,
        "turn_in_day": turn_in_day,
        "persona": persona_card(seed),
        "hidden_goal": seed.hidden_goal,
        "reaction_rules": [
            {"number": n, "when": rule.when, "then": rule.then}
            for n, rule in enumerate(seed.reaction_rules, start=1)
        ],
        "state": {
            "emotion": state.mood.emotion,
            "energy": state.mood.energy,
            "stance": state.mood.stance.value,
        },
        "memory_log": [{"day": note.day, "notes": list(note.lines)} for note in memory],
        "days_since_last_talk": state.days_since_last_talk(day),
        "visible_conversation": visible_conversation(today, window),
        "blocked_openers": blocked_openers(today, window),
    }
    if turn_in_day == 1:
        payload["instruction"] = (
            "Start a fresh conversation for today with something concrete from the "
            "character's day. Set emotion and energy for how the character starts."
        )
    else:
        payload["instruction"] = (
            "Update the state in response to the companion's last message, then write the "
            "next message the character would genuinely send now."
        )
    return [
        ChatMessage("system", turn_system_prompt(seed.pronouns, seed.language)),
        ChatMessage("user", _dump(payload)),
    ]


def build_notes_messages(
    seed: PersonaSeed, state: PersonaState, today: Sequence[ChatMessage], *, day: int
) -> list[ChatMessage]:
    """Messages asking for the persona's end-of-day notes.

    Args:
        seed: The persona.
        state: The state at the end of the day.
        today: All visible messages of the day.
        day: The day that just ended.

    Returns:
        A system message and a user message with the JSON payload.
    """
    previous = state.visible_memory(day=day, days_visible=min(3, seed.memory_days_visible))
    payload = {
        "day": day,
        "persona": persona_card(seed),
        "previous_notes": [{"day": note.day, "notes": list(note.lines)} for note in previous],
        "state_at_end_of_day": {
            "emotion": state.mood.emotion,
            "energy": state.mood.energy,
            "stance": state.mood.stance.value,
        },
        "visible_conversation": visible_conversation(today, len(today)),
    }
    return [
        ChatMessage("system", notes_system_prompt(seed.pronouns, seed.language)),
        ChatMessage("user", _dump(payload)),
    ]
