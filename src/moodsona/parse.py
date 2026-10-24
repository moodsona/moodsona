"""Parse the simulator model's output.

A turn output must hold a ``<state>{...}</state>`` hidden block followed by a
``<message>...</message>``. The parser is forgiving about form (code fences
around the JSON, a missing closing tag, numbers sent as strings, out-of-range
scores) and strict about content (a known intent and a non-empty inner
thought). Anything it cannot use raises :class:`ParseError`, and the simulator
asks the model again.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from moodsona.state import Intent, Stance, clamp_score

_STATE = re.compile(r"<state>\s*(.*?)\s*</state>", re.DOTALL | re.IGNORECASE)
_MESSAGE = re.compile(r"<message>\s*(.*?)\s*(?:</message>|$)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
#: Wrapping quote pairs (opening, closing) that :func:`clean_message` removes.
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u00ab", "\u00bb"))
MAX_NOTE_LINES = 5


class ParseError(ValueError):
    """The model output could not be used."""


@dataclass(frozen=True, slots=True)
class HiddenBlock:
    """The simulated user's private state for one turn (never sent to the companion).

    Attributes:
        inner_thought: One or two first-person sentences.
        emotion: Proposed emotion after the turn, clamped to 0-100.
        energy: Proposed energy after the turn, clamped to 0-100.
        intent: What the persona does with this message.
        rules_triggered: 1-based numbers of the reaction rules the companion's
            last message triggered, as the model reported them (unchecked).
    """

    inner_thought: str
    emotion: int
    energy: int
    intent: Intent
    rules_triggered: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "inner_thought": self.inner_thought,
            "emotion": self.emotion,
            "energy": self.energy,
            "intent": self.intent.value,
            "rules_triggered": list(self.rules_triggered),
        }


def _score(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ParseError(f"{name} is a boolean")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError as exc:
            raise ParseError(f"{name} is not a number") from exc
    if not isinstance(value, int | float):
        raise ParseError(f"{name} is not a number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:  # an int too large for a float
        raise ParseError(f"{name} is too large") from exc
    if not finite:
        raise ParseError(f"{name} is not a finite number")
    return clamp_score(value)


def _rule_numbers(value: Any) -> tuple[int, ...]:
    """Keep the plausible rule numbers; ignore anything else (it only adds noise)."""
    if not isinstance(value, list):
        return ()
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        # ASCII only: str.isdigit() also accepts "²", which int() rejects.
        if isinstance(item, str) and item.strip().isascii() and item.strip().isdigit():
            item = int(item.strip())
        if isinstance(item, int) and item >= 1 and item not in numbers:
            numbers.append(item)
    return tuple(numbers)


def _json_object(text: str, what: str) -> dict[str, Any]:
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ParseError(f"{what} holds no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ParseError(f"{what} is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ParseError(f"{what} is not a JSON object")
    return data


def clean_message(text: str) -> str:
    """Strip whitespace and one wrapping pair of quotes; keep one line per messenger bubble.

    Quotes are removed only when the whole message starts and ends with a
    matching pair, so ``he said 'no'`` and ``'cause im tired`` stay intact.
    """
    lines = [" ".join(line.split()) for line in text.strip().splitlines()]
    message = "\n".join(line for line in lines if line)
    for opening, closing in _QUOTE_PAIRS:
        if len(message) >= 2 and message.startswith(opening) and message.endswith(closing):
            return message[1:-1].strip()
    return message


def parse_turn(raw: str) -> tuple[HiddenBlock, str]:
    """Split one turn output into the hidden block and the visible message.

    Args:
        raw: The model's raw output.

    Returns:
        The hidden block and the cleaned visible message (possibly empty).

    Raises:
        ParseError: A part is missing or the hidden block is unusable.
    """
    state_match = _STATE.search(raw)
    if state_match is None:
        raise ParseError("missing <state> block")
    message_match = _MESSAGE.search(raw, state_match.end())
    if message_match is None:
        raise ParseError("missing <message> after the <state> block")
    block = _json_object(state_match.group(1), "state")
    thought = block.get("inner_thought")
    if not isinstance(thought, str) or not thought.strip():
        raise ParseError("inner_thought is missing or empty")
    intent_raw = block.get("intent")
    try:
        intent = Intent(str(intent_raw).strip().lower())
    except ValueError as exc:
        raise ParseError(f"unknown intent {intent_raw!r}") from exc
    hidden = HiddenBlock(
        inner_thought=" ".join(thought.split()),
        emotion=_score(block.get("emotion"), "emotion"),
        energy=_score(block.get("energy"), "energy"),
        intent=intent,
        rules_triggered=_rule_numbers(block.get("rules_triggered")),
    )
    return hidden, clean_message(message_match.group(1))


def parse_day_notes(raw: str) -> tuple[tuple[str, ...], Stance | None]:
    """Parse end-of-day notes: ``{"memory": [...], "stance": "..."}``.

    Args:
        raw: The model's raw output (surrounding prose and fences are tolerated).

    Returns:
        Up to five cleaned note lines, and the stance (None when missing or
        not one of the schema's stances).

    Raises:
        ParseError: No JSON object, or no usable memory lines.
    """
    data = _json_object(raw, "notes")
    memory = data.get("memory")
    if not isinstance(memory, list):
        raise ParseError("notes.memory is missing")
    lines = tuple(
        " ".join(item.split()) for item in memory if isinstance(item, str) and item.strip()
    )[:MAX_NOTE_LINES]
    if not lines:
        raise ParseError("notes.memory is empty")
    stance_raw = data.get("stance")
    try:
        stance = Stance(stance_raw.strip().lower()) if isinstance(stance_raw, str) else None
    except ValueError:
        stance = None
    return lines, stance
