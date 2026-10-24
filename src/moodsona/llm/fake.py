"""A deterministic, offline provider for tests and smoke runs.

``FakeProvider`` either replays a script of replies or calls a responder
function, and records every call. Without arguments it uses
:func:`plumbing_reply`, which produces valid simulator output from the prompt
itself. That output exercises the plumbing; it is never evidence about how a
real model behaves.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from moodsona.llm.base import ChatMessage, LLMError, LLMResponse
from moodsona.prompts import NOTES_MARKER, TURN_MARKER

Responder = Callable[[Sequence[ChatMessage]], str]

_LINES = (
    "the bus was late again",
    "made pasta, it was fine",
    "work was long today",
    "my neighbour's dog keeps barking",
    "finally fixed the bike chain",
    "it rained all afternoon",
    "tried a new coffee place",
)


@dataclass(frozen=True, slots=True)
class FakeCall:
    """One recorded call to a :class:`FakeProvider`."""

    messages: tuple[ChatMessage, ...]
    temperature: float | None
    max_tokens: int | None


class FakeProvider:
    """Provider that never touches the network.

    Args:
        replies: A list of replies returned in order, or a function from the
            messages to a reply. Defaults to :func:`plumbing_reply`.

    Attributes:
        calls: Every call made, in order.
    """

    def __init__(self, replies: Iterable[str] | Responder | None = None) -> None:
        self.calls: list[FakeCall] = []
        self._responder: Responder | None = None
        self._script: list[str] = []
        if replies is None:
            self._responder = plumbing_reply
        elif callable(replies):
            self._responder = replies
        else:
            self._script = list(replies)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return the next scripted reply (or the responder's reply).

        Raises:
            LLMError: The script has no replies left.
        """
        self.calls.append(FakeCall(tuple(messages), temperature, max_tokens))
        if self._responder is not None:
            return LLMResponse(self._responder(messages))
        if not self._script:
            raise LLMError("FakeProvider script is exhausted")
        return LLMResponse(self._script.pop(0))


def _payload(messages: Sequence[ChatMessage]) -> dict[str, object]:
    try:
        data = json.loads(messages[1].content)
    except (IndexError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _int(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def plumbing_reply(messages: Sequence[ChatMessage]) -> str:
    """Valid, deterministic simulator output derived from the prompt payload.

    Turn calls get a hidden block (emotion drifts by one point, energy falls by
    two) and a short synthetic English line; notes calls get two note lines.
    Any other call gets ``"ok"``.
    """
    system = messages[0].content if messages else ""
    payload = _payload(messages)
    day = _int(payload.get("day"), 1)
    if system.startswith(NOTES_MARKER):
        notes = {
            "memory": [f"day {day}: talked about my day", "it asked a lot of questions"],
            "stance": "curious",
        }
        return json.dumps(notes)
    if not system.startswith(TURN_MARKER):
        return "ok"
    turn = _int(payload.get("turn_in_day"), 1)
    raw_state = payload.get("state")
    state = raw_state if isinstance(raw_state, dict) else {}
    block = {
        "inner_thought": f"Day {day}, turn {turn}: I will just answer.",
        "emotion": _int(state.get("emotion"), 50) + (1 if turn % 2 else -1),
        "energy": _int(state.get("energy"), 50) - 2,
        "intent": "reply" if turn % 2 else "short_reply",
        "rules_triggered": [],
    }
    line = _LINES[(day * 3 + turn) % len(_LINES)]
    return f"<state>{json.dumps(block)}</state>\n<message>{line}</message>"
