"""The companion under test, and two test doubles.

Wrap your chatbot in a class with one method, ``reply(history) -> str``. The
history holds every visible message so far, across all simulated days:
``user`` messages come from the simulated person, ``assistant`` messages are
your companion's earlier replies. Nothing hidden (inner thoughts, state, notes)
is ever in it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from moodsona.llm.base import ChatMessage


@runtime_checkable
class Companion(Protocol):
    """The chatbot being tested."""

    def reply(self, history: Sequence[ChatMessage]) -> str:
        """Return the companion's next message.

        Args:
            history: All visible messages so far; the last one is the
                simulated user's newest message.

        Returns:
            The companion's reply text.
        """
        ...


class EchoCompanion:
    """Repeats the user's last message, with an optional prefix. For smoke tests.

    Args:
        prefix: Text put before the echoed message.
    """

    def __init__(self, prefix: str = "You said: ") -> None:
        self.prefix = prefix

    def reply(self, history: Sequence[ChatMessage]) -> str:
        """Echo the newest user message (or an empty echo when there is none)."""
        last = next((m.content for m in reversed(history) if m.role == "user"), "")
        return f"{self.prefix}{last}"


class ScriptExhaustedError(RuntimeError):
    """A :class:`ScriptedCompanion` ran out of replies."""


class ScriptedCompanion:
    """Returns fixed replies in order and records every history it was given.

    Args:
        replies: The replies to return.
        cycle: Start again from the first reply when the script runs out,
            instead of raising :class:`ScriptExhaustedError`.

    Attributes:
        histories: The history passed on each call, in order.
    """

    def __init__(self, replies: Iterable[str], *, cycle: bool = False) -> None:
        self.replies = tuple(replies)
        if not self.replies:
            raise ValueError("ScriptedCompanion needs at least one reply")
        self.cycle = cycle
        self.histories: list[tuple[ChatMessage, ...]] = []

    def reply(self, history: Sequence[ChatMessage]) -> str:
        """Return the next scripted reply.

        Raises:
            ScriptExhaustedError: No replies left and ``cycle`` is off.
        """
        index = len(self.histories)
        self.histories.append(tuple(history))
        if index >= len(self.replies) and not self.cycle:
            raise ScriptExhaustedError(f"no scripted reply left (call {index + 1})")
        return self.replies[index % len(self.replies)]
