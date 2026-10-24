"""Chat types and the provider protocol every model adapter implements.

The protocol is synchronous on purpose: one simulated conversation is a strict
sequence of calls, and a sync API is easier to read and test. Run several
simulations in threads or processes if you need parallelism.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat message.

    In a conversation history, ``user`` is the simulated person and
    ``assistant`` is the companion under test.

    Attributes:
        role: ``"system"``, ``"user"`` or ``"assistant"``.
        content: The message text.
    """

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One completion returned by a provider.

    Attributes:
        content: The assistant text (may be empty, for example on a refusal).
        tokens_in: Prompt tokens as reported by the provider, if any.
        tokens_out: Completion tokens as reported by the provider, if any.
        finish_reason: The provider's own stop label, verbatim (``"end_turn"``,
            ``"length"``, ``"refusal"`` ...), if any.
    """

    content: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None


class LLMError(Exception):
    """Base class for provider failures (network, HTTP, malformed response)."""


class LLMConfigurationError(LLMError):
    """The provider cannot be used as configured (missing SDK, key or model)."""


class LLMProvider(Protocol):
    """Anything that turns chat messages into one assistant message."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return one assistant message for the given chat messages.

        Args:
            messages: The prompt, usually one system and one user message.
            temperature: Sampling temperature; adapters may ignore it when the
                model does not accept one.
            max_tokens: Output budget for the reply.

        Returns:
            The provider's reply.

        Raises:
            LLMError: The call failed.
        """
        ...
