"""Thin adapter for Anthropic's Messages API (``pip install moodsona[anthropic]``).

The ``anthropic`` SDK is imported only when a provider is created without a
ready client, so the core package never needs it.

Notes:
    * ``temperature`` is accepted for protocol compatibility but not sent: the
      Messages API rejects sampling parameters on current models.
    * Text blocks of the reply are joined; other blocks (e.g. thinking) are
      ignored. A refusal comes back as an empty ``content`` with
      ``finish_reason="refusal"``, which the simulator treats as a failed
      attempt and retries.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any

from moodsona.llm.base import ChatMessage, LLMConfigurationError, LLMError, LLMResponse

MODEL_ENV = "MOODSONA_ANTHROPIC_MODEL"
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 4000


class AnthropicProvider:
    """``LLMProvider`` backed by ``anthropic.Anthropic().messages.create``.

    Args:
        model: Model id; defaults to ``$MOODSONA_ANTHROPIC_MODEL`` or
            ``claude-sonnet-5-5``.
        client: A ready SDK client (or a test double). When omitted, one is
            created with the SDK's usual credential lookup.
        **client_options: Passed to ``anthropic.Anthropic(...)`` (e.g. ``api_key``).

    Raises:
        LLMConfigurationError: The SDK is not installed.
    """

    def __init__(self, model: str | None = None, *, client: Any = None, **client_options: Any):
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        if client is None:
            try:
                sdk = importlib.import_module("anthropic")
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The anthropic SDK is not installed: pip install 'moodsona[anthropic]'"
                ) from exc
            client = sdk.Anthropic(**client_options)
        self.client = client

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send the messages; system messages become the ``system`` parameter.

        Raises:
            LLMError: The SDK raised.
        """
        del temperature  # not accepted by the Messages API on current models
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "messages": chat,
        }
        if system:
            request["system"] = system
        try:
            response = self.client.messages.create(**request)
        except Exception as exc:  # the SDK's error hierarchy is not importable here
            raise LLMError(f"Anthropic call failed: {exc}") from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=text,
            tokens_in=getattr(usage, "input_tokens", None),
            tokens_out=getattr(usage, "output_tokens", None),
            finish_reason=getattr(response, "stop_reason", None),
        )
