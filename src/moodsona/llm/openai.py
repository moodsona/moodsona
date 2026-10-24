"""Thin adapter for OpenAI-compatible Chat Completions (``pip install moodsona[openai]``).

Works with OpenAI and with any server that speaks the same API (pass
``base_url``). The ``openai`` SDK is imported only when a provider is created
without a ready client.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any

from moodsona.llm.base import ChatMessage, LLMConfigurationError, LLMError, LLMResponse

MODEL_ENV = "MOODSONA_OPENAI_MODEL"
# A current small model at the time of writing; update as needed.
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_TOKENS = 4000


class OpenAIProvider:
    """``LLMProvider`` backed by ``openai.OpenAI().chat.completions.create``.

    Args:
        model: Model id; defaults to ``$MOODSONA_OPENAI_MODEL`` or ``gpt-5-mini``.
        client: A ready SDK client (or a test double). When omitted, one is
            created with the SDK's usual credential lookup.
        send_temperature: Send ``temperature`` with each request. Off by
            default because reasoning models reject it; turn it on for models
            and servers that accept it.
        **client_options: Passed to ``openai.OpenAI(...)`` (e.g. ``base_url``).

    Raises:
        LLMConfigurationError: The SDK is not installed.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any = None,
        send_temperature: bool = False,
        **client_options: Any,
    ):
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.send_temperature = send_temperature
        if client is None:
            try:
                sdk = importlib.import_module("openai")
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The openai SDK is not installed: pip install 'moodsona[openai]'"
                ) from exc
            client = sdk.OpenAI(**client_options)
        self.client = client

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send the messages as a chat completion.

        Raises:
            LLMError: The SDK raised or returned no choices.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }
        if self.send_temperature and temperature is not None:
            request["temperature"] = temperature
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:  # the SDK's error hierarchy is not importable here
            raise LLMError(f"OpenAI call failed: {exc}") from exc
        if not response.choices:
            raise LLMError("OpenAI returned no choices")
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=choice.message.content or "",
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
            finish_reason=getattr(choice, "finish_reason", None),
        )
