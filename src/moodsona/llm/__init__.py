"""Model provider layer: the protocol, a deterministic fake and two thin adapters.

The Anthropic and OpenAI adapters import their SDKs lazily, so ``import moodsona``
works without either SDK installed (``pip install moodsona[anthropic]`` adds one).
"""

from moodsona.llm.base import (
    ChatMessage,
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMResponse,
    Role,
)
from moodsona.llm.fake import FakeProvider, plumbing_reply

__all__ = [
    "ChatMessage",
    "FakeProvider",
    "LLMConfigurationError",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Role",
    "plumbing_reply",
]
