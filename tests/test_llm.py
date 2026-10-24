from __future__ import annotations

import importlib
import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from moodsona.llm import anthropic as anthropic_adapter
from moodsona.llm import openai as openai_adapter
from moodsona.llm.anthropic import AnthropicProvider
from moodsona.llm.base import ChatMessage, LLMConfigurationError, LLMError, LLMResponse
from moodsona.llm.fake import FakeProvider, plumbing_reply
from moodsona.llm.openai import OpenAIProvider
from moodsona.parse import parse_day_notes, parse_turn
from moodsona.prompts import NOTES_MARKER, TURN_MARKER

PROMPT = [ChatMessage("system", "be brief"), ChatMessage("user", "hi")]


def test_fake_provider_replays_a_script_and_records_calls() -> None:
    provider = FakeProvider(["one", "two"])
    assert provider.complete(PROMPT, temperature=0.5, max_tokens=9).content == "one"
    assert provider.complete(PROMPT).content == "two"
    assert provider.calls[0].temperature == 0.5
    assert provider.calls[0].max_tokens == 9
    assert provider.calls[1].messages == tuple(PROMPT)
    with pytest.raises(LLMError, match="exhausted"):
        provider.complete(PROMPT)


def test_fake_provider_with_a_responder() -> None:
    provider = FakeProvider(lambda messages: messages[-1].content.upper())
    assert provider.complete(PROMPT) == LLMResponse("HI")


def test_plumbing_reply_is_valid_simulator_output() -> None:
    payload = {"day": 2, "turn_in_day": 2, "state": {"emotion": 0, "energy": 1}}
    turn = [ChatMessage("system", TURN_MARKER + "\n..."), ChatMessage("user", json.dumps(payload))]
    hidden, message = parse_turn(plumbing_reply(turn))
    assert (hidden.emotion, hidden.energy, hidden.intent.value) == (0, 0, "short_reply")
    assert message
    notes = [ChatMessage("system", NOTES_MARKER), ChatMessage("user", json.dumps({"day": 3}))]
    assert parse_day_notes(plumbing_reply(notes))[0][0].startswith("day 3")
    assert plumbing_reply([]) == "ok"
    assert plumbing_reply([ChatMessage("system", TURN_MARKER), ChatMessage("user", "[]")])
    assert plumbing_reply([ChatMessage("system", TURN_MARKER), ChatMessage("user", "{")])


# -- Anthropic adapter, with a stand-in client --------------------------------------


class _AnthropicClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = self
        self._response = response
        self._error = error

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response


def test_anthropic_adapter_maps_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(anthropic_adapter.MODEL_ENV, raising=False)
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="text", text="there"),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        stop_reason="end_turn",
    )
    client = _AnthropicClient(response)
    provider = AnthropicProvider(client=client)
    result = provider.complete(PROMPT, temperature=0.9, max_tokens=50)
    assert result == LLMResponse("hello there", 12, 3, "end_turn")
    assert client.requests[0] == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 50,
        "system": "be brief",
        "messages": [{"role": "user", "content": "hi"}],
    }
    provider.complete([ChatMessage("user", "x")])
    assert "system" not in client.requests[1]
    assert client.requests[1]["max_tokens"] == anthropic_adapter.DEFAULT_MAX_TOKENS


def test_anthropic_adapter_model_from_env_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(anthropic_adapter.MODEL_ENV, "claude-haiku-5-5")
    provider = AnthropicProvider(client=_AnthropicClient(error=RuntimeError("boom")))
    assert provider.model == "claude-haiku-5-5"
    with pytest.raises(LLMError, match="boom"):
        provider.complete(PROMPT)


def test_adapters_explain_a_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(LLMConfigurationError, match=r"moodsona\[anthropic\]"):
        AnthropicProvider()
    with pytest.raises(LLMConfigurationError, match=r"moodsona\[openai\]"):
        OpenAIProvider()


def test_adapters_build_the_sdk_client_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}
    fake_sdk = SimpleNamespace(
        Anthropic=lambda **kw: built.setdefault("anthropic", kw),
        OpenAI=lambda **kw: built.setdefault("openai", kw),
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_sdk)
    AnthropicProvider(api_key="k1")
    OpenAIProvider(base_url="http://localhost:1234/v1")
    assert built == {
        "anthropic": {"api_key": "k1"},
        "openai": {"base_url": "http://localhost:1234/v1"},
    }


# -- OpenAI adapter, with a stand-in client ------------------------------------------


class _OpenAIClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)
        self._response = response
        self._error = error

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response


def _openai_response(content: str | None) -> Any:
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=2)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_openai_adapter_maps_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(openai_adapter.MODEL_ENV, raising=False)
    client = _OpenAIClient(_openai_response("hey"))
    provider = OpenAIProvider(client=client)
    assert provider.model == openai_adapter.DEFAULT_MODEL
    assert provider.complete(PROMPT, temperature=0.3, max_tokens=20) == LLMResponse(
        "hey", 7, 2, "stop"
    )
    assert client.requests[0] == {
        "model": openai_adapter.DEFAULT_MODEL,
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        "max_completion_tokens": 20,
    }
    warm = OpenAIProvider("local-model", client=client, send_temperature=True)
    warm.complete(PROMPT, temperature=0.3)
    assert client.requests[1]["temperature"] == 0.3
    assert client.requests[1]["max_completion_tokens"] == openai_adapter.DEFAULT_MAX_TOKENS


def test_openai_adapter_failures() -> None:
    provider = OpenAIProvider(client=_OpenAIClient(_openai_response(None)))
    assert provider.complete(PROMPT).content == ""
    empty = OpenAIProvider(client=_OpenAIClient(SimpleNamespace(choices=[], usage=None)))
    with pytest.raises(LLMError, match="no choices"):
        empty.complete(PROMPT)
    broken = OpenAIProvider(client=_OpenAIClient(error=RuntimeError("down")))
    with pytest.raises(LLMError, match="down"):
        broken.complete(PROMPT)


# -- live (real API, never in CI) ------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_live_anthropic() -> None:
    pytest.importorskip("anthropic")
    reply = AnthropicProvider().complete(
        [ChatMessage("user", "Reply with the single word: pong")], max_tokens=2000
    )
    assert "pong" in reply.content.lower()


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_openai() -> None:
    pytest.importorskip("openai")
    reply = OpenAIProvider().complete(
        [ChatMessage("user", "Reply with the single word: pong")], max_tokens=2000
    )
    assert "pong" in reply.content.lower()
