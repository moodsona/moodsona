"""HttpCompanion and OpenAICompatibleCompanion against a local HTTP server (no network)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from moodsona.companion import (
    Companion,
    CompanionError,
    HttpCompanion,
    HttpCompanionError,
    OpenAICompatibleCompanion,
)
from moodsona.llm.base import ChatMessage
from moodsona.llm.fake import FakeProvider
from moodsona.runner import run
from support import make_seed

HISTORY = (
    ChatMessage("user", "the café was closed ☕"),
    ChatMessage("assistant", "Oh no. Which one?"),
    ChatMessage("user", "the one by the river"),
)
HIDDEN = "hidden-header-value"


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: Any


@dataclass
class LocalServer:
    """What the test server answers, and what it received."""

    url: str = ""
    status: int = 200
    body: bytes = b'{"reply": "I am here."}'
    extra_headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    received: list[Received] = field(default_factory=list)

    def answer_json(self, data: Any) -> None:
        self.body = json.dumps(data).encode("utf-8")


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalServer]:
    # Keep any configured HTTP proxy out of the way of 127.0.0.1.
    monkeypatch.setenv("no_proxy", "*")
    state = LocalServer()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            headers = {k.lower(): v for k, v in self.headers.items()}
            state.received.append(Received(self.path, headers, json.loads(raw.decode("utf-8"))))
            time.sleep(state.delay)
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(state.body)))
            for name, value in state.extra_headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(state.body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    class Server(ThreadingHTTPServer):
        def handle_error(self, request: Any, client_address: Any) -> None:
            pass  # the timeout test closes the connection before the handler writes

    httpd = Server(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


# ------------------------------------------------------------ HttpCompanion --


def test_default_shapes(server: LocalServer) -> None:
    companion = HttpCompanion(server.url + "/chat")
    assert isinstance(companion, Companion)
    assert companion.reply(HISTORY) == "I am here."

    (got,) = server.received
    assert got.path == "/chat"
    assert got.body == {
        "messages": [
            {"role": "user", "content": "the café was closed ☕"},
            {"role": "assistant", "content": "Oh no. Which one?"},
            {"role": "user", "content": "the one by the river"},
        ]
    }
    assert got.headers["content-type"] == "application/json; charset=utf-8"
    assert got.headers["accept"] == "application/json"
    assert got.headers["user-agent"].startswith("moodsona/")


def test_custom_headers_are_sent(server: LocalServer) -> None:
    headers = {"Authorization": f"Bearer {HIDDEN}", "X-Bot-Variant": "b"}
    HttpCompanion(server.url, headers=headers).reply(HISTORY)
    (got,) = server.received
    assert got.headers["authorization"] == f"Bearer {HIDDEN}"
    assert got.headers["x-bot-variant"] == "b"


def test_custom_request_and_response(server: LocalServer) -> None:
    server.answer_json({"data": {"answer": "Rivers are nice."}})

    def request(history: Sequence[ChatMessage]) -> Any:
        return {"session": "test-1", "text": history[-1].content, "turns": len(history)}

    def response(body: Any) -> str:
        return str(body["data"]["answer"])

    companion = HttpCompanion(server.url, request=request, response=response)
    assert companion.reply(HISTORY) == "Rivers are nice."
    assert server.received[0].body == {
        "session": "test-1",
        "text": "the one by the river",
        "turns": 3,
    }


def test_empty_reply_is_allowed(server: LocalServer) -> None:
    server.answer_json({"reply": ""})
    assert HttpCompanion(server.url).reply(HISTORY) == ""


def test_http_error_names_url_and_status_but_not_headers(server: LocalServer) -> None:
    server.status = 503
    server.answer_json({"error": "overloaded"})
    companion = HttpCompanion(server.url + "/chat?token=query-secret", headers={"X-Key": HIDDEN})
    with pytest.raises(HttpCompanionError) as info:
        companion.reply(HISTORY)
    error = info.value
    assert isinstance(error, CompanionError)
    assert error.status == 503
    assert error.url == server.url + "/chat"
    assert error.body == '{"error": "overloaded"}'
    message = str(error)
    assert server.url + "/chat" in message
    assert "503" in message
    assert HIDDEN not in message
    assert "query-secret" not in message


def test_invalid_header_value_is_not_quoted(server: LocalServer) -> None:
    companion = HttpCompanion(server.url, headers={"Authorization": f"{HIDDEN}\nX-Injected: 1"})
    with pytest.raises(HttpCompanionError, match="invalid request headers") as info:
        companion.reply(HISTORY)
    assert HIDDEN not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__
    assert server.received == []


def test_redirects_are_not_followed(server: LocalServer) -> None:
    server.status = 302
    server.extra_headers = {"Location": server.url + "/elsewhere"}
    with pytest.raises(HttpCompanionError, match="302") as info:
        HttpCompanion(server.url, headers={"Authorization": HIDDEN}).reply(HISTORY)
    assert "redirect" in str(info.value)
    assert [r.path for r in server.received] == ["/"]


def test_non_json_body(server: LocalServer) -> None:
    server.body = b"<html>hello</html>"
    with pytest.raises(HttpCompanionError, match="not JSON"):
        HttpCompanion(server.url).reply(HISTORY)


@pytest.mark.parametrize(
    "answer",
    [{"text": "wrong field"}, {"reply": 42}, {"reply": None}, ["reply"]],
)
def test_unreadable_reply(server: LocalServer, answer: Any) -> None:
    server.answer_json(answer)
    with pytest.raises(HttpCompanionError, match="reply"):
        HttpCompanion(server.url).reply(HISTORY)


def test_custom_reader_returning_non_string(server: LocalServer) -> None:
    def length(body: Any) -> Any:
        return len(body)

    with pytest.raises(HttpCompanionError, match="int"):
        HttpCompanion(server.url, response=length).reply(HISTORY)


def test_timeout(server: LocalServer) -> None:
    server.delay = 1.0
    started = time.monotonic()
    with pytest.raises(HttpCompanionError, match="timed out") as info:
        HttpCompanion(server.url, timeout=0.1).reply(HISTORY)
    assert time.monotonic() - started < 0.9
    assert info.value.status is None


def test_connection_refused(server: LocalServer) -> None:
    # Find a port nothing listens on: bind one, then close it.
    probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = probe.server_address[1]
    probe.server_close()
    companion = HttpCompanion(f"http://127.0.0.1:{port}/chat")
    with pytest.raises(HttpCompanionError, match=f"127.0.0.1:{port}/chat"):
        companion.reply(HISTORY)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://127.0.0.1/bot",
        "localhost:8000/chat",
        "http://",
        "http://user:password@127.0.0.1/chat",
    ],
)
def test_bad_url_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="URL") as info:
        HttpCompanion(url)
    assert "password" not in str(info.value)


def test_bad_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        HttpCompanion("http://127.0.0.1/chat", timeout=0)


def test_end_to_end_run(server: LocalServer) -> None:
    server.answer_json({"reply": "Tell me more."})
    companion = HttpCompanion(server.url)
    transcript = run(make_seed(), FakeProvider(), companion, days=1, turns_per_day=2)
    assert transcript.turns
    assert {t.companion_reply for t in transcript.turns} == {"Tell me more."}
    assert len(server.received) == len(transcript.turns)
    last = server.received[-1].body["messages"]
    assert last == [
        {"role": m.role, "content": m.content} for m in transcript.visible_history()[:-1]
    ]


# ---------------------------------------------------- OpenAICompatibleCompanion --


def completion(content: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}


def test_openai_compatible_request(server: LocalServer) -> None:
    server.answer_json(completion("Hello from the model."))
    companion = OpenAICompatibleCompanion(server.url + "/v1/", "tiny-model", "You are kind.")
    assert isinstance(companion, Companion)
    assert companion.reply(HISTORY) == "Hello from the model."

    (got,) = server.received
    assert got.path == "/v1/chat/completions"
    assert got.body == {
        "model": "tiny-model",
        "messages": [{"role": "system", "content": "You are kind."}]
        + [{"role": m.role, "content": m.content} for m in HISTORY],
        "stream": False,
    }
    assert "authorization" not in got.headers


def test_openai_compatible_options(server: LocalServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOODSONA_TEST_BOT_KEY", HIDDEN)
    server.answer_json(completion("ok"))
    companion = OpenAICompatibleCompanion(
        server.url + "/v1",
        "tiny-model",
        "",
        api_key_env="MOODSONA_TEST_BOT_KEY",
        headers={"X-Bot-Variant": "a"},
        temperature=0.2,
        max_tokens=50,
    )
    companion.reply(HISTORY)
    (got,) = server.received
    assert got.path == "/v1/chat/completions"
    assert got.headers["authorization"] == f"Bearer {HIDDEN}"
    assert got.headers["x-bot-variant"] == "a"
    assert got.body["temperature"] == 0.2
    assert got.body["max_tokens"] == 50
    # An empty system prompt sends no system message.
    assert got.body["messages"][0] == {"role": "user", "content": HISTORY[0].content}


@pytest.mark.parametrize("value", [None, "", "   "])
def test_openai_compatible_missing_key(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("MOODSONA_TEST_BOT_KEY", raising=False)
    else:
        monkeypatch.setenv("MOODSONA_TEST_BOT_KEY", value)
    with pytest.raises(ValueError, match="MOODSONA_TEST_BOT_KEY"):
        OpenAICompatibleCompanion(
            "http://127.0.0.1/v1", "m", "s", api_key_env="MOODSONA_TEST_BOT_KEY"
        )


def test_openai_compatible_error_hides_key(
    server: LocalServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOODSONA_TEST_BOT_KEY", HIDDEN)
    server.status = 401
    companion = OpenAICompatibleCompanion(
        server.url + "/v1", "m", "s", api_key_env="MOODSONA_TEST_BOT_KEY"
    )
    with pytest.raises(HttpCompanionError, match="401") as info:
        companion.reply(HISTORY)
    assert server.url + "/v1/chat/completions" in str(info.value)
    assert HIDDEN not in str(info.value)


@pytest.mark.parametrize("answer", [completion(None), {"choices": []}, {"error": "x"}])
def test_openai_compatible_unreadable_reply(server: LocalServer, answer: Any) -> None:
    server.answer_json(answer)
    with pytest.raises(HttpCompanionError, match="reply"):
        OpenAICompatibleCompanion(server.url + "/v1", "m", "s").reply(HISTORY)


def test_exported_from_package() -> None:
    import moodsona

    for name in ("CompanionError", "HttpCompanion", "HttpCompanionError"):
        assert name in moodsona.__all__
    assert moodsona.OpenAICompatibleCompanion is OpenAICompatibleCompanion


def test_openai_compatible_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError, match="URL"):
        OpenAICompatibleCompanion("file:///tmp", "m", "s")
