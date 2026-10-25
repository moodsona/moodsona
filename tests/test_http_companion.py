"""HttpCompanion and OpenAICompatibleCompanion against a local HTTP server (no network)."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from moodsona.companion import (
    Companion,
    CompanionError,
    HttpCompanion,
    HttpCompanionError,
    OpenAICompatibleCompanion,
    _is_loopback,
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
    # Announce more body bytes than are sent, so the client's read fails.
    short_body: bool = False
    received: list[Received] = field(default_factory=list)

    def answer_json(self, data: Any) -> None:
        self.body = json.dumps(data).encode("utf-8")


@pytest.fixture
def server() -> Iterator[LocalServer]:
    # No no_proxy here: loopback targets must bypass any configured proxy by themselves.
    state = LocalServer()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            headers = {k.lower(): v for k, v in self.headers.items()}
            state.received.append(Received(self.path, headers, json.loads(raw.decode("utf-8"))))
            time.sleep(state.delay)
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            length = len(state.body) + (100 if state.short_body else 0)
            self.send_header("Content-Length", str(length))
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


@pytest.mark.parametrize("value", [f"{HIDDEN}\nX-Injected: 1", f"{HIDDEN} \u2615"])
def test_bad_header_value_is_not_quoted(server: LocalServer, value: str) -> None:
    # A newline and a character outside Latin-1: http.client quotes both in its error.
    companion = HttpCompanion(server.url, headers={"Authorization": value})
    with pytest.raises(HttpCompanionError, match="HTTP does not allow") as info:
        companion.reply(HISTORY)
    assert HIDDEN not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__
    assert server.received == []


@pytest.mark.parametrize("status", [200, 500])
def test_truncated_body(server: LocalServer, status: int) -> None:
    server.status = status
    server.short_body = True
    with pytest.raises(HttpCompanionError) as info:
        HttpCompanion(server.url).reply(HISTORY)
    assert info.value.status == (None if status == 200 else 500)


BAD_URLS = [
    ("file:///etc/passwd", "http:// or https://"),
    ("ftp://127.0.0.1/bot", "http:// or https://"),
    ("localhost:8000/chat", "http:// or https://"),
    ("http://", "no host"),
    ("http://user:password@127.0.0.1/chat", "credentials"),
    ("http://127.0.0.1:http/chat", "invalid port"),
    ("http://127.0.0.1/chat?token=a b", "spaces or control characters"),
    ("http://127.0.0.1/a\rb", "spaces or control characters"),
    ("http://127.0.0.1/caf\u00e9", "non-ASCII"),
    ("http://127.0.0.1/a\u00a0b", "non-ASCII"),
]


def openai_compatible(url: str) -> OpenAICompatibleCompanion:
    return OpenAICompatibleCompanion(url, "m", "s")


@pytest.mark.parametrize(("url", "problem"), BAD_URLS)
@pytest.mark.parametrize("make", [HttpCompanion, openai_compatible], ids=["http", "openai"])
def test_bad_url_is_rejected(url: str, problem: str, make: Callable[[str], object]) -> None:
    with pytest.raises(ValueError, match=problem) as info:
        make(url)
    assert "password" not in str(info.value)
    assert "token" not in str(info.value)


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


def dead_port() -> int:
    """A port nothing listens on: bind one, then close it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def test_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    companion = HttpCompanion("http://127.0.0.1:9/chat")

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(companion._opener, "open", refuse)
    with pytest.raises(HttpCompanionError, match=r"127\.0\.0\.1:9/chat: request failed") as info:
        companion.reply(HISTORY)
    assert info.value.status is None
    assert isinstance(info.value.__cause__, urllib.error.URLError)


def test_url_and_timeout_are_read_only() -> None:
    companion = HttpCompanion("http://127.0.0.1/chat", timeout=5)
    assert companion.url == "http://127.0.0.1/chat"
    assert companion.timeout == 5
    with pytest.raises(AttributeError):
        companion.url = "file:///etc/passwd"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        companion.timeout = 0  # type: ignore[misc]


def test_opener_cannot_read_files(tmp_path: Path) -> None:
    # The URL is validated once; this swaps the private copy behind the property's
    # back to show the opener itself has no file: handler.
    page = tmp_path / "reply.json"
    page.write_text('{"reply": "read from disk"}', encoding="utf-8")
    companion = HttpCompanion("http://127.0.0.1/chat")
    companion._url = page.as_uri()
    with pytest.raises(HttpCompanionError, match="unknown url type"):
        companion.reply(HISTORY)


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("127.0.0.1", True),
        ("127.8.9.10", True),
        ("::1", True),
        ("bot.example.test", False),
        ("10.0.0.1", False),
        ("128.0.0.1", False),
        ("::2", False),
        ("localhost.x", False),
    ],
)
def test_is_loopback(host: str, loopback: bool) -> None:
    assert _is_loopback(host) is loopback


def clear_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)


def test_loopback_target_bypasses_env_proxy(
    server: LocalServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_no_proxy(monkeypatch)
    dead = f"http://127.0.0.1:{dead_port()}"
    monkeypatch.setenv("http_proxy", dead)
    monkeypatch.setenv("https_proxy", dead)
    assert HttpCompanion(server.url).reply(HISTORY) == "I am here."
    assert [r.path for r in server.received] == ["/"]


def test_env_proxy_is_used_for_other_hosts(
    server: LocalServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_no_proxy(monkeypatch)
    monkeypatch.setenv("http_proxy", server.url)  # the local server plays the proxy
    assert HttpCompanion("http://bot.example.test/chat").reply(HISTORY) == "I am here."
    assert [r.path for r in server.received] == ["http://bot.example.test/chat"]


@pytest.mark.parametrize(
    "build",
    [
        lambda history: {"x": object()},
        lambda history: {"x": float("nan")},
        lambda history: 1 / 0,
    ],
)
def test_request_build_failure(server: LocalServer, build: Callable[..., Any]) -> None:
    with pytest.raises(HttpCompanionError, match="could not build the request") as info:
        HttpCompanion(server.url, request=build).reply(HISTORY)
    assert info.value.__cause__ is not None
    assert server.received == []


def test_reader_failure_does_not_echo_response_data(server: LocalServer) -> None:
    server.answer_json({"text": "ECHOED-REQUEST-DATA"})

    def picky(body: Any) -> str:
        raise ValueError(f"unexpected body {body}")

    with pytest.raises(HttpCompanionError, match="ValueError") as info:
        HttpCompanion(server.url, response=picky).reply(HISTORY)
    assert "ECHOED" not in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_error_body_is_cut_by_characters(server: LocalServer) -> None:
    server.status = 500
    server.body = ("é" * 3000).encode("utf-8")
    with pytest.raises(HttpCompanionError) as info:
        HttpCompanion(server.url).reply(HISTORY)
    assert info.value.body == "é" * 2000


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
        extra_body={"temperature": 0.2, "max_completion_tokens": 50},
    )
    companion.reply(HISTORY)
    (got,) = server.received
    assert got.path == "/v1/chat/completions"
    assert got.headers["authorization"] == f"Bearer {HIDDEN}"
    assert got.headers["x-bot-variant"] == "a"
    assert got.body["temperature"] == 0.2
    assert got.body["max_completion_tokens"] == 50
    assert "max_tokens" not in got.body
    # An empty system prompt sends no system message.
    assert got.body["messages"][0] == {"role": "user", "content": HISTORY[0].content}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MOODSONA_TEST_BOT_KEY", None),
        ("MOODSONA_TEST_BOT_KEY", ""),
        ("MOODSONA_TEST_BOT_KEY", "  "),
        (HIDDEN, None),  # the key itself passed by mistake
    ],
)
def test_openai_compatible_missing_key(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="not the key") as info:
        OpenAICompatibleCompanion("http://127.0.0.1/v1", "m", "s", api_key_env=name)
    assert name not in str(info.value)


def test_openai_compatible_keeps_query_string(server: LocalServer) -> None:
    server.answer_json(completion("ok"))
    OpenAICompatibleCompanion(server.url + "/v1/?api-version=2", "m", "s").reply(HISTORY)
    assert server.received[0].path == "/v1/chat/completions?api-version=2"


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


def test_openai_compatible_null_content_is_empty(server: LocalServer) -> None:
    server.answer_json(completion(None))
    assert OpenAICompatibleCompanion(server.url + "/v1", "m", "s").reply(HISTORY) == ""


@pytest.mark.parametrize("answer", [completion(42), {"choices": []}, {"error": "x"}])
def test_openai_compatible_unreadable_reply(server: LocalServer, answer: Any) -> None:
    server.answer_json(answer)
    with pytest.raises(HttpCompanionError, match="reply"):
        OpenAICompatibleCompanion(server.url + "/v1", "m", "s").reply(HISTORY)
