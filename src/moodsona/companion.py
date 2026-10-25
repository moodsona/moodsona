"""The companion under test, two test doubles and two HTTP adapters.

Wrap your chatbot in a class with one method, ``reply(history) -> str``. The
history holds every visible message so far, across all simulated days:
``user`` messages come from the simulated person, ``assistant`` messages are
your companion's earlier replies. Nothing hidden (inner thoughts, state, notes)
is ever in it.

If your bot already runs behind HTTP, :class:`HttpCompanion` posts the history
as JSON to your endpoint, and :class:`OpenAICompatibleCompanion` talks to any
chat-completions server (Ollama, vLLM, LM Studio ...). Both use only the
standard library.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from moodsona.llm.base import ChatMessage
from moodsona.version import __version__


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


class CompanionError(Exception):
    """Base class for failures of a companion adapter shipped with moodsona."""


class HttpCompanionError(CompanionError):
    """An HTTP companion call failed (connection, timeout, status or response shape).

    The message names the URL (without query string) and the HTTP status. It
    never contains request headers, so API keys sent in headers stay out of
    logs and tracebacks.

    Attributes:
        url: The endpoint, without query string or fragment.
        status: The HTTP status code, or None when no response arrived.
        body: Up to 2000 characters of the response body, or None. Kept out of
            the message because servers sometimes echo request data in errors.
    """

    def __init__(
        self, message: str, *, url: str, status: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.body = body


RequestBuilder = Callable[[Sequence[ChatMessage]], Any]
"""Turns the history into a JSON-serialisable request body."""

ResponseReader = Callable[[Any], str]
"""Takes the parsed JSON response body and returns the reply text."""

_BODY_LIMIT = 2000


def messages_request(history: Sequence[ChatMessage]) -> dict[str, Any]:
    """The default request body of :class:`HttpCompanion`.

    Returns:
        ``{"messages": [{"role": "user", "content": "..."}, ...]}`` with every
        message of the history in order (roles ``user`` and ``assistant``).
    """
    return {"messages": [{"role": m.role, "content": m.content} for m in history]}


def reply_field(body: Any) -> str:
    """The default response reader of :class:`HttpCompanion`: ``body["reply"]``."""
    reply: str = body["reply"]
    return reply


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: urllib would turn the POST into a GET and resend the headers."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _check_url(url: str) -> str:
    """Validate an http(s) URL and return it without query string or fragment."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError(f"the URL must start with http:// or https:// (got {parts.scheme!r})")
    if not parts.hostname:
        raise ValueError("the URL has no host")
    if parts.username is not None or parts.password is not None:
        raise ValueError("credentials in the URL are not supported; send them with headers=")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class HttpCompanion:
    """Your bot behind an HTTP endpoint: posts the history as JSON, reads the reply.

    Each :meth:`reply` sends one ``POST`` with a UTF-8 JSON body and
    ``Content-Type: application/json``, and parses the response as JSON. By
    default the body is ``{"messages": [{"role": ..., "content": ...}, ...]}``
    (see :func:`messages_request`) and the reply is the response's ``"reply"``
    field (see :func:`reply_field`). Pass ``request`` and ``response`` to fit
    any other shape.

    The reply must be a string; an empty string is accepted and recorded as
    is. Redirects are not followed (a redirect is an error), and proxies from
    the usual environment variables are honoured.

    Args:
        url: The endpoint; only ``http://`` and ``https://`` URLs are accepted.
        headers: Extra request headers, for example ``Authorization``. They
            override the defaults and never appear in error messages.
        timeout: Seconds to wait for the connection and for each read.
        request: Builds the JSON body from the history.
        response: Returns the reply text from the parsed JSON body.

    Raises:
        ValueError: The URL is not http(s), has no host or holds credentials,
            or ``timeout`` is not positive.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30,
        request: RequestBuilder = messages_request,
        response: ResponseReader = reply_field,
    ) -> None:
        self._safe_url = _check_url(url)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url = url
        self.timeout = timeout
        self.request = request
        self.response = response
        self._headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": f"moodsona/{__version__}",
            **(headers or {}),
        }
        self._opener = urllib.request.build_opener(_NoRedirect)

    def reply(self, history: Sequence[ChatMessage]) -> str:
        """Post the history and return the bot's reply.

        Raises:
            HttpCompanionError: The connection failed or timed out, the status
                was not 2xx, the body was not JSON, or ``response`` could not
                read a string reply from it.
        """
        data = json.dumps(self.request(history), ensure_ascii=False).encode("utf-8")
        # S310: the scheme was checked in __init__ (http/https only), so urllib's
        # file:// and ftp:// handlers cannot be reached.
        req = urllib.request.Request(self.url, data, self._headers, method="POST")  # noqa: S310
        raw = self._send(req)
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:  # JSONDecodeError and UnicodeDecodeError
            raise self._error("the response is not JSON", body=raw) from None
        try:
            reply = self.response(body)
        except (LookupError, TypeError, ValueError, AttributeError) as exc:
            raise self._error(f"could not read the reply from the response ({exc!r})") from exc
        if not isinstance(reply, str):
            raise self._error(f"the reply must be a string, got {type(reply).__name__}")
        return reply

    def _send(self, req: urllib.request.Request) -> bytes:
        try:
            with self._opener.open(req, timeout=self.timeout) as answer:
                raw: bytes = answer.read()
                return raw
        except urllib.error.HTTPError as exc:
            hint = " (redirects are not followed)" if 300 <= exc.code < 400 else ""
            with exc:
                raw = exc.read()
            problem = f"HTTP {exc.code} {exc.reason}{hint}"
            raise self._error(problem, status=exc.code, body=raw) from None
        except (OSError, http.client.HTTPException) as exc:  # URLError is an OSError
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, TimeoutError):
                raise self._error(f"timed out after {self.timeout:g} s") from exc
            raise self._error(f"request failed: {reason}") from exc
        except ValueError:
            # http.client rejects a bad header name or value with a message that
            # quotes it; drop that message so a key never reaches a log.
            raise self._error("invalid request headers") from None

    def _error(
        self, problem: str, *, status: int | None = None, body: bytes | None = None
    ) -> HttpCompanionError:
        text = None if body is None else body[:_BODY_LIMIT].decode("utf-8", "replace")
        return HttpCompanionError(
            f"{self._safe_url}: {problem}", url=self._safe_url, status=status, body=text
        )


class OpenAICompatibleCompanion(HttpCompanion):
    """A model-backed bot behind any chat-completions endpoint, with no extra dependency.

    Posts ``{"model": ..., "messages": [system] + history, "stream": false}``
    to ``<base_url>/chat/completions`` and returns
    ``choices[0].message.content``. Works with local servers such as Ollama
    (``http://localhost:11434/v1``), LM Studio (``http://localhost:1234/v1``)
    and vLLM (``http://localhost:8000/v1``), and with hosted ones.

    Args:
        base_url: The API root, ending in ``/v1`` for most servers; a
            trailing slash is fine.
        model: The model name the server knows.
        system_prompt: Your bot's system prompt; an empty string sends none.
        api_key_env: Name of an environment variable that holds the API key.
            When set, the key is read once, here, and sent as
            ``Authorization: Bearer <key>``. None (the default) sends no key,
            which is what local servers expect, so a key is never sent to a
            URL you did not mean it for.
        headers: Extra request headers.
        timeout: Seconds to wait; local models can be slow, hence the
            longer default.
        temperature: Sent as ``temperature`` when given.
        max_tokens: Sent as ``max_tokens`` when given.

    Raises:
        ValueError: The URL is not http(s), or ``api_key_env`` names a
            variable that is unset or empty.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        api_key_env: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 120,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        all_headers = dict(headers or {})
        if api_key_env is not None:
            key = os.environ.get(api_key_env, "").strip()
            if not key:
                raise ValueError(f"api_key_env is {api_key_env!r}, but that variable is not set")
            all_headers["Authorization"] = f"Bearer {key}"
        super().__init__(
            base_url.rstrip("/") + "/chat/completions",
            headers=all_headers,
            timeout=timeout,
            request=self._chat_request,
            response=_first_choice,
        )

    def _chat_request(self, history: Sequence[ChatMessage]) -> dict[str, Any]:
        system = [ChatMessage("system", self.system_prompt)] if self.system_prompt else []
        body: dict[str, Any] = {"model": self.model, **messages_request([*system, *history])}
        body["stream"] = False
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        return body


def _first_choice(body: Any) -> str:
    content: str = body["choices"][0]["message"]["content"]
    return content
