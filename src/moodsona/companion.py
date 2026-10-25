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

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from moodsona.llm.base import ChatMessage
from moodsona.version import __version__

if TYPE_CHECKING:
    # The HTTP modules (urllib.request pulls in http.client, ssl and email) are
    # imported where they are used, so `import moodsona` stays cheap.
    import urllib.request
    from urllib.parse import SplitResult


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

    Args:
        history: The messages to send, in order.

    Returns:
        ``{"messages": [{"role": "user", "content": "..."}, ...]}`` with one
        entry per message of the history (roles ``user`` and ``assistant``).
    """
    return {"messages": [{"role": m.role, "content": m.content} for m in history]}


def reply_field(body: Any) -> str:
    """The default response reader of :class:`HttpCompanion`.

    Args:
        body: The parsed JSON response body.

    Returns:
        ``body["reply"]``. :class:`HttpCompanion` checks that it is a string.

    Raises:
        KeyError: The body has no ``"reply"`` field.
        TypeError: The body is not a JSON object.
    """
    reply: str = body["reply"]
    return reply


def _is_loopback(host: str) -> bool:
    """True for ``localhost`` and loopback addresses (127.0.0.0/8, ::1)."""
    import ipaddress

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _opener(host: str) -> urllib.request.OpenerDirector:
    """An opener that speaks only HTTP(S) and follows no redirects.

    Built from explicit handlers, so there is no file:, ftp: or data: handler,
    and no redirect handler (urllib would turn the POST into a GET and resend
    the headers): a 3xx becomes an HTTPError. Proxies from the environment are
    honoured, except for loopback hosts, so local traffic and its headers never
    go to a proxy.
    """
    import urllib.request

    opener = urllib.request.OpenerDirector()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
        urllib.request.UnknownHandler(),  # any other scheme: URLError("unknown url type")
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ]
    if not _is_loopback(host):
        handlers.append(urllib.request.ProxyHandler())
    for handler in handlers:
        opener.add_handler(handler)
    return opener


def _check_url(url: str) -> SplitResult:
    """Validate an http(s) URL and return its parts."""
    import urllib.parse

    # Checked before parsing: http.client would otherwise reject these later with
    # a message that quotes the path and query string.
    if not url.isascii():
        raise ValueError("the URL contains non-ASCII characters; percent-encode them")
    if any(char <= " " or char == "\x7f" for char in url):
        raise ValueError("the URL contains spaces or control characters; percent-encode them")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError(f"the URL must start with http:// or https:// (got {parts.scheme!r})")
    if not parts.hostname:
        raise ValueError("the URL has no host")
    if parts.username is not None or parts.password is not None:
        raise ValueError("credentials in the URL are not supported; send them with headers=")
    try:
        parts.port  # noqa: B018 - raises ValueError for a port that is not a number
    except ValueError:
        raise ValueError("the URL has an invalid port") from None
    return parts


class HttpCompanion:
    """Your bot behind an HTTP endpoint: posts the history as JSON, reads the reply.

    Each :meth:`reply` sends one ``POST`` with a UTF-8 JSON body and
    ``Content-Type: application/json``, and parses the response as JSON. By
    default the body is ``{"messages": [{"role": ..., "content": ...}, ...]}``
    (see :func:`messages_request`) and the reply is the response's ``"reply"``
    field (see :func:`reply_field`). Pass ``request`` and ``response`` to fit
    any other shape.

    The reply must be a string; an empty string is accepted and recorded as
    is. Only HTTP(S) is spoken and redirects are not followed (a redirect is
    an error). Proxies from the usual environment variables are honoured,
    except for ``localhost`` and loopback addresses, which are always reached
    directly.

    Args:
        url: The endpoint; only ``http://`` and ``https://`` URLs are accepted.
        headers: Extra request headers, for example ``Authorization``. They
            override the defaults and never appear in error messages.
        timeout: Seconds to wait for the connection and for each read.
        request: Builds the JSON body from the history.
        response: Returns the reply text from the parsed JSON body.

    Raises:
        ValueError: The URL is not http(s), has no host, an invalid port,
            credentials, non-ASCII characters, spaces or control characters,
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
        parts = self._endpoint(_check_url(url))
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._url = parts.geturl()
        self._safe_url = parts._replace(query="", fragment="").geturl()  # for messages
        self._timeout = timeout
        self.request = request
        self.response = response
        self._headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": f"moodsona/{__version__}",
            **(headers or {}),
        }
        self._opener = _opener(parts.hostname or "")

    def _endpoint(self, parts: SplitResult) -> SplitResult:
        """The URL to post to, from the validated ``url``; subclasses may adjust it."""
        return parts

    @property
    def url(self) -> str:
        """The endpoint (read-only: it was validated once)."""
        return self._url

    @property
    def timeout(self) -> float:
        """Seconds to wait for the connection and for each read (read-only)."""
        return self._timeout

    def reply(self, history: Sequence[ChatMessage]) -> str:
        """Post the history and return the bot's reply.

        Raises:
            HttpCompanionError: ``request`` failed or built a body that is not
                JSON (NaN and infinity are refused), the connection failed or
                timed out, the status was not 2xx, the body was not JSON, or
                ``response`` could not read a string reply from it.
        """
        try:
            payload = json.dumps(self.request(history), ensure_ascii=False, allow_nan=False)
        except Exception as exc:  # the builder is the caller's code; any failure is theirs
            raise self._error(f"could not build the request ({type(exc).__name__})") from exc
        import urllib.request

        # S310: the scheme was checked in __init__ (http/https only), and the
        # opener has no file:, ftp: or data: handler anyway.
        req = urllib.request.Request(  # noqa: S310
            self._url, payload.encode("utf-8"), self._headers, method="POST"
        )
        raw = self._send(req)
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:  # JSONDecodeError and UnicodeDecodeError
            raise self._error("the response is not JSON", body=raw) from None
        try:
            reply = self.response(body)
        except Exception as exc:  # the reader is the caller's code, like the builder
            # Only the type: the exception text may quote the response body.
            problem = f"could not read the reply from the response ({type(exc).__name__})"
            raise self._error(problem) from exc
        if not isinstance(reply, str):
            raise self._error(f"the reply must be a string, got {type(reply).__name__}")
        return reply

    def _send(self, req: urllib.request.Request) -> bytes:
        import http.client
        import urllib.error

        try:
            with self._opener.open(req, timeout=self._timeout) as answer:
                raw: bytes = answer.read()
                return raw
        except urllib.error.HTTPError as exc:
            hint = " (redirects are not followed)" if 300 <= exc.code < 400 else ""
            problem = f"HTTP {exc.code} {exc.reason}{hint}"
            error_body: bytes | None
            try:
                with exc:
                    error_body = exc.read()
            except (OSError, http.client.HTTPException):  # the body is only a diagnostic
                error_body = None
            raise self._error(problem, status=exc.code, body=error_body) from None
        except (OSError, http.client.HTTPException) as exc:  # URLError is an OSError
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, TimeoutError):
                raise self._error(f"timed out after {self._timeout:g} s") from exc
            raise self._error(f"request failed: {reason}") from exc
        except ValueError:  # includes UnicodeEncodeError
            # http.client rejects a bad header name or value with a message that
            # quotes it; drop that message so a key never reaches a log.
            problem = "invalid request: a header or the URL has characters HTTP does not allow"
            raise self._error(problem) from None

    def _error(
        self, problem: str, *, status: int | None = None, body: bytes | None = None
    ) -> HttpCompanionError:
        text = None if body is None else body.decode("utf-8", "replace")[:_BODY_LIMIT]
        return HttpCompanionError(
            f"{self._safe_url}: {problem}", url=self._safe_url, status=status, body=text
        )


class OpenAICompatibleCompanion(HttpCompanion):
    """A model-backed bot behind any chat-completions endpoint, with no extra dependency.

    Posts ``{"model": ..., "messages": [system] + history, "stream": false}``
    to ``<base_url>/chat/completions`` and returns
    ``choices[0].message.content`` (a ``null`` content, as from a refusal or
    a reasoning model that ran out of tokens, becomes ``""``). Written for
    local servers such as Ollama (``http://localhost:11434/v1``), LM Studio
    (``http://localhost:1234/v1``) and vLLM (``http://localhost:8000/v1``);
    hosted servers that follow the same API work too, but their accepted
    parameters differ, so sampling options go in ``extra_body``.

    Args:
        base_url: The API root, ending in ``/v1`` for most servers; a
            trailing slash is fine, and a query string is kept.
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
        extra_body: Extra fields merged into every request body, for example
            ``{"temperature": 0.7, "max_tokens": 300}`` (or
            ``max_completion_tokens`` for servers that want it). They
            override the fields above, ``stream`` included.

    Raises:
        ValueError: The URL is invalid (see :class:`HttpCompanion`), or
            ``api_key_env`` names a variable that is unset or empty. The
            message never repeats ``api_key_env``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        *,
        api_key_env: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 120,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.extra_body = dict(extra_body or {})
        all_headers = dict(headers or {})
        if api_key_env is not None:
            key = os.environ.get(api_key_env, "").strip()
            if not key:
                # Never echo the value: passed by mistake, it may be the key itself.
                raise ValueError(
                    "the environment variable named by api_key_env is not set"
                    " (pass the variable's name, not the key)"
                )
            all_headers["Authorization"] = f"Bearer {key}"
        super().__init__(
            base_url,
            headers=all_headers,
            timeout=timeout,
            request=self._chat_request,
            response=_first_choice,
        )

    def _endpoint(self, parts: SplitResult) -> SplitResult:
        """``<base_url>/chat/completions``, keeping any query string."""
        return parts._replace(path=parts.path.rstrip("/") + "/chat/completions")

    def _chat_request(self, history: Sequence[ChatMessage]) -> dict[str, Any]:
        system = [ChatMessage("system", self.system_prompt)] if self.system_prompt else []
        return {
            "model": self.model,
            **messages_request([*system, *history]),
            "stream": False,
            **self.extra_body,
        }


def _first_choice(body: Any) -> str:
    content: str | None = body["choices"][0]["message"]["content"]
    return "" if content is None else content
