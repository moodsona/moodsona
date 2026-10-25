# Plugging in your bot

moodsona needs two things for a run:

- the **companion**: your bot, the thing under test. It gets the visible
  conversation and returns its next message.
- the **provider**: the model that plays the simulated user. It is separate
  from your bot. `FakeProvider` writes placeholder lines, costs nothing and
  needs no network, so use it to check the wiring first. A real provider
  (`AnthropicProvider`, `OpenAIProvider`) makes paid API calls on every turn.

This page is about the companion. There are three ways to plug yours in.

## 1. A Python class

Anything with a `reply(history) -> str` method is a companion. `history` is a
sequence of `moodsona.ChatMessage` (`role` is `"user"` for the simulated person
and `"assistant"` for your bot's earlier replies, `content` is the text); the
last message is the newest user message.

```python
import moodsona


class MyBot:
    def reply(self, history):
        last = history[-1].content
        return f"That sounds like a lot. What happened after '{last[:40]}'?"


persona = moodsona.load_bundled_persona("anxious-newcomer")
transcript = moodsona.run(persona, moodsona.FakeProvider(), MyBot(), days=2, turns_per_day=3)
```

Use this when your bot is a Python function or SDK call you can import.

## 2. Your HTTP endpoint: `HttpCompanion`

If your bot runs as a web service, `HttpCompanion` posts the history to it on
every turn and reads the reply. It uses only the Python standard library.

By default it sends

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "rainy day"}]}
```

as a `POST` with `Content-Type: application/json; charset=utf-8`, and expects
a JSON response with a `"reply"` string:

```json
{"reply": "Rainy days can feel long. How is yours going?"}
```

```python
from moodsona import FakeProvider, HttpCompanion, load_bundled_persona, run

bot = HttpCompanion(
    "http://localhost:8080/chat",
    headers={"Authorization": "Bearer test-token"},  # optional
    timeout=30,  # seconds
)
transcript = run(load_bundled_persona("anxious-newcomer"), FakeProvider(), bot, 2, 3)
```

If your endpoint uses other field names, pass two small functions: `request`
builds the JSON body from the history, `response` takes the parsed JSON body
and returns the reply text.

```python
def request(history):
    # This endpoint keeps its own memory and wants only the newest message.
    return {"session_id": "moodsona-test-1", "text": history[-1].content}


def response(body):
    return body["data"]["answer"]


bot = HttpCompanion("https://bot.example.test/v2/message", request=request, response=response)
```

The defaults are `moodsona.companion.messages_request` and
`moodsona.companion.reply_field`, if you want to build on them.

Behaviour worth knowing:

- Only `http://` and `https://` URLs are accepted, in ASCII (percent-encode
  anything else). Put credentials in `headers`, not in the URL.
- One request per turn, in order, with no retries and no streaming.
- Redirects are not followed; a redirect is reported as an error.
- Proxies set in `HTTP_PROXY` / `HTTPS_PROXY` are used, except for
  `localhost` and loopback addresses (127.x.x.x, `::1`), which are always
  reached directly.
- The reply must be a string. An empty string is accepted and recorded as is.
- Any failure (building the request, connection, timeout, a non-2xx status,
  a body that is not JSON, a reply that cannot be read) raises
  `moodsona.HttpCompanionError` (a `moodsona.CompanionError`). The message
  names the URL without its query string and the HTTP status; it never
  contains your headers or the response body. The error's `status` and
  `body` attributes hold the status code and the first 2000 characters of
  the response body.
- A failed call stops `run()` with that error, and the turns so far are not
  returned. Retry inside your own `reply` if your endpoint is flaky.

## 3. A model behind a chat-completions server: `OpenAICompatibleCompanion`

To test a bot that is a model plus a system prompt, point
`OpenAICompatibleCompanion` at any server that speaks the chat-completions
API. It posts `model`, the system prompt followed by the history, and
`"stream": false` to `<base_url>/chat/completions`, and returns
`choices[0].message.content`. A `null` content (a refusal, or a reasoning
model that used up its token budget) is recorded as an empty reply. No extra
package is needed.

```python
from moodsona import FakeProvider, OpenAICompatibleCompanion, load_bundled_persona, run

bot = OpenAICompatibleCompanion(
    "http://localhost:11434/v1",  # Ollama
    "your-model",  # a model the server has, e.g. from `ollama list`
    "You are a warm, patient companion. Keep replies short.",
    extra_body={"temperature": 0.7, "max_tokens": 300},  # optional
)
transcript = run(load_bundled_persona("anxious-newcomer"), FakeProvider(), bot, 2, 3)
```

`extra_body` is merged into every request as is, so you choose the parameter
names your server accepts. Local servers take `max_tokens`; some hosted
reasoning models reject it and want `max_completion_tokens` instead, and may
reject `temperature`.

Usual base URLs of local servers (check your server's own docs if you changed
its port):

| Server    | Base URL                    |
|-----------|-----------------------------|
| Ollama    | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1`  |
| vLLM      | `http://localhost:8000/v1`  |

Local servers need no API key, so by default none is sent. For a server that
needs one, name the environment variable that holds it:

```python
bot = OpenAICompatibleCompanion(
    "https://llm.example.test/v1",
    "your-model",
    "You are a warm, patient companion.",
    api_key_env="MY_BOT_API_KEY",
)
```

The key is read when the companion is created (so a missing variable fails
before any turn runs) and sent as `Authorization: Bearer <key>`. There is no
default variable on purpose: a key is only sent where you asked for it. A
hosted model bills you for every companion reply, on top of what a real
provider costs for the simulated user.

## Next

Once the wiring works with `FakeProvider`, swap in a real provider (see the
Quickstart in the [README](../../README.md#quickstart)) and read the results
with `moodsona report`.
