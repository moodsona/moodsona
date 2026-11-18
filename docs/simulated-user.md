# SimulatedUser: a persona in any eval harness

A companion has `reply(history) -> str`; `SimulatedUser` has the same method.
Whatever alternates the two (openevals, langwatch/scenario, deepeval or a loop
of your own) runs a moodsona persona against your chatbot. moodsona records the
hidden state: each call returns only the persona's visible message, while
emotion, energy, stance and day notes stay inside and end up in the transcript.
`moodsona.run()` is exactly this loop with a `Companion`.

```python
from moodsona import SimulatedUser, load_bundled_persona
from moodsona.llm.anthropic import AnthropicProvider

user = SimulatedUser(
    load_bundled_persona("anxious-newcomer"),
    AnthropicProvider(),  # the model that plays the persona
    turns_per_day=4,
    days=3,
)
```

## What it reads

`history` is the conversation so far, seen from the harness: `user` messages
are this simulated user's earlier messages, `assistant` messages are your
bot's. Items can be moodsona `ChatMessage`s or OpenAI-style mappings
(`{"role": ..., "content": ...}`, with `content` a string or a list of
`{"type": "text", "text": ...}` parts). System, developer and tool messages,
and assistant messages without text (bare tool calls), are not shown to the
persona. The bot's reply to a persona message is all assistant text after it,
up to the next user message.

Use one `SimulatedUser` per conversation. A history shorter than the one the
previous call saw raises `ValueError`.

## Days

The first `reply` starts day 1; a bot message before it is part of that day.
A day ends when the persona ends it (`end_day`, or `skip_tomorrow`, which also
skips the next day) or after `turns_per_day` messages. The next `reply` then
writes that day's notes and starts the next day: a new conversation for the
persona, with its notes from earlier days and the overnight mood change.

With `days=3`, `user.done` turns True once the last turn of day 3 is written
(or earlier, when the remaining days are skipped). After that, `reply` raises
`RuntimeError`. With `days=None` (the default) there is no limit, and the
harness decides when to stop.

## A hand-written loop

```python
from moodsona import ChatMessage

history = []
while not user.done:
    history.append(ChatMessage("user", user.reply(history), day=user.day))
    history.append(ChatMessage("assistant", my_bot(history), day=user.day))  # your chatbot
transcript = user.finish(history)
transcript.to_jsonl("anxious-newcomer.jsonl")
```

`user.day` after a `reply` is the simulated day that message belongs to, from
1. Tagging both messages with it gives your bot the same day signal that
`run()` gives. The persona ignores the tags, so a history with or without them
(or OpenAI-style mappings with an extra `"day"` key) reads the same.

`finish(history)` fills in the bot's last reply, writes the open day's notes
and returns the same `Transcript` as `run()`. Read the JSONL file with
`moodsona report anxious-newcomer.jsonl`. Calling `finish` again returns the
same transcript without new model calls. `user.transcript(history)` and
`user.metrics(history)` give a snapshot at any point and never call the model.
If the model fails (`LLMError`), nothing is lost: repeat the same call.

## Day-aware companions

A bot with its own memory needs to know where one simulated day ends and the
next begins, or "does it remember yesterday" cannot be tested fairly. Every
message `run()` passes to the companion, and every message of
`Transcript.visible_history()`, has a `day` (from 1). A new day starts when
the `day` of the newest user message differs from the previous message's:

```python
class MyBot:
    def __init__(self):
        self.day = None

    def reply(self, history):
        newest = history[-1]
        if newest.day != self.day:  # the first message of a new simulated day
            self.day = newest.day
            self.start_new_session()  # your bot: a fresh session, yesterday in memory
        return self.answer(newest.content)
```

Days the persona skips never appear, so the next `day` can be more than one
higher. Messages you build yourself have `day=None` unless you pass one.
Model providers send only `role` and `content` to the API.

## openevals

This example was checked against the source of openevals 0.2.5
(`python/openevals/simulators/multiturn.py`, the same as the main branch in
November 2026). It is not run in CI, because openevals is not a moodsona
dependency.

```python
from openevals.simulators import run_multiturn_simulation


def simulated_user(trajectory, **kwargs):  # openevals adds thread_id and turn_counter
    return {"role": "user", "content": user.reply(trajectory)}


def app(message, **kwargs):  # message: the newest user message only
    return {"role": "assistant", "content": my_bot.answer(message["content"])}


result = run_multiturn_simulation(
    app=app,
    user=simulated_user,
    max_turns=12,  # 3 days x 4 turns; a safety net, `done` normally ends the run
    stopping_condition=lambda trajectory, **kwargs: user.done,
)
transcript = user.finish(result["trajectory"])
```

Notes on openevals 0.2.5:

- The `user` callable receives the whole trajectory as a list of OpenAI-style
  messages, which `reply` reads as it is. It must return a message mapping,
  not a bare string.
- The `app` callable receives only the newest user message, so your app keeps
  the conversation itself (per `thread_id`, which openevals passes along).
- `stopping_condition` runs after each app reply, so the trajectory you pass to
  `finish` already contains the bot's answer to the last persona message.
- `trajectory_evaluators` judge the bot. `user.metrics(result["trajectory"])`
  describes the simulated user (polluted and forced turns, repeats, mood
  drift), which helps to tell "the simulator drifted" from "the bot failed".

## License

This page, like the rest of the documentation, is licensed under
[CC BY 4.0](../LICENSE-DATA.md).
