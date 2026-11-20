# moodsona in langwatch/scenario

**Status: experimental.** The adapter and its API may change in any release.

[langwatch/scenario](https://github.com/langwatch/scenario) runs simulated
conversations against your agent and lets a judge decide whether it behaved.
`moodsona.adapters.scenario` lets a moodsona persona play the user in such a
run, in place of scenario's own user simulator. The agent is a
[`SimulatedUser`](../simulated-user.md) that scenario can call: each call
returns `user.reply(input.messages)`. The persona keeps its hidden state
(emotion, energy, stance, notes about earlier days) inside the agent across the
whole run; scenario, your agent and the judge only see the visible messages.

## Install

```bash
pip install 'moodsona[scenario]'
```

This installs `langwatch-scenario` next to moodsona. The core package does not
need it: `scenario` is imported only when you build the agent, and without it
you get an `ImportError` that names the extra.

## Example

A pytest test (with [pytest-asyncio](https://pypi.org/project/pytest-asyncio/),
as in scenario's own docs). `MyCompanion` stands in for your chatbot; here it
always gives the same answer.

```python
import pytest
import scenario

from moodsona import FakeProvider, load_bundled_persona
from moodsona.adapters.scenario import user_simulator_agent

# scenario's judge is a real model call (a litellm model id; needs that
# provider's API key). Without a model, scenario.JudgeAgent() raises.
scenario.configure(default_model="openai/gpt-4.1-mini")


class MyCompanion(scenario.AgentAdapter):
    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        return "That sounds like a lot. What happened next?"  # your chatbot here


@pytest.mark.asyncio
async def test_companion_with_an_anxious_newcomer() -> None:
    user = user_simulator_agent(
        load_bundled_persona("anxious-newcomer"),
        FakeProvider(),  # synthetic persona lines; swap for a real model, see below
        turns_per_day=4,
    )
    result = await scenario.run(
        name="first week with an anxious newcomer",
        description="A new user talks to the companion over several days.",
        agents=[
            MyCompanion(),
            user,
            scenario.JudgeAgent(criteria=["The companion does not ask several questions at once"]),
        ],
        max_turns=12,
    )
    assert result.success

    print(user.metrics(result.messages))
    transcript = user.finish(result.messages)
    transcript.to_jsonl("anxious-newcomer.jsonl")  # then: moodsona report anxious-newcomer.jsonl
```

To check the wiring without any model (no judge, no API key), leave out
`scenario.configure` and the `JudgeAgent` and give the run a script that ends
it:

```python
    steps = []
    for _ in range(8):
        steps += [scenario.user(), scenario.agent()]
    result = await scenario.run(
        name="wiring check",
        description="A new user talks to the companion over several days.",
        agents=[MyCompanion(), user],
        script=[*steps, scenario.succeed()],
    )
```

- `user.finish(result.messages)` returns a `moodsona.Transcript`, the same
  shape `moodsona.run` returns, so `moodsona report` and `moodsona.metrics`
  work on it. It writes the last day's notes (one more model call). Pass
  `result.messages` so the agent's last reply is recorded; without it that
  reply is empty. `user.transcript(result.messages)` is the same without the
  last day's notes and without a model call.
- `user.metrics(result.messages)` computes the metrics directly. It uses the
  persona's language profile; `moodsona report` uses English. For every
  bundled persona the two are the same.
- `user.state` is the persona's hidden state at any point, `user.day` the
  current moodsona day.
- Use one agent per `scenario.run`: it carries the persona's state from call to
  call. Its first successful call binds it to scenario's `thread_id`; a call
  from another thread raises a `ValueError`.
- Cancelling a call from outside (a test timeout, for example) does not stop a
  model call that has already started; the worker thread finishes it.
- The transcript holds the persona's turns and the replies to them. Scripted
  user lines and an opening agent message are not in it, so its
  `visible_history()` can be shorter than `result.messages`.

## Costs and what a fake provider tests

With `FakeProvider()` the persona's lines are synthetic, so the run only checks
the wiring. To test your companion, give the persona a real user model, for
example `AnthropicProvider()` from `moodsona.llm.anthropic` (needs
`moodsona[anthropic]` and an API key) or `OpenAIProvider()` from
`moodsona.llm.openai`. Every persona turn is then at least one paid model call,
retries and end-of-day notes are more, and scenario's judge makes calls of its
own.

## What maps to what

| scenario | moodsona |
|---|---|
| a call to the user agent | one persona turn (`SimulatedUser.reply`) |
| `turns_per_day` user turns, or fewer if the persona ends the day | one moodsona day; the next call writes the day's notes and starts the next day |
| assistant messages after the persona's message | the companion's reply |
| system, tool and developer messages, tool calls without text | not shown to the persona |
| `max_turns`, the judge or the script | when the run stops; `user.finish(result.messages)` writes the last day's notes |
| a scripted user line (`scenario.user("...")`), an opening agent message | shown to the persona as part of the day, not recorded in the transcript |

A day the persona decides to skip (`skip_tomorrow`) has no messages: the
conversation continues with the day after, and the transcript lists the skipped
day. The persona's mood moves overnight between days as in `moodsona.run`.
[SimulatedUser](../simulated-user.md) describes the days and what the persona
reads in detail.

The adapter reads only `AgentInput.messages` (OpenAI-style messages; content as
a string or as a list of text parts) and `AgentInput.thread_id`, and returns the
persona's message as a string, which scenario adds as a user message. The
persona's model calls run in a worker thread, so scenario's event loop is not
blocked.

## License

This page, like the rest of the documentation, is licensed under
[CC BY 4.0](../../LICENSE-DATA.md).
