# moodsona in langwatch/scenario

**Status: experimental.** The adapter and its API may change in any release.

[langwatch/scenario](https://github.com/langwatch/scenario) runs simulated
conversations against your agent and lets a judge decide whether it behaved.
`moodsona.adapters.scenario` lets a moodsona persona play the user in such a
run, in place of scenario's own user simulator. The persona keeps its hidden
state (emotion, energy, stance, notes about earlier days) inside the adapter
across the whole run; scenario, your agent and the judge only see the visible
messages.

## Install

```bash
pip install 'moodsona[scenario]'
```

This installs `langwatch-scenario` next to moodsona. The core package does not
need it: `scenario` is imported only when you build the agent, and without it
you get an `ImportError` that names the extra.

## Example

```python
import pytest
import scenario

from moodsona import FakeProvider, load_bundled_persona
from moodsona.adapters.scenario import user_simulator_agent


class MyCompanion(scenario.AgentAdapter):
    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        return my_companion_reply(input.messages)  # your chatbot


@pytest.mark.asyncio
async def test_companion_with_an_anxious_newcomer() -> None:
    user = user_simulator_agent(
        load_bundled_persona("anxious-newcomer"),
        FakeProvider(),  # swap for a real provider, see below
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

    transcript = user.transcript(result.messages)
    transcript.to_jsonl("anxious-newcomer.jsonl")  # then: moodsona report anxious-newcomer.jsonl
    print(user.metrics(result.messages))
```

- `user.transcript(result.messages)` returns a `moodsona.Transcript`, the same
  shape `moodsona.run` returns, so `moodsona report` and `moodsona.metrics`
  work on it. Pass `result.messages` so the agent's last reply is recorded;
  without it that reply is empty.
- `user.metrics(result.messages)` computes the metrics directly.
- `user.simulator.state` is the persona's hidden state at any point.
- Use one agent per `scenario.run`: it carries the persona's state from call to
  call, so reusing it would continue the same persona's week.

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
| a call to the user agent | one persona turn (`Simulator.next_turn`) |
| `turns_per_day` user turns, or fewer if the persona ends the day | one moodsona day; the next call writes the day's notes and starts the next day |
| assistant messages after the persona's message | the companion's reply |
| system, tool and developer messages, tool calls without text | not shown to the persona |
| `max_turns`, the judge or the script | when the run stops; the last day's notes are not written |

A day the persona decides to skip (`skip_tomorrow`) has no messages: the
conversation continues with the day after, and the transcript lists the skipped
day. The persona's mood moves overnight between days as in `moodsona.run`.

The adapter reads only `AgentInput.messages` (OpenAI-style messages; content as
a string or as a list of text parts) and returns the persona's message as a
string, which scenario adds as a user message. The persona's model calls run in
a worker thread, so scenario's event loop is not blocked.
