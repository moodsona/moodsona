# moodsona in promptfoo

**Experimental.** The config keys and the output shape may still change
before 1.0.

[promptfoo](https://www.promptfoo.dev) is a Node.js tool that runs test cases
against "providers" and checks their output with assertions. moodsona ships a
promptfoo Python provider, `moodsona.adapters.promptfoo.call_api`: each
promptfoo test case is one multi-day moodsona run (a persona talks to your
companion for several simulated days), and the output is the transcript plus
the metrics that `moodsona report --json` prints. Your assertions then check
those numbers and messages.

This page was written against the promptfoo documentation for Python
providers as of October 2026 (promptfoo 0.125.1 was the latest release). The
adapter does not import promptfoo and was tested by calling `call_api`
directly, the way promptfoo calls it; it has not been run inside promptfoo by
the moodsona maintainers yet. Reports of what works and what does not are
welcome.

## Setup

promptfoo starts a Python process and imports a provider file from it, so that
Python needs moodsona (and your companion's code) installed. Make an
environment for it:

```bash
uv venv && uv pip install moodsona       # or: python -m venv .venv && .venv/bin/pip install moodsona
```

Then either run promptfoo with that environment active, or set
`pythonExecutable` in the provider config to its interpreter
(`.venv/bin/python`, on Windows `.venv/Scripts/python.exe`).

promptfoo loads providers from files, so copy the three-line provider file
[`examples/promptfoo/provider.py`](../../examples/promptfoo/provider.py) next
to your `promptfooconfig.yaml`. It only does
`from moodsona.adapters.promptfoo import call_api`.

A complete offline example (the fake user model and the echo companion, free,
no API key) is
[`examples/promptfoo/promptfooconfig.yaml`](../../examples/promptfoo/promptfooconfig.yaml).
From that folder:

```bash
npx promptfoo@latest eval
```

## The provider config

```yaml
providers:
  - id: file://provider.py
    config:
      persona: anxious-newcomer          # bundled id or a persona YAML path
      companion: my_bot.companion:MyBot  # the companion under test
      companion_args: {greeting: "Hi"}   # keyword arguments for it
      provider: fake                     # the user model
      days: 2
      turns_per_day: 3
```

| key | default | meaning |
| --- | --- | --- |
| `persona` | `anxious-newcomer` | A bundled persona id, or the path of a persona YAML file. A relative path is read from the folder of the promptfoo config. |
| `companion` | `moodsona:EchoCompanion` | Import path `package.module:Name` of the chatbot under test. `Name` is a class or a factory function. |
| `companion_args` | none | Keyword arguments `Name` is called with. |
| `provider` | `fake` | The user model that plays the persona: `fake` (offline), `anthropic` or `openai`. |
| `model` | the adapter's default | Model id for `anthropic` / `openai`. |
| `allow_paid` | `false` | Must be `true` for any provider but `fake` (see below). |
| `days` | `3` | Simulated days. |
| `turns_per_day` | `6` | Maximum persona messages per day. |
| `seed` | `0` | Random seed of the run. |

Whatever `companion` names is called once per test case with
`companion_args`, and must return an object with a `reply(history) -> str`
method (the `moodsona.Companion` protocol; `history` is a sequence of
`moodsona.ChatMessage` with `role` and `content`). Its module must be
importable by the Python that promptfoo runs: install your package in that
environment or put its folder on `PYTHONPATH`.

An unknown key is an error, so a typo does not silently run the default.
promptfoo's own keys (`pythonExecutable` and other camelCase keys, `workers`,
`timeout`) are left to promptfoo. A multi-day run with a real model can take
longer than promptfoo's default timeout of 5 minutes; raise `timeout`
(milliseconds) if it does.

### Test vars and the prompt

A test case's `vars` may override `persona`, `days`, `turns_per_day` and
`seed`, so one provider can run several personas:

```yaml
tests:
  - vars: {persona: anxious-newcomer}
  - vars: {persona: lonely-night-owl, days: 4}
```

Setting `provider`, `model`, `allow_paid`, `companion` or `companion_args` in
vars is an error: those belong in the provider config, so that a test case
cannot switch to a paid model. Other vars are ignored. promptfoo needs a
prompt, but moodsona does not use it (the persona writes the user's lines);
a prompt such as `"moodsona: {{persona}}"` just labels the results table.

## The output

`call_api` returns `{"output": {...}}` with a JSON object:

| field | content |
| --- | --- |
| `persona`, `companion`, `model` | What ran (`model` is the user-model label, e.g. `fake`). |
| `days`, `turns_per_day`, `seed` | The run settings. |
| `transcript` | Every visible message: `{day, turn, role, content}`, `role` is `user` (the persona) or `assistant` (your companion). |
| `trajectory` | The persona's mood after each turn: `{day, turn, emotion, energy, stance}`. |
| `skipped_days` | Days the persona chose not to write. |
| `metrics` | The `moodsona report --json` dict (see `moodsona.metrics` for definitions). |

Problems come back as promptfoo's `{"error": "..."}` instead of an exception:
a bad setting, an unknown persona, a companion that cannot be imported or
built, a missing SDK, or an exception raised during the run (your companion's
included, with its type and message).

## Assertions

In a JavaScript assertion written as an expression, `output` is the object
above, so fields are plain property access:

```yaml
defaultTest:
  assert:
    - type: javascript
      value: output.metrics.polluted_share === 0
    - type: javascript
      value: output.metrics.own_repeat_rate <= 0.2
tests:
  - vars: {persona: anxious-newcomer}
    assert:
      - type: javascript
        value: output.transcript.length === 12
      - type: javascript
        value: output.transcript.filter(m => m.role === 'assistant').every(m => m.content.length < 400)
```

Keep in mind what the numbers describe: most metrics are about the simulated
user (did it hold the persona?), not about your companion. Check your
companion with assertions on the `assistant` messages of the transcript, or
with promptfoo's model-graded assertions on them, and use the metrics to
notice runs where the simulation itself went wrong (`polluted_share`,
`forced_share`, `own_repeat_rate`). With the fake user model the persona's
lines are plumbing filler, never evidence about a real model; use it to test
your setup.

## Money

With `provider: anthropic` or `openai` every test case calls a paid API, and a
run of `d` days x `t` turns can make up to `d * (3t + 2)` model calls with
the default settings (each turn can be retried, plus the end-of-day notes).
The adapter refuses to start such a run unless the provider config sets
`allow_paid: true`, and the refusal names that maximum. Install the SDK in
promptfoo's Python (`moodsona[anthropic]` or `moodsona[openai]`) and set the
usual API key environment variable in the shell that starts promptfoo.

```yaml
providers:
  - id: file://provider.py
    config:
      provider: anthropic
      model: claude-sonnet-5-5
      allow_paid: true          # this costs money
      days: 2
      turns_per_day: 4
      timeout: 1200000
```

The limit is per test case: multiply by the number of test cases, and by
`--repeat` if you use it.
