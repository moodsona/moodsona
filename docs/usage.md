# Usage guide

Everything past the [README](../README.md) quickstart: real model providers,
the command line, metrics, the benchmark scaffold and the fidelity check.

## Real models and your bot

`FakeProvider` produces placeholder user messages, so it is for wiring and CI.
For real runs, use `AnthropicProvider` (from `moodsona.llm.anthropic`) or
`OpenAIProvider` (from `moodsona.llm.openai`) as the `user_model`. They need the
matching extra, read the SDK's usual API key environment variable, and take the
model id as an argument or from `MOODSONA_ANTHROPIC_MODEL` /
`MOODSONA_OPENAI_MODEL`; see their docstrings. Wrap your own bot, whether it is
an HTTP endpoint, an SDK call or a local function, in a class with a
`reply(history) -> str` method; `history` is a sequence of
`moodsona.ChatMessage` (`role`, `content`) with everything the bot has seen so
far. `moodsona.list_bundled_personas()` lists the shipped persona ids.

```python
import moodsona
from moodsona.llm.anthropic import AnthropicProvider  # pip install "moodsona[anthropic]"

persona = moodsona.load_bundled_persona("boundary-tester")
# MyBot: your companion, as in the README quickstart
transcript = moodsona.run(persona, MyBot(), user_model=AnthropicProvider(), days=3, turns_per_day=6)
```

`run()` takes the persona and your bot by position and everything else by
keyword. `user_model` has no default, so a run never plays a placeholder user
by accident; `days` and `turns_per_day` default to 3 and 6. Every turn is one
or more paid model calls; start with a short run.

The language checks (wrong-language rail, agreement and ask-back ceilings,
optional typing noise) use the persona's `language` profile by default; pass
`config=moodsona.SimulatorConfig(language_checks=False)` to turn them off. For
finer control (step by step, custom stopping), use `moodsona.Simulator`
directly.

## CLI

```bash
moodsona validate                     # all bundled personas
moodsona validate my-persona.yaml     # your own files or directories
moodsona smoke boundary-tester        # short offline run of a bundled persona
moodsona smoke my-persona.yaml --days 2 --turns 4
```

`validate` checks persona files against [`personas/schema.json`](../personas/schema.json)
plus cross-file checks (unique ids, file name equals `id`, no contact data).
`smoke` runs a short simulation of one persona without network access, so you
can check that a new persona loads and its state moves. Both take file paths
or the ids of bundled personas (an existing file wins over an id), and both
exit with 0 on success, 1 when problems are found or the run fails, and 2 on a
usage error. Run `moodsona <command> --help` for options.

## Metrics

```bash
moodsona report anxious-newcomer.jsonl          # compact table
moodsona report anxious-newcomer.jsonl --json   # the same numbers as JSON
```

```python
from moodsona.metrics import compute_metrics, metrics_from_jsonl

metrics = compute_metrics(transcript)  # or metrics_from_jsonl("anxious-newcomer.jsonl")
print(metrics.to_dict())
```

`moodsona.metrics` measures how well the simulated user held its persona in
one run: emotion and energy (mean, drift per day, volatility), the share of
turns per intent, how often reaction rules fired, how often the user repeated
itself, bare agreement and ask-back rates, message length and the share of
short messages, messages in the wrong language, stance changes per day and
days skipped. The numbers describe the simulated user, not your bot; compare
them across user models, personas or seeds. Exact definitions are in the
module docstring. New in v0.2; names may still change before 1.0.

## Benchmark (scaffold)

`moodsona bench` runs bundled personas with one user model (the model that
plays the persona) against a fixed scripted companion, writes one JSONL per
run to `<out>/<model>/`, and writes `<out>/bench.md`: the metrics above per
model and per run. Run it once per model into the same directory to compare
models. Offline check, free:

```bash
moodsona bench --days 2 --turns 3          # fake provider, the pinned default set
```

With a real provider **every run costs money**: by default the pinned set of
personas (`BENCH_DEFAULT_PERSONAS`, seven today; `--persona` picks others) for
3 days of up to 6 turns, up to 20 model calls per persona and day.
The command refuses to start without `--yes` and prints the worst-case number
of calls first. It checks the results already in `--out` before the first
call. If a run fails, it stops, still writes `bench.md` for the finished runs,
and `--skip-existing` resumes without paying again for runs that are already
there (same days and turns).

```bash
pip install "moodsona[anthropic]"          # or moodsona[openai]
moodsona bench --provider anthropic --model claude-sonnet-5-5 --yes --out bench-results
moodsona bench --provider openai --model gpt-5-mini --yes --out bench-results
```

`bench.md` averages runs per model and run length (days x turns), so runs of
different lengths are never mixed in one row.

This is the scaffold for the user-model leaderboard on the roadmap: there is
no single score yet, and the companion is a fixed script, so the numbers say
how well a user model holds the personas, not how good a companion is.

## Fidelity check (experimental)

Can a simple classifier tell simulated users from real ones? Bring your own
consented reference conversations (moodsona ships none):

```python
import moodsona
from moodsona.fidelity import classifier_two_sample_test

provider = moodsona.FakeProvider()  # placeholder text: swap in a real provider
simulated, personas = [], []
for persona_id in moodsona.list_bundled_personas():
    persona = moodsona.load_bundled_persona(persona_id)
    for seed in range(3):
        transcript = moodsona.run(
            persona, MyBot(), user_model=provider, days=2, turns_per_day=3, rng_seed=seed
        )
        simulated.append(transcript.visible_history())
        personas.append(persona_id)  # one speaker per persona, not per run
reference, speakers = load_my_conversations()  # yours, with one speaker id each
result = classifier_two_sample_test(
    reference, simulated, reference_groups=speakers, simulated_groups=personas
)
print(result.accuracy, result.p_value, result.weights)
```

Pass the speaker ids: conversations by the same person (or persona) are not
independent, and without them the p-value is far too optimistic. Use the same
companion for both sets. With `FakeProvider` every run is the same placeholder
text, so this only shows the wiring; use a real provider (see [Real models and your bot](#real-models-and-your-bot)). It
is a statistical screen on eight surface features, not proof of realism: a
large p-value is no evidence that the sets are alike. See
[the method](METHOD.md#fidelity-check-experimental).
