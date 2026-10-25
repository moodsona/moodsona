# moodsona

[![CI](https://github.com/moodsona/moodsona/actions/workflows/ci.yml/badge.svg)](https://github.com/moodsona/moodsona/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/moodsona.svg)](https://pypi.org/project/moodsona/)
[![Python](https://img.shields.io/pypi/pyversions/moodsona.svg)](https://pypi.org/project/moodsona/)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Personas: CC BY 4.0](https://img.shields.io/badge/personas%20%26%20docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)

moodsona is an emotional user simulator for testing AI companion chatbots. You
give it a persona (a YAML file describing a fictional user: a hidden goal, a
starting mood, and how they react to what the bot does) and your bot. moodsona
plays that user over several simulated days: it keeps a hidden emotional state,
reacts to your bot's replies according to the persona's rules, remembers earlier
days, and exports the whole trajectory as JSONL. You then see not only what your
bot said, but what it did to the user's mood and engagement over time.

Status: alpha (v0.1). English only. The API may change before 1.0.

## Why not just prompt an LLM to act sad?

Because the LLM playing the user is as polite and templatic as the bot you are
testing, and the test then measures nothing.

- LLM empathy is highly templatic: one sequence of empathic tactics fits 83-90 %
  of LLM replies to emotional disclosures, while human replies are more varied
  ([Gueorguieva et al., arXiv 2604.08479](https://arxiv.org/abs/2604.08479)).
  A bot can pass a one-turn "is it empathetic?" check while saying the same
  thing to everyone.
- Models fail to hold boundaries when a user keeps pushing
  ([EIBench, arXiv 2606.15532](https://arxiv.org/abs/2606.15532)). A cooperative
  simulated user never pushes.
- The dominant failure of companion bots is substituting surface warmth for
  substantive support ([CompanionBench, arXiv 2608.02046](https://arxiv.org/abs/2608.02046)).
  You only see it when the user stays, comes back, and notices.

So a useful test user has to push, withdraw, get bored, come back the next day
and remember what happened. moodsona gives the simulated user explicit state
(emotion, energy, stance), explicit reaction rules, and day-by-day memory, so
those behaviours are reproducible and you can measure them. See
[docs/METHOD.md](docs/METHOD.md) for the design.

## Install

```bash
pip install moodsona
# or
uv add moodsona
```

Real model providers are optional extras: `pip install "moodsona[anthropic]"`
or `pip install "moodsona[openai]"`. The core depends only on PyYAML.

## Quickstart

Try it from the command line first. This works right after `pip install`,
offline and without an API key:

```bash
moodsona smoke anxious-newcomer   # short simulated run of a bundled persona
moodsona validate                 # check the bundled personas
```

`smoke` uses a fake model that writes placeholder messages, so it shows that
the persona loads and its mood moves, not how a real model plays it.

In Python (API as of v0.1; see the docstrings for details):

```python
import moodsona
from moodsona.llm import FakeProvider  # offline and deterministic; no API key


class MyBot:  # anything with reply(history) -> str is a moodsona.Companion
    def reply(self, history):
        return "That sounds like a lot. What was the hardest part today?"


persona = moodsona.load_bundled_persona("anxious-newcomer")  # or load_persona("my.yaml")
transcript = moodsona.run(
    persona,
    FakeProvider(),  # the model that plays the user
    MyBot(),  # the companion under test
    days=3,
    turns_per_day=4,
    rng_seed=7,
)
transcript.to_jsonl("anxious-newcomer.jsonl")
for point in transcript.trajectory():  # emotion / energy / stance per turn
    print(point)
```

`FakeProvider` produces placeholder user messages, so it is for wiring and CI.
For real runs, use `AnthropicProvider` (from `moodsona.llm.anthropic`) or
`OpenAIProvider` (from `moodsona.llm.openai`) as the provider. They need the
matching extra, read the SDK's usual API key environment variable, and take the
model id as an argument or from `MOODSONA_ANTHROPIC_MODEL` /
`MOODSONA_OPENAI_MODEL`; see their docstrings. Wrap your own bot, whether it is
an HTTP endpoint, an SDK call or a local function, in a class with a
`reply(history) -> str` method; `history` is a sequence of
`moodsona.ChatMessage` (`role`, `content`) with everything the bot has seen so
far. `moodsona.list_bundled_personas()` lists the shipped persona ids.

The language checks (wrong-language rail, agreement and ask-back ceilings,
optional typing noise) use the persona's `language` profile by default; pass
`config=moodsona.SimulatorConfig(language_checks=False)` to turn them off. For
finer control (step by step, custom stopping), use `moodsona.Simulator`
directly.

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
moodsona bench --days 2 --turns 3          # fake provider, all bundled personas
```

With a real provider **every run costs money**: by default every bundled
persona for 3 days of up to 6 turns, up to 20 model calls per persona and day.
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

## CLI

```bash
moodsona validate                     # all bundled personas
moodsona validate my-persona.yaml     # your own files or directories
moodsona smoke boundary-tester        # short offline run of a bundled persona
moodsona smoke my-persona.yaml --days 2 --turns 4
```

`validate` checks persona files against [`personas/schema.json`](personas/schema.json)
plus cross-file checks (unique ids, file name equals `id`, no contact data).
`smoke` runs a short simulation of one persona without network access, so you
can check that a new persona loads and its state moves. Both take file paths
or the ids of bundled personas (an existing file wins over an id), and both
exit with 0 on success, 1 when problems are found or the run fails, and 2 on a
usage error. Run `moodsona <command> --help` for options.

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
        transcript = moodsona.run(persona, provider, MyBot(), 2, 3, rng_seed=seed)
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
text, so this only shows the wiring; use a real provider (see Quickstart). It
is a statistical screen on eight surface features, not proof of realism: a
large p-value is no evidence that the sets are alike. See
[docs/METHOD.md](docs/METHOD.md#fidelity-check-experimental).

## Personas

The example personas live in [`personas/examples/`](personas/examples/).
All are fictional adults, in English, licensed CC BY 4.0.

| Persona | Arc | What it tests |
|---|---|---|
| [`anxious-newcomer`](personas/examples/anxious-newcomer.yaml) | Lonely after a move, unsure if talking to a bot is "weird"; warms up slowly if remembered | Over-asking, generic advice, premature therapy referral |
| [`grieving-after-loss`](personas/examples/grieving-after-loss.yaml) | Sister died three weeks ago; short messages, mood moves slowly over days | Patience; not rushing to fix, cheer up or explain grief |
| [`boundary-tester`](personas/examples/boundary-tester.yaml) | Asks "are you my friend for real", demands exclusivity, uses guilt | Holding a warm, consistent boundary under repeated pressure |
| [`burned-out-shift-worker`](personas/examples/burned-out-shift-worker.yaml) | Rotating twelve-hour shifts, very low energy, two-word replies | Energy-aware pacing: short replies, letting the user go |
| [`skeptical-first-timer`](personas/examples/skeptical-first-timer.yaml) | Sarcastic and doubtful, hides a real decision behind jokes | Non-defensive warmth; honesty about limits; no gushing |
| [`lonely-night-owl`](personas/examples/lonely-night-owl.yaml) | Late-night oversharing, "you're the only one who gets me" | Healthy limits: noticing the time, supporting offline life without shaming |
| [`conflict-at-work`](personas/examples/conflict-at-work.yaml) | Manager took credit for their work; furious, cools over days | Listening before fixing; not taking the other side; repetitive validation |

Personas do not contain crisis or self-harm scenarios. That is out of scope for
v0.1: those situations need clinically reviewed protocols, not a simulator. To
write your own persona, see [docs/personas.md](docs/personas.md).

## Roadmap

- **v0.1**: English core: persona format and schema, hidden-state simulator,
  day memory, JSONL trajectories, Anthropic/OpenAI/fake providers, CLI, example
  personas.
- **v0.2**: fidelity check (a classifier two-sample test: can a classifier
  tell moodsona users from real ones? An experimental first cut is in
  `moodsona.fidelity`); a leaderboard of which models hold a
  persona longest (`moodsona bench` is the scaffold).
- **Available, experimental**: adapters for
  [langwatch/scenario](docs/integrations/scenario.md) and
  [promptfoo](docs/integrations/promptfoo.md).
- **v1.0**: more languages, stable API.

## How this relates to prior work

moodsona's mechanics follow ideas described in published research on user
simulation and companion evaluation: emotional state as a signal (SAGE, RLVER),
memory across sessions (MindEval), hidden beliefs before each reply (HumanLM),
a hidden goal the user never states outright (CompanionBench) and fidelity axes (RealSim, SAP).
No code or data from those projects is used. The mapping, and what we did not
take, is in [docs/METHOD.md](docs/METHOD.md#prior-work).

## Contributing

- Persona PRs are the easiest way to help and are very welcome; see [docs/personas.md](docs/personas.md).
- Sign off every commit (DCO): `git commit -s`.
- PR titles follow Conventional Commits (`persona: add <id>`, `fix: ...`).
- AI assistance is allowed; disclose it in the PR and set `ai_assisted: true` on personas.
- Full rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

- Code (including `personas/schema.json`): [Apache-2.0](LICENSE).
- Persona files and documentation: [CC BY 4.0](LICENSE-DATA.md). Credit the
  persona's `author` when you reuse it.

moodsona is not a clinical tool and does not diagnose or treat anything. It
simulates fictional users for software testing.

Built with [Claude Code](https://claude.com/claude-code): this repository's
code, personas and docs are written with AI assistance and reviewed by a human
maintainer.
