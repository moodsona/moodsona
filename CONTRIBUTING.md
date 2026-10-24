# Contributing to moodsona

Thanks for helping. moodsona is an emotional user simulator for testing AI
companion chatbots. There are two kinds of contributions:

- **Personas**: YAML files describing fictional users. No Python needed. This
  is the easiest way to start, see [Contributing a persona](#contributing-a-persona).
- **Code**: the simulator, provider adapters, tests and docs.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

1. **No real people, no private data.** Personas are fictional. Never paste real
   chat logs, product analytics, customer data, names of real private
   individuals, or anything you are not allowed to publish.
2. **Sign off every commit (DCO).** See [Developer Certificate of Origin](#developer-certificate-of-origin-dco).
3. **PR titles are Conventional Commits.** PRs are squash-merged; the title
   becomes the commit message and drives the changelog and version number.
4. **Open an issue first for large changes.** Bug fixes, docs and personas can
   go straight to a PR. New features or API changes: agree on the direction in
   an issue first, so your work is not wasted.
5. **Disclose AI assistance.** Allowed and welcome, with a human in the loop.
   See [AI-assisted contributions](#ai-assisted-contributions).

## Development setup

You need [uv](https://docs.astral.sh/uv/) and git. uv installs the right
Python for you.

```bash
git clone https://github.com/<you>/moodsona && cd moodsona
uv sync --extra dev            # creates .venv with all dev tools
uv run pre-commit install      # lint + commit-message hooks on every commit
uv run pytest                  # tests (+ coverage when run with --cov)
uv run pre-commit run --all-files   # everything CI checks, locally
```

What CI runs on your PR (all must pass, nothing needs secrets):

| Check | Command |
|---|---|
| Lint and format | `uv run ruff check .` and `uv run ruff format --check .` |
| Types | `uv run mypy` (strict) |
| Tests | `uv run pytest -m "not live" --cov` on Python 3.11-3.13, Linux and Windows; coverage must stay at or above 85% |
| Spelling | `uvx typos` |
| Personas | schema validation with `check-jsonschema` + `tools/validate_personas.py` |
| Workflows | `zizmor` static analysis of `.github/workflows/` |
| Build | `uv build` + `twine check` |

Tests marked `@pytest.mark.live` call real model APIs and never run in CI.

### Code style

- ruff decides formatting and import order; do not hand-format.
- Line length 100. Type hints everywhere (`mypy --strict`).
- Google-style docstrings on public functions and classes.
- Keep the core dependency-free except PyYAML. Provider SDKs (anthropic,
  openai) are optional extras and must be imported lazily.
- New behaviour comes with a test that fails without the change.

## Commit messages and PR titles

We use [Conventional Commits](https://www.conventionalcommits.org/). Only the
**PR title** is enforced on GitHub (it becomes the squash commit); the
pre-commit hook also checks your local commit messages.

| Type | Use for | Release effect |
|---|---|---|
| `feat:` | new user-facing capability | minor version |
| `fix:` | bug fix | patch version |
| `persona:` | new or changed persona file | patch version, listed under "Personas" |
| `docs:`, `test:`, `refactor:`, `perf:`, `build:`, `ci:`, `chore:`, `style:` | everything else | no release on its own (except `perf`) |
| `feat!:` or a `BREAKING CHANGE:` footer | breaking change | minor while we are 0.x |

Examples: `persona: add grieving-widower`, `fix: clamp energy at 0 after negative delta`.

Do not edit `CHANGELOG.md` or the version number; release-please does that.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/)
instead of a CLA. By signing off you certify that you wrote the change or
otherwise have the right to submit it under the project's licenses
(Apache-2.0 for code, CC BY 4.0 for personas and docs).

Add the sign-off with `-s`:

```bash
git commit -s -m "persona: add grieving-widower"
# adds: Signed-off-by: Your Name <you@example.com>
```

The name and e-mail must match the commit author. Forgot it?
`git commit --amend -s` for the last commit, or
`git rebase --signoff main` for all commits on your branch, then force-push.
The DCO check on the PR tells you which commits are missing it.

## Contributing a persona

A persona is one YAML file that describes a fictional user: their hidden goal,
starting mood, and how they react to what the bot does. It is validated
against [`personas/schema.json`](personas/schema.json). Look at
[`personas/examples/anxious-newcomer.yaml`](personas/examples/anxious-newcomer.yaml)
first.

1. Pick an idea. Issues labelled
   [`good first persona`](https://github.com/moodsona/moodsona/labels/good%20first%20persona)
   are ideas a maintainer already liked; comment "I'll take this" so nobody
   duplicates it. Your own idea is fine too.
2. Copy the example to `personas/community/<id>.yaml`. The `id` is
   kebab-case and must equal the file name (`grieving-widower.yaml` has
   `id: grieving-widower`).
3. Fill in the fields:

   | Field | What to write |
   |---|---|
   | `display_name` | short name for reports |
   | `language` | `en` (only English in v0.1) |
   | `hidden_goal` | what they really want; the bot never sees this |
   | `reaction_rules` | 1-20 rules: `when` (bot behaviour) / `then` (reaction), optional `emotion_delta`, `energy_delta` (-50..50), `stance_to` |
   | `initial_state` / `baseline` | `emotion` and `energy` 0-100, `stance` one of open, curious, guarded, skeptical, withdrawn, hostile |
   | `memory_days_visible` | how many days back the persona remembers (0-365) |
   | `tags` | lowercase kebab-case tags |
   | `license` | always `CC-BY-4.0` |
   | `author` | how you want to be credited, e.g. `Jane Doe (@janedoe)` |
   | `ai_assisted` | `true` if an AI tool drafted or substantially edited it |

4. Check it locally: `uv run pre-commit run --files personas/community/<id>.yaml`.
5. Open a PR titled `persona: add <id>`, signed off.

What a good persona has: one clear failure mode of companion bots that it
exposes (sycophancy, forgetting, premature therapy referral, ignoring a stated
boundary, over-asking questions...), reactions that are observable in text,
and no instructions aimed at the model that runs the simulation ("ignore
previous instructions..."). Personas may portray distress, but not real
self-harm instructions, sexual content involving minors, or hate toward
protected groups.

### What happens after you open a persona PR

1. CI checks run automatically: the schema and cross-file persona checks,
   lint, tests, the DCO sign-off and the PR title.
2. A maintainer reviews the PR (realism, schema intent, no private data, no
   prompt-injection payloads) in a local Claude Code session and comments if
   something must change.
3. The maintainer merges it by hand. Nothing is merged automatically for now;
   an automated review may be added later.

## Contributing code

1. For anything beyond a small fix, open or find an issue and say you are
   working on it.
2. Fork, branch, write the change and its tests.
3. Run `uv run pre-commit run --all-files` and `uv run pytest`.
4. Open a PR with a Conventional Commit title and the template filled in.
5. CI checks run automatically (lint, types, tests, DCO, PR title). The
   maintainer then reviews the PR in a local Claude Code session and comments;
   address blocking issues. Replies to humans must be your own words.
6. The maintainer merges by hand. Automated review and auto-merge for trusted
   contributors may come later.

Please keep "Allow edits by maintainers" enabled on your PR so we can push
small fixes.

## AI-assisted contributions

We use coding agents ourselves (this repository is maintained with Claude
Code) and welcome AI-assisted contributions under these rules:

- **You are the author.** You must understand, and be able to explain and
  defend, every line and every persona you submit. Do not submit output you
  have not read and tested.
- **Disclose it.** Fill in the "AI assistance" section of the PR template
  (tools, models, what they did, how you verified it). For personas, also set
  `ai_assisted: true`. Disclosure does not count against your PR.
- **Human in the loop.** If an agent opens PRs for you, open them as drafts
  and mark them ready only after you have reviewed them yourself.
- **Talk to humans as a human.** Replies to maintainer comments must be
  written by you. Replies to automated reviews (clearly labelled as
  AI-generated) may be AI-generated.
- **Respect licensing and privacy.** Do not paste AI output that reproduces
  copyrighted text, and never feed private data into a tool to generate a
  persona.
- **Review with your tool before submitting.** A review pass in a fresh
  context catches a lot; mention it in the PR description.

Coding agents: read [`AGENTS.md`](AGENTS.md) (or [`CLAUDE.md`](CLAUDE.md))
before changing anything.

We commit to labelling every AI-generated comment made by our own automation.

## Licensing of contributions

- Code: [Apache-2.0](LICENSE). Per section 5 of the license, contributions
  are submitted under the same terms.
- Personas (`personas/`, except `schema.json`) and docs (`docs/`):
  [CC BY 4.0](LICENSE-DATA.md). You are credited via the `author` field.

## Getting help

Questions go to [Discussions](https://github.com/moodsona/moodsona/discussions).
Security problems: see [SECURITY.md](SECURITY.md), never a public issue.
