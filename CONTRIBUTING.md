# Contributing to moodsona

Thanks for helping. moodsona is an emotional user simulator for testing AI
companion chatbots; by participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md). Two ways to contribute: a
[persona](#contributing-a-persona) (one YAML file, no Python needed) or
[code and docs](#contributing-code).

## Setup (2 minutes)

You need [uv](https://docs.astral.sh/uv/) and git; uv installs Python for you.

```bash
git clone https://github.com/<you>/moodsona && cd moodsona
uv sync --locked --extra dev        # .venv with all dev tools
uv run pre-commit install           # lint + commit-message hooks on every commit
uv run pre-commit run --all-files   # lint, types, typos and persona checks
uv run pytest -m "not live" -q      # tests (`live` ones call real APIs and never run in CI)
# CI runs the same two plus a wheel build and twine check.
```

## Contributing a persona

The format is [`personas/schema.json`](personas/schema.json); [Writing a
persona](docs/personas.md) explains every field and how to design reaction
rules. Issues labelled
[`good first persona`](https://github.com/moodsona/moodsona/labels/good%20first%20persona)
are ideas a maintainer already liked; comment on one before you start.

1. Copy [`personas/examples/anxious-newcomer.yaml`](personas/examples/anxious-newcomer.yaml)
   to `personas/community/<id>.yaml`; `id` is kebab-case and equals the file name.
2. Fill in the fields. `license: CC-BY-4.0`, `author` = how you want to be
   credited, `ai_assisted: true` if an AI tool drafted or substantially edited it.
3. Check it: `uv run moodsona validate personas/community/<id>.yaml`, then
   `uv run moodsona smoke personas/community/<id>.yaml`.
4. Open a PR titled `persona: add <id>` and paste both outputs into the template.

Personas are fictional: no real people, private conversations, contact data,
URLs or instructions aimed at the model that runs the simulation. Distress is
fine; self-harm instructions, sexual content involving minors and hate are not.

### What happens after you open a persona PR

CI validates the schema, the cross-file checks, lint, tests and the PR title.
The maintainer then reads the file by hand (realism, no private data, no
prompt-injection payloads), comments if something must change, and merges.

## Contributing code

1. For anything beyond a small fix, open or find an issue first to agree on the direction.
2. Fork, branch from `main`, write a test that fails without the change, then
   the change. Run the two gate commands from [Setup](#setup-2-minutes).
3. Open a PR with the template filled in; keep "Allow edits by maintainers" on.

Conventions the gates do not check: the core stays dependency-free except PyYAML
(provider SDKs are optional extras, imported lazily); Google-style docstrings on
public API; `CHANGELOG.md` and the version are edited by release-please only.

## Pull request rules

- **Title** is a [Conventional Commit](https://www.conventionalcommits.org/)
  with a lowercase subject (`persona: add grieving-widower`,
  `fix: clamp energy at 0 after negative delta`, `feat!:` for a breaking change).
  PRs are squash-merged: the title becomes the commit and drives the changelog.
- **CI must be green** (it needs no secrets) and **AI assistance is disclosed**.
- **Review is by hand.** The maintainer reads every PR, comments and merges;
  nothing is merged automatically.

## AI-assisted contributions

Allowed and welcome, with a human in the loop (this repository is itself
maintained with [Claude Code](https://claude.com/claude-code)):

- **You are the author.** Understand and be able to explain every line you
  submit; do not submit output you have not read and tested. Reply to review
  comments yourself.
- **Disclose it** in the PR template: tools and models, what they did, how you
  verified it. For personas also set `ai_assisted: true`. AI co-author trailers
  (`Co-Authored-By:`) in commits are fine. Disclosure never counts against your PR.
- **Respect licensing and privacy.** Do not paste AI output that reproduces
  copyrighted text, and never feed private data into a tool to generate a persona.

Coding agents: read [`AGENTS.md`](AGENTS.md) before changing anything.

## Licensing of contributions

Inbound = outbound: by submitting a contribution you agree that it is licensed
under the same terms as the part of the project it lands in. Opening the PR is
the agreement; there is no CLA and no sign-off.

- Code: [Apache-2.0](LICENSE). Per section 5 of the license, contributions
  are submitted under the same terms.
- Personas (`personas/`, except `schema.json`) and docs (`docs/`):
  [CC BY 4.0](LICENSE-DATA.md). You are credited via the `author` field.

Only submit work you wrote yourself or otherwise have the right to submit under these licenses.

## Getting help

Questions: [Discussions](https://github.com/moodsona/moodsona/discussions).
Security problems: [SECURITY.md](SECURITY.md), never a public issue.
