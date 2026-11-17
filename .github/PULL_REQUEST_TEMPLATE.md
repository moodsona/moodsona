<!-- Title = the squash commit: Conventional Commit, lowercase subject (feat: / fix: / persona: add <id> / docs: ...). -->

## What and why

<!-- 2-3 sentences. Link the issue: "Closes #123". -->

## How verified

- [ ] `uv run pre-commit run --all-files` passed
- [ ] `uv run pytest -m "not live" -q` passed; new behaviour has a test that fails without the change
- [ ] Persona PR: output of `moodsona validate` and `moodsona smoke` pasted below
- [ ] No real people, private conversations, product logs or customer data

<details><summary>moodsona validate / smoke output (persona PRs)</summary>

```text

```

</details>

## AI assistance

<!-- Required, see CONTRIBUTING.md#ai-assisted-contributions. Write "none" if no AI tools were used. -->
Tools/models: <!-- e.g. Claude Code (Claude Opus 5.5) -->
What they did / how I verified it: <!-- drafted the persona; I read every line and ran the gates -->
