<!--
PR title = the squash commit on main, so it must be a Conventional Commit:
  feat: ..., fix: ..., persona: add <id>, docs: ..., test: ..., chore: ...
Every commit needs a DCO sign-off: `git commit -s`.
-->

## What and why

<!-- 2-3 sentences. Link the issue: "Closes #123". -->

## Type

- [ ] Persona (only adds files under `personas/community/`)
- [ ] Bug fix
- [ ] Feature / behaviour change
- [ ] Docs, tests or tooling only

## Checklist

- [ ] `uv run pre-commit run --all-files` passes locally
- [ ] `uv run pytest` passes; new behaviour has a test that fails without the change
- [ ] No real people, private conversations, product logs or customer data (personas are fictional)
- [ ] Commits are signed off (`git commit -s`), see CONTRIBUTING.md

## AI assistance

<!-- Required. See "AI-assisted contributions" in CONTRIBUTING.md. -->

- [ ] No AI tools were used
- [ ] AI tools were used. Tools/models: <!-- e.g. Claude Code (Claude Opus 5.5) -->
      What they did: <!-- drafted persona text / wrote tests / reviewed the diff -->
      How I verified it: <!-- read every line, ran the tests, compared with ... -->

## Notes for the reviewer

<!-- Anything non-obvious: trade-offs, follow-ups, how to test manually. -->
