# Deferred automation

`claude-review.yml` is a ready-to-use GitHub Actions workflow that reviews pull
requests with Claude (via `anthropics/claude-code-action`) and can enable
auto-merge for narrow classes of PRs. It is parked here on purpose: in the
current stage the maintainer reviews every PR in a local Claude Code session
and merges by hand. To enable it later, move the file to `.github/workflows/`,
add the `ANTHROPIC_API_KEY` secret, and allow the `pull_request_target` event
in the repository's Actions policy.
