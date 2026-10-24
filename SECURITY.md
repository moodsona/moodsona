# Security policy

## Supported versions

moodsona is pre-1.0. Only the latest release on PyPI receives security fixes.

## Reporting a vulnerability

Please **do not** open a public issue, discussion or pull request for a
security problem.

Report it privately through GitHub:
**Security tab > Report a vulnerability**
(https://github.com/moodsona/moodsona/security/advisories/new).

Include what you found, how to reproduce it, and the impact you expect. You
will get an acknowledgement within 7 days and a status update at least every
14 days until the issue is resolved. We credit reporters in the advisory unless
you ask us not to.

## Scope

In scope:

- code in `src/moodsona/` and the published `moodsona` package on PyPI;
- the persona loader and schema (for example, a persona file that leads to
  code execution, file access, or unbounded resource use when loaded);
- the GitHub Actions workflows in this repository (for example, a way for a
  pull request from a fork to obtain secrets or write access).

Out of scope:

- behaviour of third-party model APIs that moodsona calls;
- prompt-injection text inside a persona that only changes what the simulated
  user *says* (please report these as normal issues so we can remove the
  persona; personas are reviewed for this before merge);
- vulnerabilities in dependencies that are already public (Dependabot tracks
  those).

## Supply chain

Releases are built in GitHub Actions and published to PyPI with Trusted
Publishing (OIDC, no long-lived tokens), with Sigstore attestations. Workflow
actions are pinned to full commit SHAs, and workflows are scanned with zizmor,
CodeQL and OpenSSF Scorecard.
