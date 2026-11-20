"""Check rule probe files against the bundled personas they name, offline.

Thin wrapper kept for pre-commit; the logic lives in ``moodsona.probe._check_main``.
A persona edit can shift its rule numbers, so pre-commit runs this whenever a
persona or a probe file changes.

Usage: ``python tools/validate_probes.py [PATH ...]`` (default: the bundled ``probes/``).
Exit code 1 if any problem is found.
"""

from __future__ import annotations

import sys

from moodsona.probe import _check_main

if __name__ == "__main__":
    sys.exit(_check_main(sys.argv[1:]))
