"""Validate persona YAML files: schema plus cross-file checks.

Thin wrapper kept for pre-commit and CI; the logic lives in
``moodsona.validation`` and is also available as ``moodsona validate``.

Usage: ``python tools/validate_personas.py [PATH ...]`` (default: ``personas``).
Exit code 1 if any problem is found.
"""

from __future__ import annotations

import sys

from moodsona.validation import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
