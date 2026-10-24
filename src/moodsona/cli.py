"""Command line interface: ``moodsona validate`` and ``moodsona smoke``.

Exit codes:
    0: success.
    1: ``validate`` found problems, or ``smoke`` could not load or run the persona.
    2: usage error (unknown command or bad arguments, printed by argparse; or
       ``validate`` got a missing path or found no persona file to check).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from moodsona.companion import EchoCompanion
from moodsona.llm.base import LLMError
from moodsona.llm.fake import FakeProvider
from moodsona.personas import PersonaError, load_persona
from moodsona.runner import run
from moodsona.validation import main as validate_main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moodsona",
        description="Emotional user simulator for testing AI companion chatbots.",
        epilog="Exit codes: 0 success, 1 problems found or run failed, 2 usage error.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "validate", help="check persona files (schema, ids, no contact data)"
    )
    check.add_argument(
        "paths", nargs="*", help="files or directories (default: the bundled personas)"
    )
    smoke = commands.add_parser(
        "smoke", help="run a persona offline (fake model, echo companion) and print its mood"
    )
    smoke.add_argument("persona", help="path to a persona YAML file")
    smoke.add_argument("--days", type=int, default=1, help="simulated days (default: 1)")
    smoke.add_argument("--turns", type=int, default=3, help="turns per day (default: 3)")
    smoke.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser


def _smoke(args: argparse.Namespace) -> int:
    try:
        seed = load_persona(args.persona)
    except (PersonaError, FileNotFoundError) as exc:  # FileNotFoundError: broken install
        sys.stderr.write(f"{exc}\n")
        return 1
    if args.days < 1 or args.turns < 1:
        sys.stderr.write("--days and --turns must be at least 1\n")
        return 1
    try:
        transcript = run(seed, FakeProvider(), EchoCompanion(), args.days, args.turns, args.seed)
    except (LookupError, LLMError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    out = sys.stdout
    out.write(f"persona {seed.id}: {len(transcript.turns)} turn(s), offline fake model\n")
    out.write(f"{'day':>3} {'turn':>4} {'emotion':>7} {'energy':>6}  stance\n")
    for point in transcript.trajectory():
        out.write(
            f"{point.day:>3} {point.turn:>4} {point.emotion:>7} {point.energy:>6}  {point.stance}\n"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    args = _parser().parse_args(argv)
    if args.command == "validate":
        return validate_main(args.paths)
    return _smoke(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
