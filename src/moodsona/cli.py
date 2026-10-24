"""Command line interface: ``moodsona validate`` and ``moodsona smoke``.

Exit codes:
    0: success.
    1: ``validate`` found problems, ``smoke`` could not load or run the persona,
       or ``report`` could not read the transcript.
    2: usage error (unknown command or bad arguments, printed by argparse; or
       ``validate`` got a missing path or found no persona file to check).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from moodsona.companion import EchoCompanion
from moodsona.llm.base import LLMError
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import metrics_from_jsonl
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
    report = commands.add_parser(
        "report", help="print metrics of a transcript JSONL (how well the persona was held)"
    )
    report.add_argument("transcript", help="a file written by Transcript.to_jsonl")
    report.add_argument("--json", action="store_true", help="print the metrics as JSON")
    return parser


def _num(value: float | None, spec: str = ".1f") -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value: float | None) -> str:
    return _num(value, ".0%")


def _report(args: argparse.Namespace) -> int:
    try:
        m = metrics_from_jsonl(args.transcript)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(m.to_dict(), indent=2) + "\n")
        return 0
    lines = [
        f"persona {m.persona_id} (seed {m.rng_seed}): {m.turns} turn(s) on "
        f"{m.active_days} active day(s), {m.days_skipped} day(s) skipped",
        f"{'':<15}{'mean':>6}{'drift/day':>11}{'volatility':>12}",
    ]
    for name, s in (("emotion", m.emotion), ("energy", m.energy)):
        lines.append(
            f"{name:<15}{_num(s.mean):>6}{_num(s.drift_per_day, '+.2f'):>11}"
            f"{_num(s.volatility):>12}"
        )
    intents = ", ".join(f"{k} {_pct(v)}" for k, v in m.intent_shares.items() if v)
    rules = ", ".join(f"rule {n} x{count}" for n, count in m.rule_counts.items())
    rows = [
        ("intents", intents or "n/a"),
        ("rule triggers", _pct(m.rule_trigger_rate) + (f" ({rules})" if rules else "")),
        ("own repeats", _pct(m.own_repeat_rate)),
        ("agreement", _pct(m.agreement_rate)),
        ("ask-back", _pct(m.ask_back_rate)),
        ("length", f"{_num(m.mean_message_chars)} chars, {_pct(m.short_message_share)} short"),
        ("wrong language", _pct(m.wrong_language_share)),
        ("stance changes", f"{_num(m.stance_changes_per_day, '.2f')} per day"),
        ("polluted", _pct(m.polluted_share)),
        ("forced", _pct(m.forced_share)),
    ]
    lines += [f"{label:<15}{value}" for label, value in rows]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


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
    if args.command == "report":
        return _report(args)
    if args.command == "validate":
        return validate_main(args.paths)
    return _smoke(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
