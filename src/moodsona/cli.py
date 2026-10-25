"""Command line interface: ``moodsona validate``, ``smoke``, ``run``, ``report`` and ``bench``.

``validate``, ``smoke`` and ``run`` take persona file paths or the ids of
bundled personas (``moodsona smoke anxious-newcomer``). Only a bare name (no
directory part, no ``.yaml``/``.yml`` suffix) is looked up as an id, and only
when no file of that name exists (for ``validate``: no file or directory).

Exit codes:
    0: success.
    1: ``validate`` found problems, ``smoke`` or ``run`` could not load or run
       the persona (or ``run`` could not write the transcript), ``report``
       could not read the transcript, or ``bench`` failed.
    2: usage error (unknown command or bad arguments, printed by argparse; or
       ``validate`` got a missing path or found no persona file to check; or
       ``bench`` got an unknown persona; or ``run`` got an unknown companion
       spec, an unreadable or empty script file, or an ``--out`` it cannot
       write, or a paid run would overwrite an existing ``--out``; or ``bench``
       or ``run`` got a paid provider without ``--yes``).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from moodsona import bench
from moodsona.companion import Companion, EchoCompanion, ScriptedCompanion
from moodsona.llm.base import LLMError
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import (
    TranscriptMetrics,
    compute_metrics,
    format_number,
    format_share,
    metrics_from_jsonl,
)
from moodsona.personas import PersonaError, bundled_persona_path, load_persona
from moodsona.runner import Transcript, run
from moodsona.state import PersonaSeed
from moodsona.validation import YAML_SUFFIXES
from moodsona.validation import main as validate_main

#: The ``moodsona run --companion`` specs, as shown in help and error messages.
COMPANION_SPECS = ("echo", "script:<file>")


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


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
        "paths",
        nargs="*",
        help="files, directories or bundled persona ids (default: all bundled personas)",
    )
    smoke = commands.add_parser(
        "smoke", help="run a persona offline (fake model, echo companion) and print its mood"
    )
    smoke.add_argument(
        "persona", help="path to a persona YAML file, or a bundled persona id (anxious-newcomer)"
    )
    smoke.add_argument("--days", type=int, default=1, help="simulated days (default: 1)")
    smoke.add_argument("--turns", type=int, default=3, help="turns per day (default: 3)")
    smoke.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    run_cmd = commands.add_parser(
        "run",
        help="run one persona against a companion, write the transcript JSONL and print "
        "its mood and metrics (a real provider costs money)",
    )
    run_cmd.add_argument(
        "persona", help="path to a persona YAML file, or a bundled persona id (anxious-newcomer)"
    )
    run_cmd.add_argument(
        "--provider", choices=bench.PROVIDERS, default="fake", help="the user model (default: fake)"
    )
    run_cmd.add_argument("--model", help="model id for anthropic/openai (default: the adapter's)")
    run_cmd.add_argument(
        "--companion",
        default="echo",
        help=f"the companion under test: {' or '.join(COMPANION_SPECS)}, a UTF-8 file with "
        "one reply per line, used in order and then from the start again (default: echo)",
    )
    run_cmd.add_argument("--days", type=_positive, default=3, help="simulated days (default: 3)")
    run_cmd.add_argument("--turns", type=_positive, default=6, help="turns per day (default: 6)")
    run_cmd.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    run_cmd.add_argument(
        "--out",
        default="run.jsonl",
        help="transcript JSONL to write; a paid run never overwrites one (default: run.jsonl)",
    )
    run_cmd.add_argument(
        "--yes", action="store_true", help="confirm that a paid provider may be called"
    )
    report = commands.add_parser(
        "report", help="print metrics of a transcript JSONL (how well the persona was held)"
    )
    report.add_argument("transcript", help="a file written by Transcript.to_jsonl")
    report.add_argument("--json", action="store_true", help="print the metrics as JSON")
    bench_cmd = commands.add_parser(
        "bench",
        help="run bundled personas with one user model and write bench.md "
        "(a real provider costs money)",
    )
    bench_cmd.add_argument(
        "--provider", choices=bench.PROVIDERS, default="fake", help="the user model (default: fake)"
    )
    bench_cmd.add_argument("--model", help="model id for anthropic/openai (default: the adapter's)")
    bench_cmd.add_argument(
        "--persona",
        action="append",
        help="bundled persona id; repeat for several (default: all bundled personas)",
    )
    bench_cmd.add_argument("--days", type=_positive, default=3, help="simulated days (default: 3)")
    bench_cmd.add_argument("--turns", type=_positive, default=6, help="turns per day (default: 6)")
    bench_cmd.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    bench_cmd.add_argument(
        "--out", default="bench-results", help="output directory (default: bench-results)"
    )
    bench_cmd.add_argument(
        "--yes", action="store_true", help="confirm that a paid provider may be called"
    )
    bench_cmd.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep runs already in --out with the same days and turns (resume a failed bench)",
    )
    return parser


def _persona_path(arg: str, *, dirs: bool) -> str:
    """``arg``, or the bundled persona file when ``arg`` is a bare id and no such path exists.

    Only a bare name (no directory part, no YAML suffix) is looked up as an id.

    Args:
        arg: A command line argument.
        dirs: Whether an existing directory counts as a path (``validate``) or
            not (``smoke``, which needs a file).

    Raises:
        KeyError: ``arg`` is a bare name, but no bundled persona has that id
            (the message lists the bundled ids).
        FileNotFoundError: The bundled personas are missing (a broken install).
    """
    path = Path(arg)
    exists = path.exists() if dirs else path.is_file()
    if exists or path.name != arg or path.suffix in YAML_SUFFIXES:
        return arg
    return str(bundled_persona_path(arg))


def _validate(args: argparse.Namespace) -> int:
    paths: list[str] = []
    for arg in args.paths:
        try:
            paths.append(_persona_path(arg, dirs=True))
        except KeyError as exc:
            sys.stderr.write(f"{arg}: no such file or directory, and {exc.args[0]}\n")
            return 2
        except FileNotFoundError as exc:  # broken install
            sys.stderr.write(f"{arg}: {exc}\n")
            return 2
    return validate_main(paths)


def _load_cli_persona(arg: str) -> PersonaSeed | None:
    """Load the persona ``arg`` (a file or a bundled id); print the problem and return None."""
    try:
        return load_persona(_persona_path(arg, dirs=False))
    except KeyError as exc:  # only the id lookup raises it; load_persona checks the schema first
        sys.stderr.write(f"{arg}: no such file, and {exc.args[0]}\n")
    except (PersonaError, FileNotFoundError) as exc:  # FileNotFoundError: broken install
        sys.stderr.write(f"{exc}\n")
    return None


def _write_trajectory(out: TextIO, transcript: Transcript) -> None:
    """Write the day / turn / emotion / energy / stance table of a run."""
    out.write(f"{'day':>3} {'turn':>4} {'emotion':>7} {'energy':>6}  stance\n")
    for point in transcript.trajectory():
        out.write(
            f"{point.day:>3} {point.turn:>4} {point.emotion:>7} {point.energy:>6}  {point.stance}\n"
        )


def _smoke(args: argparse.Namespace) -> int:
    seed = _load_cli_persona(args.persona)
    if seed is None:
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
    _write_trajectory(out, transcript)
    return 0


def _read_script(path: str) -> list[str]:
    """The replies in a script file: one per line, blank lines skipped.

    Raises:
        ValueError: The file cannot be read, is not UTF-8, or has no replies.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read the script file {path}: {exc}") from exc
    # read_text turns \r\n into \n; split on it only, so a reply stays one line.
    replies = [line for line in text.split("\n") if line.strip()]
    if not replies:
        raise ValueError(f"the script file {path} has no replies (one reply per line)")
    return replies


def _companion(spec: str) -> Companion:
    """The companion for a ``--companion`` spec (see :data:`COMPANION_SPECS`).

    Raises:
        ValueError: An unknown spec, or a script file that cannot be used.
    """
    kind, _, arg = spec.partition(":")
    if spec == "echo":
        return EchoCompanion()
    if kind == "script" and arg:
        return ScriptedCompanion(_read_script(arg), cycle=True)
    raise ValueError(f"unknown companion {spec!r}; use {' or '.join(COMPANION_SPECS)}")


def _refuse_paid(provider: str, yes: bool, personas: int, days: int, turns: int) -> bool:
    """Print the cost estimate and return True when a paid provider lacks ``--yes``."""
    if provider == "fake" or yes:
        return False
    calls = bench.max_model_calls(personas, days, turns)
    sys.stderr.write(
        f"--provider {provider} calls a paid API and costs money: up to {calls} model "
        f"calls ({personas} persona(s) x {days} day(s) x {turns} turn(s)). "
        "Add --yes to run it.\n"
    )
    return True


def _check_out(path: Path, *, paid: bool) -> str | None:
    """The problem with ``--out`` that can be seen without writing anything, or None."""
    if path.is_dir():
        return f"--out {path}: is a directory"
    if paid and path.exists():
        return (
            f"--out {path}: already exists, and a paid run does not overwrite it; "
            "choose another --out or remove the file"
        )
    return None


def _run(args: argparse.Namespace) -> int:
    err, out = sys.stderr, sys.stdout
    seed = _load_cli_persona(args.persona)
    if seed is None:
        return 1
    try:
        companion = _companion(args.companion)
    except ValueError as exc:
        err.write(f"--companion: {exc}\n")
        return 2
    target = Path(args.out)
    problem = _check_out(target, paid=args.provider != "fake")
    if problem:
        err.write(f"{problem}\n")
        return 2
    if _refuse_paid(args.provider, args.yes, 1, args.days, args.turns):
        return 2
    try:  # make sure the transcript can be written before the first (maybe paid) call
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=target.parent):
            pass
    except OSError as exc:
        err.write(f"--out {target}: cannot write there: {exc}\n")
        return 2
    try:
        provider = bench.make_provider(args.provider, args.model)
        transcript = run(seed, provider, companion, args.days, args.turns, args.seed)
        transcript.to_jsonl(target)
    except (LookupError, LLMError, OSError) as exc:  # LLMError: e.g. the SDK is not installed
        err.write(f"{exc}\n")
        return 1
    label = bench.model_label(args.provider, provider)
    out.write(
        f"persona {seed.id}: {len(transcript.turns)} turn(s), "
        f"model {label}, companion {args.companion}\n"
    )
    _write_trajectory(out, transcript)
    out.write(f"wrote {target}\n")
    lines = _summary_lines(compute_metrics(transcript, language=seed.language))
    out.write("\n".join(lines) + "\n")
    return 0


def _summary_lines(m: TranscriptMetrics) -> list[str]:
    """The compact metrics table that ``moodsona report`` prints."""
    num, pct = format_number, format_share
    lines = [
        f"persona {m.persona_id} (seed {m.rng_seed}): {m.turns} turn(s) on "
        f"{m.active_days} active day(s), {m.days_skipped} day(s) skipped",
        f"{'':<15}{'mean':>6}{'drift/day':>11}{'volatility':>12}",
    ]
    for name, s in (("emotion", m.emotion), ("energy", m.energy)):
        lines.append(
            f"{name:<15}{num(s.mean):>6}{num(s.drift_per_day, '+.2f'):>11}{num(s.volatility):>12}"
        )
    intents = ", ".join(f"{k} {pct(v)}" for k, v in m.intent_shares.items() if v)
    rules = ", ".join(f"rule {n} x{count}" for n, count in m.rule_counts.items())
    rows = [
        ("intents", intents or "n/a"),
        ("rule triggers", pct(m.rule_trigger_rate) + (f" ({rules})" if rules else "")),
        ("own repeats", pct(m.own_repeat_rate)),
        ("agreement", pct(m.agreement_rate)),
        ("ask-back", pct(m.ask_back_rate)),
        (
            "length",
            f"{num(m.mean_message_chars)} chars, {pct(m.short_message_share)} short",
        ),
        ("wrong language", pct(m.wrong_language_share)),
        ("stance changes", f"{num(m.stance_changes_per_day, '.2f')} per day"),
        ("polluted", pct(m.polluted_share)),
        ("forced", pct(m.forced_share)),
    ]
    lines += [f"{label:<15}{value}" for label, value in rows]
    return lines


def _report(args: argparse.Namespace) -> int:
    try:
        m = metrics_from_jsonl(args.transcript)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(m.to_dict(), indent=2) + "\n")
        return 0
    sys.stdout.write("\n".join(_summary_lines(m)) + "\n")
    return 0


def _bench(args: argparse.Namespace) -> int:
    err, out = sys.stderr, sys.stdout
    try:
        personas = bench.load_personas(args.persona)
    except KeyError as exc:
        err.write(f"{exc.args[0]}\n")
        return 2
    except (PersonaError, FileNotFoundError) as exc:  # FileNotFoundError: broken install
        err.write(f"{exc}\n")
        return 1
    if _refuse_paid(args.provider, args.yes, len(personas), args.days, args.turns):
        return 2
    try:
        provider = bench.make_provider(args.provider, args.model)
        report = bench.run_bench(
            personas,
            provider,
            model=bench.model_label(args.provider, provider),
            out_dir=args.out,
            days=args.days,
            turns_per_day=args.turns,
            rng_seed=args.seed,
            skip_existing=args.skip_existing,
            progress=lambda line: out.write(f"{line}\n"),
        )
    except bench.BenchRunError as exc:
        err.write(f"{exc}\n")
        if exc.report is not None:
            out.write(f"wrote {exc.report}\n")
        err.write("Stopped at the first failure; rerun with --skip-existing to resume.\n")
        return 1
    except (LLMError, OSError, ValueError) as exc:  # LLMError: e.g. the SDK is not installed
        err.write(f"{exc}\n")
        return 1
    out.write(f"wrote {report}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    args = _parser().parse_args(argv)
    if args.command == "report":
        return _report(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "run":
        return _run(args)
    if args.command == "validate":
        return _validate(args)
    return _smoke(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
