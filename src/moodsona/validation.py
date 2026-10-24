"""Persona file checks: the schema plus the rules JSON Schema cannot express.

For every file:

* it parses with a safe YAML loader and has no duplicate keys,
* it matches ``personas/schema.json``,
* ``id`` equals the file name (without ``.yaml``),
* ids are unique across the whole ``personas/`` tree,
* free-text fields contain no e-mail addresses, phone numbers or URLs
  (personas are fictional; contact data or links usually mean a real person
  or an injected instruction).

Used by ``moodsona validate`` and by ``tools/validate_personas.py``.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

import yaml

from moodsona.personas import bundled_personas_dir, load_schema
from moodsona.schema import validate

TEXT_FIELDS = ("display_name", "description", "hidden_goal", "author")
RULE_TEXT_FIELDS = ("when", "then")

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
CONTACT_PATTERNS = (("an e-mail address", EMAIL), ("a URL", URL))
#: Runs of digit groups joined by a single space, dot or dash; parentheses only
#: around a whole group, as in "(555)". Candidates for :func:`looks_like_phone`.
_PHONE_CANDIDATE = re.compile(
    r"(?<![\w+])\+?(?:\(\d{1,4}\)|\d+)(?:[\s.-]?\(\d{1,4}\)[\s.-]?\d+|[\s.-]\d+)*", re.ASCII
)
_YEAR_RANGE = re.compile(r"\d{4}\s*-\s*\d{4}", re.ASCII)
PHONE_DIGITS = range(7, 16)
"""Digit counts that look like a phone number (E.164 allows at most 15)."""


def looks_like_phone(text: str) -> bool:
    """Tell whether ``text`` contains something that looks like a phone number.

    A candidate needs 7 to 15 digits; a year range such as ``2019-2024`` is not one.
    """
    for match in _PHONE_CANDIDATE.finditer(text):
        candidate = match.group()
        digits = sum(ch.isdigit() for ch in candidate)
        if digits in PHONE_DIGITS and not _YEAR_RANGE.fullmatch(candidate):
            return True
    return False


def default_root() -> Path | None:
    """The ``personas/`` tree checked by default, independent of the working directory.

    That is the bundled copy, or the repository's ``personas/`` in a source
    checkout; None when neither exists (a broken install).
    """
    try:
        return bundled_personas_dir()
    except FileNotFoundError:
        return None


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def iter_files(args: Sequence[str | Path]) -> Iterator[Path]:
    """Yield persona YAML files from the given files/directories."""
    for target in map(Path, args):
        if target.is_dir():
            yield from sorted(p for p in target.rglob("*.y*ml") if p.suffix in {".yaml", ".yml"})
        elif target.suffix in {".yaml", ".yml"}:
            yield target


def text_values(data: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield (field path, text) pairs for every free-text field in a persona."""
    for name in TEXT_FIELDS:
        value = data.get(name)
        if isinstance(value, str):
            yield name, value
    rules = data.get("reaction_rules")
    if isinstance(rules, list):
        for i, rule in enumerate(rules):
            if isinstance(rule, dict):
                for name in RULE_TEXT_FIELDS:
                    value = rule.get(name)
                    if isinstance(value, str):
                        yield f"reaction_rules[{i}].{name}", value


def check_file(path: Path, ids: dict[str, Path], *, schema: bool = True) -> list[str]:
    """Return a list of problems found in one persona file.

    Args:
        path: The persona file.
        ids: Ids seen so far (updated in place) for the uniqueness check.
        schema: Also validate against ``personas/schema.json``.
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return [f"{path}: cannot parse YAML: {exc}"]
    if not isinstance(data, dict):
        return [f"{path}: top level must be a mapping"]

    problems: list[str] = []
    if schema:
        problems += [f"{path}: schema: {error}" for error in validate(data, load_schema())]
    persona_id = data.get("id")
    if persona_id != path.stem:
        problems.append(f"{path}: id {persona_id!r} must equal the file name {path.stem!r}")
    if isinstance(persona_id, str):
        if persona_id in ids and ids[persona_id].resolve() != path.resolve():
            problems.append(f"{path}: id {persona_id!r} is already used by {ids[persona_id]}")
        else:
            ids[persona_id] = path

    for name, value in text_values(data):
        for label, pattern in CONTACT_PATTERNS:
            if name == "author" and pattern is EMAIL:
                continue  # an author may credit themselves with an address
            if pattern.search(value):
                problems.append(f"{path}: {name} contains what looks like {label}")
        if looks_like_phone(value):
            problems.append(f"{path}: {name} contains what looks like a phone number")
    return problems


def main(
    argv: Sequence[str],
    *,
    schema: bool = True,
    out: TextIO | None = None,
    err: TextIO | None = None,
    root: Path | None = None,
) -> int:
    """Validate the given persona files or directories.

    Args:
        argv: Files or directories; none means the whole ``root`` tree.
        schema: Also validate against ``personas/schema.json``.
        out: Stream for the summary line (default: stdout).
        err: Stream for problems (default: stderr).
        root: The ``personas/`` tree whose ids must stay unique (default:
            :func:`default_root`, which does not depend on the working directory).

    Returns:
        0 when every file passes, 1 when any problem is found, 2 when a path
        does not exist or there is no persona file to check.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    root = root or default_root()
    # A typo or an empty selection must not pass as "checked 0 files".
    missing = [arg for arg in argv if not Path(arg).exists()]
    targets: list[str | Path] = list(argv) or ([root] if root is not None else [])
    explicit = list(iter_files(targets))
    if missing or not explicit:
        for arg in missing:
            err.write(f"{arg}: no such file or directory\n")
        if not missing:
            err.write("no persona files to check (expected .yaml/.yml files or directories)\n")
        return 2
    ids: dict[str, Path] = {}
    # Register every existing id first, so a new file that reuses one is caught
    # even when only the new file is passed on the command line.
    passed = {path.resolve() for path in explicit}
    universe = list(iter_files([root])) if root is not None and root.is_dir() else []
    for path in universe:
        if path.resolve() not in passed:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                continue
            if isinstance(data, dict) and isinstance(data.get("id"), str):
                ids.setdefault(data["id"], path)

    problems: list[str] = []
    for path in explicit:
        problems.extend(check_file(path, ids, schema=schema))

    for problem in problems:
        err.write(problem + "\n")
    out.write(f"checked {len(explicit)} persona file(s), {len(problems)} problem(s)\n")
    return 1 if problems else 0
