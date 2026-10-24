"""Load persona files and turn them into simulator seeds.

A persona file is YAML that must validate against ``personas/schema.json``,
the public contract. Files are read with ``yaml.safe_load`` only.

Bundled personas ship inside the wheel as ``moodsona/personas/`` (see
``[tool.hatch.build.targets.wheel.force-include]``); in a source checkout the
repository's top-level ``personas/`` directory is used instead.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from os import PathLike
from pathlib import Path
from typing import Any

import yaml

from moodsona.schema import validate
from moodsona.state import DEFAULT_PRONOUNS, Mood, PersonaSeed, Pronouns, ReactionRule, Stance


class PersonaError(ValueError):
    """A persona file cannot be read or does not match the schema.

    Attributes:
        problems: One message per problem found.
    """

    def __init__(self, source: str, problems: list[str]) -> None:
        super().__init__(f"{source}: " + "; ".join(problems))
        self.problems = problems


def bundled_personas_dir() -> Path:
    """The directory holding ``schema.json`` and the bundled persona YAML files.

    Raises:
        FileNotFoundError: Neither the packaged nor the source-checkout directory exists.
    """
    packaged = resources.files("moodsona") / "personas"
    if isinstance(packaged, Path) and packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "personas"
    if (checkout / "schema.json").is_file():
        return checkout
    raise FileNotFoundError("bundled personas directory not found")


@cache
def load_schema() -> dict[str, Any]:
    """The persona JSON Schema shipped with the package."""
    text = (bundled_personas_dir() / "schema.json").read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def _persona_files() -> dict[str, Path]:
    root = bundled_personas_dir()
    return {path.stem: path for path in sorted(root.rglob("*.yaml"))}


def list_bundled_personas() -> list[str]:
    """Ids of the personas shipped with the package, sorted."""
    return sorted(_persona_files())


def bundled_persona_path(persona_id: str) -> Path:
    """The file of a bundled persona.

    Raises:
        KeyError: No bundled persona has that id.
        FileNotFoundError: The bundled personas directory is missing (a broken install).
    """
    files = _persona_files()
    if persona_id not in files:
        raise KeyError(f"no bundled persona {persona_id!r}; known: {sorted(files)}")
    return files[persona_id]


def load_bundled_persona(persona_id: str) -> PersonaSeed:
    """Load a bundled persona by id.

    Raises:
        KeyError: No bundled persona has that id.
        PersonaError: The file is invalid.
    """
    return load_persona(bundled_persona_path(persona_id))


def load_persona(path: str | PathLike[str]) -> PersonaSeed:
    """Read, validate and convert one persona file.

    Args:
        path: A persona YAML file.

    Returns:
        The simulator seed.

    Raises:
        PersonaError: The file cannot be read, is not valid YAML, or does not
            match the schema.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PersonaError(str(path), [f"cannot read YAML: {exc}"]) from exc
    return persona_from_mapping(data, source=str(path))


def persona_from_mapping(data: Any, *, source: str = "persona") -> PersonaSeed:
    """Validate a loaded persona mapping against the schema and build the seed.

    Args:
        data: The parsed YAML/JSON document.
        source: Name used in error messages.

    Raises:
        PersonaError: The data does not match the schema.
    """
    problems = validate(data, load_schema())
    if problems:
        raise PersonaError(source, problems)
    return PersonaSeed(
        id=data["id"],
        display_name=data["display_name"].strip(),
        description=" ".join(str(data.get("description", "")).split()),
        language=data["language"],
        pronouns=Pronouns.from_label(data.get("pronouns", DEFAULT_PRONOUNS)),
        hidden_goal=" ".join(data["hidden_goal"].split()),
        reaction_rules=tuple(_rule(item) for item in data["reaction_rules"]),
        initial_state=_mood(data["initial_state"]),
        baseline=_mood(data["baseline"]),
        memory_days_visible=int(data["memory_days_visible"]),
        tags=tuple(data["tags"]),
    )


def _mood(data: dict[str, Any]) -> Mood:
    return Mood(int(data["emotion"]), int(data["energy"]), Stance(data["stance"]))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _rule(data: dict[str, Any]) -> ReactionRule:
    stance = data.get("stance_to")
    return ReactionRule(
        when=" ".join(data["when"].split()),
        then=" ".join(data["then"].split()),
        emotion_delta=_optional_int(data.get("emotion_delta")),
        energy_delta=_optional_int(data.get("energy_delta")),
        stance_to=Stance(stance) if stance is not None else None,
    )
