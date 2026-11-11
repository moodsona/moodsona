from __future__ import annotations

import copy
import importlib.resources
import io
from pathlib import Path
from typing import Any

import pytest
import yaml

from moodsona import personas, validation
from moodsona.companion import EchoCompanion
from moodsona.llm.fake import FakeProvider
from moodsona.personas import (
    PersonaError,
    bundled_persona_path,
    bundled_personas_dir,
    list_bundled_personas,
    load_bundled_persona,
    load_persona,
    load_schema,
    persona_from_mapping,
)
from moodsona.runner import run
from moodsona.schema import UnsupportedSchemaError, validate
from moodsona.state import Mood, Stance

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "personas" / "examples" / "anxious-newcomer.yaml"
COMMUNITY_DIR = REPO / "personas" / "community"
COMMUNITY_FILES = list(validation.iter_files([COMMUNITY_DIR]))

# A synthetic persona written for these tests.
VALID: dict[str, Any] = {
    "id": "test-night-baker",
    "display_name": "Night baker",
    "language": "en",
    "pronouns": "she/her",
    "hidden_goal": "Wants company during the quiet hours before the bakery opens.",
    "reaction_rules": [
        {
            "when": "The bot asks about bread.",
            "then": "Talks happily about sourdough.",
            "emotion_delta": 5,
            "stance_to": "open",
        },
        {"when": "The bot is curt.", "then": "Keeps it short.", "energy_delta": -5},
    ],
    "initial_state": {"emotion": 60, "energy": 70, "stance": "curious"},
    "baseline": {"emotion": 55, "energy": 65, "stance": "curious"},
    "memory_days_visible": 4,
    "tags": ["night-shift"],
    "license": "CC-BY-4.0",
    "author": "moodsona tests",
    "ai_assisted": True,
}


def test_example_persona_loads() -> None:
    seed = load_persona(EXAMPLE)
    assert seed.id == "anxious-newcomer"
    assert seed.pronouns.label == "they/them"  # optional field, default
    assert seed.initial_state == Mood(35, 40, Stance.GUARDED)
    assert seed.reaction_rules[0].energy_delta == -10
    assert seed.reaction_rules[0].emotion_delta is None
    assert seed.reaction_rules[1].stance_to == Stance.OPEN
    assert "  " not in seed.description


def test_mapping_fields_reach_the_seed() -> None:
    seed = persona_from_mapping(VALID)
    assert seed.pronouns.possessive == "her"
    assert seed.memory_days_visible == 4
    assert seed.tags == ("night-shift",)
    assert seed.description == ""


def test_bundled_personas() -> None:
    assert (bundled_personas_dir() / "schema.json").is_file()
    ids = list_bundled_personas()
    assert bundled_persona_path("anxious-newcomer").name == "anxious-newcomer.yaml"
    with pytest.raises(KeyError, match="no bundled persona"):
        bundled_persona_path("does-not-exist")
    assert "anxious-newcomer" in ids
    assert ids == sorted(ids)
    for persona_id in ids:
        assert load_bundled_persona(persona_id).id == persona_id
    with pytest.raises(KeyError, match="no bundled persona"):
        load_bundled_persona("does-not-exist")


def test_bundled_dir_falls_back_to_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.resources, "files", lambda _: Path("/nonexistent"))
    assert bundled_personas_dir() == REPO / "personas"


def test_bundled_dir_prefers_the_packaged_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "personas").mkdir()
    monkeypatch.setattr(importlib.resources, "files", lambda _: tmp_path)
    assert bundled_personas_dir() == tmp_path / "personas"


def test_missing_bundled_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(importlib.resources, "files", lambda _: tmp_path)
    monkeypatch.setattr(personas, "__file__", str(tmp_path / "a" / "b" / "personas.py"))
    with pytest.raises(FileNotFoundError):
        bundled_personas_dir()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"language": "fr"}, "language"),
        ({"pronouns": "xe/xem"}, "pronouns"),
        ({"extra": 1}, "unknown field 'extra'"),
        ({"id": "Not Kebab"}, "does not match"),
        ({"initial_state": {"emotion": 101, "energy": 5, "stance": "open"}}, "maximum"),
        ({"reaction_rules": []}, "at least 1"),
        ({"tags": ["a", "a"]}, "unique"),
        ({"license": "MIT"}, "must be 'CC-BY-4.0'"),
        ({"ai_assisted": "yes"}, "expected boolean"),
    ],
)
def test_invalid_personas_are_rejected(change: dict[str, Any], message: str) -> None:
    data = {**copy.deepcopy(VALID), **change}
    with pytest.raises(PersonaError, match=message) as info:
        persona_from_mapping(data, source="test.yaml")
    assert info.value.problems


def test_missing_required_field() -> None:
    data = copy.deepcopy(VALID)
    del data["hidden_goal"]
    with pytest.raises(PersonaError, match="missing required field 'hidden_goal'"):
        persona_from_mapping(data)


def test_unreadable_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(PersonaError, match="cannot read YAML"):
        load_persona(bad)
    with pytest.raises(PersonaError, match="cannot read YAML"):
        load_persona(tmp_path / "missing.yaml")
    listing = tmp_path / "list.yaml"
    listing.write_text("- just a list", encoding="utf-8")
    with pytest.raises(PersonaError, match="expected object"):
        load_persona(listing)


def test_yaml_is_loaded_safely(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("id: !!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")
    with pytest.raises(PersonaError, match="cannot read YAML"):
        load_persona(unsafe)


# -- the built-in validator agrees with the reference implementation ---------------


def _mutations() -> list[Any]:
    cases: list[Any] = [copy.deepcopy(VALID), yaml.safe_load(EXAMPLE.read_text("utf-8")), [], 3]
    for key in VALID:
        dropped = copy.deepcopy(VALID)
        del dropped[key]
        cases.append(dropped)
    odd_values: list[Any] = [None, True, 0, 2.0, -1, 500, "", "x" * 700, [], {}, ["a", "a"]]
    for key in VALID:
        for value in odd_values:
            cases.append({**copy.deepcopy(VALID), key: value})
    for key in ("emotion", "energy", "stance"):
        for value in odd_values:
            changed = copy.deepcopy(VALID)
            changed["baseline"][key] = value
            cases.append(changed)
    for key in ("when", "then", "emotion_delta", "energy_delta", "stance_to", "other"):
        for value in odd_values:
            changed = copy.deepcopy(VALID)
            changed["reaction_rules"][0][key] = value
            cases.append(changed)
    return cases


def test_builtin_validator_agrees_with_jsonschema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = load_schema()
    reference = jsonschema.Draft202012Validator(schema)
    reference.check_schema(schema)
    for case in _mutations():
        assert (validate(case, schema) == []) == reference.is_valid(case), case


def test_unsupported_schema_keywords_fail_loudly() -> None:
    with pytest.raises(UnsupportedSchemaError, match="oneOf"):
        validate({}, {"oneOf": []})
    with pytest.raises(UnsupportedSchemaError, match="local"):
        validate({}, {"$ref": "https://example.org/schema"})
    with pytest.raises(UnsupportedSchemaError, match="type"):
        validate({}, {"type": ["string", "null"]})
    with pytest.raises(UnsupportedSchemaError, match="additionalProperties"):
        validate({"a": 1}, {"additionalProperties": {"type": "string"}})
    assert validate(1.0, {"enum": [1]}) == []
    assert validate(True, {"enum": [1]}) != []


# -- the community personas ---------------------------------------------------------


def test_community_personas_exist() -> None:
    # The parametrized tests below would be skipped, not failed, on an empty directory.
    assert COMMUNITY_FILES, f"no persona files in {COMMUNITY_DIR}"


def test_community_personas_pass_the_validator() -> None:
    out, err = io.StringIO(), io.StringIO()
    # The same checks as `moodsona validate`, with ids unique across the whole tree;
    # each problem line names its file.
    code = validation.main([str(COMMUNITY_DIR)], out=out, err=err, root=REPO / "personas")
    assert code == 0, err.getvalue()
    assert f"checked {len(COMMUNITY_FILES)} persona file(s)" in out.getvalue()


@pytest.mark.parametrize("path", COMMUNITY_FILES, ids=lambda path: path.stem)
def test_community_persona_is_bundled_and_runs_offline(path: Path) -> None:
    seed = load_persona(path)
    # The id lookup reaches the community/ subdirectory (the wheel ships all of personas/).
    assert seed.id in list_bundled_personas()
    transcript = run(seed, FakeProvider(), EchoCompanion(), days=2, turns_per_day=3)
    assert transcript.persona_id == seed.id
    assert {point.day for point in transcript.trajectory()} == {1, 2}
