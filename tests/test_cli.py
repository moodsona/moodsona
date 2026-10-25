from __future__ import annotations

import io
import runpy
import shutil
import sys
from pathlib import Path

import pytest

from moodsona import validation
from moodsona.cli import main
from moodsona.llm.fake import FakeProvider

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "personas" / "examples" / "anxious-newcomer.yaml"


def _persona_dir(tmp_path: Path) -> Path:
    root = tmp_path / "personas" / "community"
    root.mkdir(parents=True)
    shutil.copy(EXAMPLE, root / "anxious-newcomer.yaml")
    return root


def test_validate_passes_on_the_bundled_personas(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    assert main(["validate"]) == 0
    assert "0 problem(s)" in capsys.readouterr().out


def test_validate_reports_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _persona_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    text = EXAMPLE.read_text(encoding="utf-8")
    (root / "renamed.yaml").write_text(text, encoding="utf-8")
    (root / "contact.yaml").write_text(
        text.replace("id: anxious-newcomer", "id: contact").replace(
            "Recently moved", "Mail me at someone@example.org, recently moved"
        ),
        encoding="utf-8",
    )
    (root / "dupe-key.yaml").write_text("id: dupe-key\nid: dupe-key\n", encoding="utf-8")
    (root / "scalar.yaml").write_text("just text\n", encoding="utf-8")
    (root / "schema-bad.yaml").write_text(
        text.replace("id: anxious-newcomer", "id: schema-bad").replace(
            "language: en", "language: de"
        ),
        encoding="utf-8",
    )
    assert main(["validate", str(root)]) == 1
    err = capsys.readouterr().err
    assert "must equal the file name 'renamed'" in err
    assert "is already used by" in err
    assert "looks like an e-mail address" in err
    assert "duplicate key 'id'" in err
    assert "top level must be a mapping" in err
    assert "schema: language" in err


def test_validate_registers_ids_of_files_not_passed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _persona_dir(tmp_path)
    (root / "broken.yaml").write_text("id: [", encoding="utf-8")
    new = tmp_path / "new"
    new.mkdir()
    (new / "anxious-newcomer.yaml").write_text(EXAMPLE.read_text("utf-8"), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a persona", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    args = [str(new / "anxious-newcomer.yaml"), "notes.txt"]
    assert validation.main(args, out=out, err=err, root=tmp_path / "personas") == 1
    assert "already used" in err.getvalue()
    assert "checked 1 persona file(s)" in out.getvalue()


def test_validate_skips_the_schema_when_asked(tmp_path: Path) -> None:
    path = tmp_path / "loose.yaml"
    path.write_text("id: loose\n", encoding="utf-8")
    assert validation.check_file(path, {}, schema=False) == []
    assert validation.check_file(path, {}) != []


def test_smoke_prints_the_trajectory(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["smoke", str(EXAMPLE), "--days", "2", "--turns", "2", "--seed", "4"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "persona anxious-newcomer: 4 turn(s), offline fake model"
    assert out[1].split() == ["day", "turn", "emotion", "energy", "stance"]
    assert len(out) == 6


def test_smoke_accepts_a_bundled_persona_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no persona files here, as after `pip install moodsona`
    assert main(["smoke", "anxious-newcomer", "--turns", "2"]) == 0
    assert capsys.readouterr().out.startswith("persona anxious-newcomer: 2 turn(s)")


def test_smoke_prefers_an_existing_file_over_a_bundled_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    other = REPO / "personas" / "examples" / "boundary-tester.yaml"
    (tmp_path / "anxious-newcomer").write_text(other.read_text("utf-8"), encoding="utf-8")
    assert main(["smoke", "anxious-newcomer"]) == 0
    assert capsys.readouterr().out.startswith("persona boundary-tester:")


def test_smoke_names_the_bundled_personas_when_nothing_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["smoke", "no-such-persona"]) == 1
    err = capsys.readouterr().err
    assert "no-such-persona: no such file" in err
    assert "anxious-newcomer" in err


def test_smoke_ignores_a_directory_named_like_a_bundled_persona(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "boundary-tester").mkdir()
    assert main(["smoke", "boundary-tester", "--turns", "1"]) == 0
    assert capsys.readouterr().out.startswith("persona boundary-tester:")


@pytest.mark.parametrize("arg", ["typo.yaml", "anxious-newcomer.yml", "sub/anxious-newcomer"])
def test_paths_are_never_looked_up_as_ids(
    arg: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["smoke", arg]) == 1
    err = capsys.readouterr().err
    assert "cannot read YAML" in err
    assert "known:" not in err
    assert main(["validate", arg]) == 2
    assert capsys.readouterr().err == f"{arg}: no such file or directory\n"


def test_validate_accepts_bundled_persona_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["validate", "anxious-newcomer", "boundary-tester"]) == 0
    assert "checked 2 persona file(s), 0 problem(s)" in capsys.readouterr().out
    assert main(["validate", "no-such-persona"]) == 2
    err = capsys.readouterr().err
    assert "no-such-persona: no such file or directory" in err
    assert "anxious-newcomer" in err  # the known ids are listed


def test_bundled_names_fail_cleanly_on_a_broken_install(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> Path:
        raise FileNotFoundError("bundled personas directory not found")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("moodsona.personas.bundled_personas_dir", missing)
    assert main(["smoke", "anxious-newcomer"]) == 1
    assert "bundled personas directory not found" in capsys.readouterr().err
    assert main(["validate", "anxious-newcomer"]) == 2
    assert "bundled personas directory not found" in capsys.readouterr().err


def test_smoke_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: bad\n", encoding="utf-8")
    assert main(["smoke", str(bad)]) == 1
    assert main(["smoke", str(EXAMPLE), "--days", "0"]) == 1
    with pytest.raises(SystemExit) as info:
        main(["unknown"])
    assert info.value.code == 2
    assert "missing required field" in capsys.readouterr().err


def test_smoke_fails_cleanly_without_a_language_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_profile(code: str) -> None:
        raise LookupError(f"No language profile for {code!r}.")

    monkeypatch.setattr("moodsona.simulator.get_profile", no_profile)
    assert main(["smoke", str(EXAMPLE)]) == 1
    assert "No language profile for 'en'" in capsys.readouterr().err


def test_tool_wrapper_runs_the_package_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["validate_personas.py", str(EXAMPLE)])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(REPO / "tools" / "validate_personas.py"), run_name="__main__")
    assert info.value.code == 0


def test_validate_default_does_not_depend_on_the_working_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 0
    count = len(list((REPO / "personas").rglob("*.yaml")))
    assert count > 0
    assert f"checked {count} persona file(s), 0 problem(s)" in capsys.readouterr().out


def test_validate_with_nothing_to_check_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["validate", str(tmp_path / "typo.yaml")]) == 2
    assert "typo.yaml: no such file or directory" in capsys.readouterr().err
    assert main(["validate", str(tmp_path)]) == 2  # an existing directory without personas
    assert "no persona files to check" in capsys.readouterr().err

    def missing() -> Path:
        raise FileNotFoundError("bundled personas directory not found")

    monkeypatch.setattr(validation, "bundled_personas_dir", missing)
    assert main(["validate"]) == 2
    assert "no persona files to check" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Worked night shifts 2019-2024 (5 years), then quit.", False),
        ("Moved twice, in 1998 - 2003 and in 2011.", False),
        ("Has 3 cats and 2 dogs.", False),
        ("Call +1 (555) 123-4567 any time.", True),
        ("my number is 555-123-4567", True),
        ("(555) 123 4567", True),
    ],
)
def test_phone_number_detection(text: str, found: bool) -> None:
    assert validation.looks_like_phone(text) is found


def test_check_file_reports_phone_numbers_but_not_year_ranges(tmp_path: Path) -> None:
    path = tmp_path / "caller.yaml"
    path.write_text(
        "id: caller\n"
        "description: Nurse 2019-2024 (5 years).\n"
        "hidden_goal: Text me on 555-123-4567.\n",
        encoding="utf-8",
    )
    assert validation.check_file(path, {}, schema=False) == [
        f"{path}: hidden_goal contains what looks like a phone number"
    ]


def test_smoke_fails_cleanly_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("moodsona.cli.FakeProvider", lambda: FakeProvider([]))
    assert main(["smoke", str(EXAMPLE)]) == 1
    assert "FakeProvider script is exhausted" in capsys.readouterr().err


def test_smoke_fails_cleanly_on_a_broken_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing() -> dict[str, object]:
        raise FileNotFoundError("bundled personas directory not found")

    monkeypatch.setattr("moodsona.personas.load_schema", missing)
    assert main(["smoke", str(EXAMPLE)]) == 1
    assert "bundled personas directory not found" in capsys.readouterr().err
