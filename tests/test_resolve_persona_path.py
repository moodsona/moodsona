from __future__ import annotations

from pathlib import Path

import pytest

from moodsona.personas import bundled_persona_path, resolve_persona_path

EXAMPLE = Path(__file__).resolve().parents[1] / "personas" / "examples" / "anxious-newcomer.yaml"


def test_resolve_persona_path_finds_a_bundled_id() -> None:
    assert resolve_persona_path("anxious-newcomer") == bundled_persona_path("anxious-newcomer")


def test_resolve_persona_path_keeps_paths(tmp_path: Path) -> None:
    assert resolve_persona_path(str(EXAMPLE)) == EXAMPLE
    assert resolve_persona_path("missing.yaml") == Path("missing.yaml")  # a YAML suffix
    assert resolve_persona_path("sub/missing") == Path("sub/missing")  # a directory part


def test_resolve_persona_path_prefers_an_existing_file_over_an_id(tmp_path: Path) -> None:
    (tmp_path / "anxious-newcomer").write_text("x", encoding="utf-8")
    found = resolve_persona_path("anxious-newcomer", base=tmp_path)
    assert found == tmp_path / "anxious-newcomer"


def test_resolve_persona_path_reads_relative_paths_from_base(tmp_path: Path) -> None:
    assert resolve_persona_path("p/a.yaml", base=tmp_path) == tmp_path / "p" / "a.yaml"
    assert resolve_persona_path(str(EXAMPLE), base=tmp_path) == EXAMPLE  # absolute stays
    assert resolve_persona_path("lonely-night-owl", base=tmp_path) == bundled_persona_path(
        "lonely-night-owl"
    )


def test_resolve_persona_path_rejects_an_unknown_bare_id() -> None:
    with pytest.raises(KeyError, match="no bundled persona 'no-such-persona'"):
        resolve_persona_path("no-such-persona")
