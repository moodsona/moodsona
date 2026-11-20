"""The README's quickstart runs, and the output it shows is what the CLI prints.

The README is the first thing a visitor reads, so its examples must not drift
from the code. Each ```console block is a ``$ moodsona ...`` command followed
by its output; a trailing ``...`` line means "the rest is omitted", so only
the lines before it are compared.
"""

from __future__ import annotations

import re
import runpy
import shlex
from pathlib import Path

import pytest

from moodsona.cli import main

README = Path(__file__).resolve().parents[1] / "README.md"


def _blocks(lang: str) -> list[str]:
    text = README.read_text(encoding="utf-8")
    return re.findall(rf"^```{lang}\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)


def test_the_readme_has_a_python_quickstart_and_console_examples() -> None:
    assert len(_blocks("python")) == 1
    assert len(_blocks("console")) >= 2


def test_the_readme_quickstart_runs_and_its_console_output_is_real(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # the quickstart writes its JSONL into the current directory
    script = tmp_path / "quickstart.py"
    script.write_text(_blocks("python")[0], encoding="utf-8")
    runpy.run_path(str(script), run_name="__main__")
    capsys.readouterr()

    for block in _blocks("console"):
        command, *shown = block.rstrip("\n").splitlines()
        assert command.startswith("$ moodsona "), command
        assert main(shlex.split(command)[2:]) == 0, command
        printed = capsys.readouterr().out.splitlines()
        if shown and shown[-1] == "...":
            shown = shown[:-1]
            printed = printed[: len(shown)]
        assert printed == shown, command


def test_the_readme_has_no_relative_links_or_images() -> None:
    """PyPI renders the README away from the repository, so every target must be absolute."""
    text = README.read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)  # code is not a link
    targets = re.findall(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)", text)  # inline links and images
    targets += re.findall(r"""\b(?:src|href)=["']([^"']+)["']""", text)  # raw HTML
    targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", text, flags=re.M)  # reference definitions
    relative = [target for target in targets if not target.startswith(("http://", "https://"))]
    assert relative == []
