from __future__ import annotations

from pathlib import Path

import pytest

from moodsona import bench
from moodsona.cli import main
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import compute_metrics
from moodsona.runner import Transcript
from support import forbid_make_provider

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "personas" / "examples" / "anxious-newcomer.yaml"


def _run(*args: str) -> list[str]:
    return ["run", *args]


def test_run_with_a_bundled_id_writes_the_transcript_and_prints_table_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no persona files here: the id is looked up
    args = _run("anxious-newcomer", "--days", "2", "--turns", "3", "--seed", "4")
    assert main(args) == 0
    out = capsys.readouterr().out
    transcript = Transcript.from_jsonl(tmp_path / "run.jsonl")  # the default --out
    assert transcript.persona_id == "anxious-newcomer"
    assert (transcript.days, transcript.turns_per_day, transcript.rng_seed) == (2, 3, 4)
    lines = out.splitlines()
    assert lines[0] == (
        f"persona anxious-newcomer: {len(transcript.turns)} turn(s), model fake, companion echo"
    )
    assert lines[1].split() == ["day", "turn", "emotion", "energy", "stance"]
    for i, point in enumerate(transcript.trajectory(), start=2):
        assert lines[i].split() == [
            str(point.day),
            str(point.turn),
            str(point.emotion),
            str(point.energy),
            point.stance,
        ]
    assert "wrote run.jsonl" in lines
    assert "persona anxious-newcomer (seed 4):" in out
    assert "drift/day" in out
    assert "polluted" in out


def test_run_summary_matches_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "sub" / "dir" / "out.jsonl"  # missing folders are created
    assert main(_run(str(EXAMPLE), "--days", "2", "--turns", "2", "--out", str(path))) == 0
    out = capsys.readouterr().out
    assert main(["report", str(path)]) == 0
    report = capsys.readouterr().out
    assert out.endswith(report)
    assert f"wrote {path}\n" in out


def test_run_with_a_script_companion_cycles_and_skips_blank_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "bot.txt"
    # A byte order mark (some Windows editors write one) is not part of the first reply.
    script.write_bytes(b"\xef\xbb\xbfFirst reply.\n\n   \nSecond reply.\r\n")
    out = tmp_path / "run.jsonl"
    args = _run(str(EXAMPLE), "--companion", f"script:{script}", "--days", "1", "--turns", "5")
    assert main([*args, "--out", str(out)]) == 0
    assert f"companion script:{script}" in capsys.readouterr().out
    replies = [t.companion_reply for t in Transcript.from_jsonl(out).turns]
    assert len(replies) >= 3
    expected = ["First reply.", "Second reply."] * 3
    assert replies == expected[: len(replies)]


def test_run_scores_in_the_persona_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[object] = []

    def recording(transcript: Transcript, *, language: object) -> object:  # must be passed
        seen.append(language)
        return compute_metrics(transcript, language="en")

    monkeypatch.setattr("moodsona.cli.compute_metrics", recording)
    out = str(tmp_path / "r.jsonl")
    assert main(_run(str(EXAMPLE), "--days", "1", "--turns", "1", "--out", out)) == 0
    assert seen == ["en"]


@pytest.mark.parametrize(
    "spec", ["nope", "http://127.0.0.1:9/chat", "script:", "echo:loud", "Echo"]
)
def test_unknown_companion_specs_are_usage_errors(
    spec: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_make_provider(monkeypatch)
    out = tmp_path / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--companion", spec, "--out", str(out))) == 2
    err = capsys.readouterr().err
    assert "echo" in err
    assert "script:<file>" in err
    assert not out.exists()


def test_a_missing_or_empty_script_is_a_usage_error_before_any_model_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    out = tmp_path / "run.jsonl"
    missing = tmp_path / "missing.txt"
    assert main(_run(str(EXAMPLE), "--companion", f"script:{missing}", "--out", str(out))) == 2
    assert "missing.txt" in capsys.readouterr().err
    empty = tmp_path / "empty.txt"
    empty.write_text("\n  \n\n", encoding="utf-8")
    assert main(_run(str(EXAMPLE), "--companion", f"script:{empty}", "--out", str(out))) == 2
    assert "no replies" in capsys.readouterr().err
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00bad")
    assert main(_run(str(EXAMPLE), "--companion", f"script:{binary}", "--out", str(out))) == 2
    assert "binary.txt" in capsys.readouterr().err
    assert not out.exists()


def test_run_fails_cleanly_on_a_bad_persona(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert main(_run("no-such-persona")) == 1
    err = capsys.readouterr().err
    assert "no-such-persona: no such file" in err
    assert "anxious-newcomer" in err
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: bad\n", encoding="utf-8")
    assert main(_run(str(bad))) == 1
    assert "missing required field" in capsys.readouterr().err
    assert not (tmp_path / "run.jsonl").exists()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_days_and_turns_must_be_positive(value: str) -> None:
    for flag in ("--days", "--turns"):
        with pytest.raises(SystemExit) as info:
            main(_run(str(EXAMPLE), flag, value))
        assert info.value.code == 2


def test_out_must_not_be_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    assert main(_run(str(EXAMPLE), "--out", str(tmp_path))) == 2
    assert "is a directory" in capsys.readouterr().err


def test_an_unwritable_out_is_a_usage_error_before_any_model_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    blocker = tmp_path / "file"
    blocker.write_text("not a folder", encoding="utf-8")
    assert main(_run(str(EXAMPLE), "--out", str(blocker / "run.jsonl"))) == 2
    assert "cannot write" in capsys.readouterr().err


def test_an_existing_out_is_kept_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text("an earlier (maybe paid) run\n", encoding="utf-8")
    short = ["--turns", "1", "--days", "1", "--out", str(out)]
    forbid_make_provider(monkeypatch)
    assert main(_run(str(EXAMPLE), *short)) == 2  # a fake run too
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "an earlier (maybe paid) run\n"
    monkeypatch.undo()
    assert main(_run(str(EXAMPLE), *short, "--force")) == 0
    assert Transcript.from_jsonl(out).persona_id == "anxious-newcomer"


def test_out_never_replaces_the_input_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    persona = tmp_path / "anxious-newcomer.yaml"
    persona.write_bytes(EXAMPLE.read_bytes())
    script = tmp_path / "replies.txt"
    script.write_text("Hello.\n", encoding="utf-8")
    assert main(_run(str(persona), "--out", str(persona))) == 2
    assert main(_run(str(persona), "--companion", f"script:{script}", "--out", str(script))) == 2
    assert persona.read_bytes() == EXAMPLE.read_bytes()
    assert script.read_text(encoding="utf-8") == "Hello.\n"


def test_a_dangling_symlink_counts_as_an_existing_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    out = tmp_path / "run.jsonl"
    try:
        out.symlink_to(tmp_path / "gone.jsonl")
    except OSError:  # Windows without the symlink privilege
        pytest.skip("cannot create symlinks here")
    assert main(_run(str(EXAMPLE), "--out", str(out))) == 2
    assert "already exists" in capsys.readouterr().err


def test_a_run_leaves_only_the_transcript(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--days", "1", "--turns", "1", "--out", str(out))) == 0
    assert list(tmp_path.iterdir()) == [out]  # no temp file, no probe file


def test_a_failed_run_creates_no_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "make_provider", lambda name, model: FakeProvider([]))
    out = tmp_path / "new" / "sub" / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--out", str(out))) == 1
    assert not any(tmp_path.iterdir())


def test_a_failed_replace_keeps_the_old_out_and_the_new_transcript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text("an earlier paid run\n", encoding="utf-8")

    def refuse(src: object, dst: object) -> None:
        raise PermissionError("the file is in use")

    monkeypatch.setattr("moodsona.cli.os.replace", refuse)
    args = _run(str(EXAMPLE), "--force", "--days", "1", "--turns", "1", "--out", str(out))
    assert main(args) == 1
    monkeypatch.undo()
    err = capsys.readouterr().err
    assert "the file is in use" in err
    assert out.read_text(encoding="utf-8") == "an earlier paid run\n"
    (kept,) = [p for p in tmp_path.iterdir() if p != out]
    assert str(kept) in err
    assert Transcript.from_jsonl(kept).persona_id == "anxious-newcomer"


def test_a_temp_file_that_cannot_be_created_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_temp(**kwargs: object) -> None:
        raise PermissionError("access denied")

    monkeypatch.setattr("moodsona.cli.tempfile.NamedTemporaryFile", no_temp)
    out = tmp_path / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--days", "1", "--turns", "1", "--out", str(out))) == 1
    monkeypatch.undo()
    assert "cannot write the transcript" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_a_failed_write_keeps_the_old_out_and_leaves_no_temp_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text("an earlier paid run\n", encoding="utf-8")

    def disk_full(self: Transcript, path: object) -> None:
        Path(str(path)).write_text('{"kind": "ru', encoding="utf-8")  # a partial write
        raise OSError("No space left on device")

    monkeypatch.setattr(Transcript, "to_jsonl", disk_full)
    args = _run(str(EXAMPLE), "--force", "--days", "1", "--turns", "1", "--out", str(out))
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "cannot write the transcript" in err
    assert "No space left on device" in err
    assert out.read_text(encoding="utf-8") == "an earlier paid run\n"
    assert list(tmp_path.iterdir()) == [out]


def test_a_failed_forced_run_keeps_the_old_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "make_provider", lambda name, model: FakeProvider([]))
    out = tmp_path / "run.jsonl"
    out.write_text("keep\n", encoding="utf-8")
    assert main(_run(str(EXAMPLE), "--force", "--out", str(out))) == 1
    assert out.read_text(encoding="utf-8") == "keep\n"


def test_model_without_a_real_provider_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    out = tmp_path / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--model", "some-model", "--yes", "--out", str(out))) == 2
    assert "--model needs --provider" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_paid_providers_need_yes_and_never_start_without_it(
    provider: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_make_provider(monkeypatch)
    out = tmp_path / "new" / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--provider", provider, "--out", str(out))) == 2
    err = capsys.readouterr().err
    assert "costs money" in err
    calls = bench.max_model_calls(1, 3, 6)  # the defaults
    assert f"up to {calls} model calls" in err
    assert "Add --yes" in err
    assert not any(tmp_path.iterdir())  # not even the folder of --out


def test_run_and_bench_share_the_cost_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    common = ["--provider", "openai", "--days", "2", "--turns", "1"]
    assert main(_run("anxious-newcomer", *common, "--out", str(tmp_path / "r.jsonl"))) == 2
    from_run = capsys.readouterr().err
    assert main(["bench", "--persona", "anxious-newcomer", *common, "--out", str(tmp_path)]) == 2
    assert capsys.readouterr().err == from_run


def test_a_paid_run_does_not_overwrite_an_existing_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_make_provider(monkeypatch)
    out = tmp_path / "run.jsonl"
    out.write_text("an earlier run\n", encoding="utf-8")
    assert main(_run(str(EXAMPLE), "--provider", "anthropic", "--yes", "--out", str(out))) == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "an earlier run\n"


class Recorder(FakeProvider):
    model = "test-model/v1"


def test_paid_provider_with_yes_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[tuple[str, str | None]] = []

    def fake_factory(name: str, model: str | None) -> Recorder:
        made.append((name, model))
        return Recorder()

    monkeypatch.setattr(bench, "make_provider", fake_factory)
    out = tmp_path / "paid.jsonl"
    args = _run("anxious-newcomer", "--provider", "openai", "--model", "test-model/v1", "--yes")
    assert main([*args, "--days", "1", "--turns", "2", "--out", str(out)]) == 0
    assert made == [("openai", "test-model/v1")]
    assert "model openai-test-model-v1" in capsys.readouterr().out
    assert Transcript.from_jsonl(out).persona_id == "anxious-newcomer"


def test_run_fails_cleanly_when_the_model_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "make_provider", lambda name, model: FakeProvider([]))
    out = tmp_path / "run.jsonl"
    assert main(_run(str(EXAMPLE), "--out", str(out))) == 1
    assert "FakeProvider script is exhausted" in capsys.readouterr().err
    assert not out.exists()
