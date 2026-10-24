from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from moodsona import bench
from moodsona.cli import main
from moodsona.llm.base import ChatMessage, LLMConfigurationError, LLMError
from moodsona.llm.fake import FakeProvider, plumbing_reply
from moodsona.metrics import TranscriptMetrics, compute_metrics, metrics_from_jsonl
from moodsona.personas import list_bundled_personas, load_bundled_persona
from moodsona.runner import Transcript
from moodsona.state import PersonaSeed


def _run(
    tmp_path: Path, model: str, persona_id: str, *, days: int = 1, turns: int = 2
) -> bench.BenchRun:
    persona = load_bundled_persona(persona_id)
    return bench.run_one(
        persona, FakeProvider(), model=model, out_dir=tmp_path, days=days, turns_per_day=turns
    )


def _raising(error: BaseException) -> FakeProvider:
    """A provider whose every call raises ``error``."""

    def fail(messages: Sequence[ChatMessage]) -> str:
        raise error

    return FakeProvider(fail)


def _forbid_make_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(name: str, model: str | None) -> FakeProvider:
        raise AssertionError("no provider may be created")

    monkeypatch.setattr(bench, "make_provider", forbidden)


def test_run_one_writes_a_readable_transcript(tmp_path: Path) -> None:
    persona = load_bundled_persona("anxious-newcomer")
    result = bench.run_one(
        persona, FakeProvider(), model="fake", out_dir=tmp_path, days=2, turns_per_day=2
    )
    assert result.path == tmp_path / "fake" / "anxious-newcomer-seed0.jsonl"
    transcript = Transcript.from_jsonl(result.path)
    assert transcript.persona_id == "anxious-newcomer"
    assert result.metrics == metrics_from_jsonl(result.path)
    assert result.metrics.turns == 4


def test_the_bench_companion_is_fixed_and_cycles() -> None:
    companion = bench.bench_companion()
    replies = [companion.reply([ChatMessage("user", "hi")]) for _ in range(20)]
    assert replies[: len(bench.BENCH_COMPANION_LINES)] == list(bench.BENCH_COMPANION_LINES)
    assert replies[len(bench.BENCH_COMPANION_LINES)] == bench.BENCH_COMPANION_LINES[0]


def test_write_report_compares_models(tmp_path: Path) -> None:
    for model in ("model-a", "model-b"):
        for persona_id in ("anxious-newcomer", "boundary-tester"):
            _run(tmp_path, model, persona_id)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    path = bench.write_report(tmp_path)
    assert path == tmp_path / "bench.md"
    text = path.read_text(encoding="utf-8")
    summary = [line for line in text.splitlines() if line.startswith("| model-")]
    assert [line.split(" | ")[0] for line in summary[:2]] == ["| model-a", "| model-b"]
    assert "| model-a | 1 x 2 | 2 |" in summary[0]  # two runs of 1 day x 2 turns
    assert text.count("| model-a | anxious-newcomer |") == 1
    assert "## Per run" in text
    header = next(
        line for line in text.splitlines() if line.startswith("| model | days x turns | runs")
    )
    assert len(header.split("|")) == len(summary[0].split("|"))


def test_write_report_needs_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no bench runs"):
        bench.write_report(tmp_path)


@given(st.lists(st.one_of(st.none(), st.floats(min_value=-100, max_value=100)), max_size=10))
def test_mean_of_runs_ignores_missing_values(values: list[float | None]) -> None:
    present = [v for v in values if v is not None]
    result = bench._mean_or_none(values)
    if not present:
        assert result is None
    else:
        assert result is not None
        assert min(present) - 1e-9 <= result <= max(present) + 1e-9


@pytest.mark.parametrize(
    ("raw", "label"),
    [("claude-sonnet-5-5", "claude-sonnet-5-5"), ("org/model:v1", "org-model-v1"), ("..", "model")],
)
def test_labels_are_safe_directory_names(raw: str, label: str) -> None:
    assert bench.safe_label(raw) == label


# ------------------------------------------------------------------------ cli


def test_bench_cli_with_the_fake_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    args = ["bench", "--provider", "fake", "--persona", "anxious-newcomer"]
    args += ["--persona", "boundary-tester", "--days", "1", "--turns", "2", "--out", str(out)]
    assert main(args) == 0
    assert sorted(p.name for p in (out / "fake").iterdir()) == [
        "anxious-newcomer-seed0.jsonl",
        "boundary-tester-seed0.jsonl",
    ]
    assert (out / "bench.md").is_file()
    printed = capsys.readouterr().out
    assert "fake / anxious-newcomer: 2 turn(s)" in printed
    assert str(out / "bench.md") in printed


def test_bench_cli_runs_every_bundled_persona_by_default(tmp_path: Path) -> None:
    assert main(["bench", "--days", "1", "--turns", "1", "--out", str(tmp_path)]) == 0
    assert len(list((tmp_path / "fake").glob("*.jsonl"))) == len(list_bundled_personas())


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_paid_providers_need_yes_and_never_start_without_it(
    provider: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_make_provider(monkeypatch)
    assert main(["bench", "--provider", provider, "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "costs money" in err
    calls = bench.max_model_calls(len(list_bundled_personas()), 3, 6)  # the defaults
    assert f"up to {calls} model calls" in err
    assert not any(tmp_path.iterdir())


class Recorder(FakeProvider):
    model = "test-model/v1"


def test_paid_provider_with_yes_is_labelled_by_its_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[tuple[str, str | None]] = []

    def fake_factory(name: str, model: str | None) -> Recorder:
        made.append((name, model))
        return Recorder()

    monkeypatch.setattr(bench, "make_provider", fake_factory)
    args = ["bench", "--provider", "openai", "--model", "test-model/v1", "--yes"]
    args += ["--persona", "anxious-newcomer", "--days", "1", "--turns", "1", "--out", str(tmp_path)]
    assert main(args) == 0
    assert made == [("openai", "test-model/v1")]
    assert (tmp_path / "openai-test-model-v1" / "anxious-newcomer-seed0.jsonl").is_file()


def test_make_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(bench.make_provider("fake", None), FakeProvider)
    # Stand-ins, so no SDK is imported and nothing can reach the network.
    made: list[tuple[str, str | None]] = []

    def stand_in(name: str) -> Callable[[str | None], FakeProvider]:
        def create(model: str | None) -> FakeProvider:
            made.append((name, model))
            return FakeProvider()

        return create

    monkeypatch.setattr(bench, "AnthropicProvider", stand_in("anthropic"))
    monkeypatch.setattr(bench, "OpenAIProvider", stand_in("openai"))
    bench.make_provider("anthropic", "m1")
    bench.make_provider("openai", None)
    assert made == [("anthropic", "m1"), ("openai", None)]
    with pytest.raises(ValueError, match="unknown provider"):
        bench.make_provider("nope", None)


def test_bench_cli_fails_cleanly_on_a_broken_install(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> Path:
        raise FileNotFoundError("bundled personas directory not found")

    monkeypatch.setattr("moodsona.personas.bundled_personas_dir", missing)
    assert main(["bench", "--out", str(tmp_path)]) == 1
    assert "bundled personas directory not found" in capsys.readouterr().err


def test_bench_cli_checks_earlier_results_before_spending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "other-model").mkdir()
    (tmp_path / "other-model" / "broken.jsonl").write_text("nope\n", encoding="utf-8")

    no_calls = _raising(AssertionError("no model call while the results are unreadable"))
    monkeypatch.setattr(bench, "make_provider", lambda name, model: no_calls)
    args = ["bench", "--provider", "openai", "--yes", "--persona", "anxious-newcomer"]
    assert main([*args, "--out", str(tmp_path)]) == 1
    assert "broken.jsonl:1" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other-model"]


def test_run_one_creates_its_folder_before_the_first_model_call(tmp_path: Path) -> None:
    def checking(messages: Sequence[ChatMessage]) -> str:
        assert (tmp_path / "fake").is_dir()
        return plumbing_reply(messages)

    persona = load_bundled_persona("anxious-newcomer")
    provider = FakeProvider(checking)
    bench.run_one(persona, provider, model="fake", out_dir=tmp_path, days=1, turns_per_day=1)


def test_run_one_scores_in_the_persona_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    languages: list[object] = []

    def recording(transcript: Transcript, **kwargs: Any) -> TranscriptMetrics:
        languages.append(kwargs.get("language"))
        return compute_metrics(transcript, **kwargs)

    monkeypatch.setattr("moodsona.bench.compute_metrics", recording)
    persona = load_bundled_persona("anxious-newcomer")
    bench.run_one(persona, FakeProvider(), model="m", out_dir=tmp_path, days=1, turns_per_day=1)
    assert languages == [persona.language]


def test_write_report_keeps_different_settings_apart(tmp_path: Path) -> None:
    for persona_id, days in (("anxious-newcomer", 1), ("boundary-tester", 2)):
        _run(tmp_path, "model-a", persona_id, days=days, turns=1)
    text = bench.write_report(tmp_path).read_text(encoding="utf-8")
    assert "| model-a | 1 x 1 | 1 |" in text
    assert "| model-a | 2 x 1 | 1 |" in text
    assert "| model-a | anxious-newcomer | 1 x 1 | 0 |" in text


def test_write_report_uses_the_runs_it_is_given(tmp_path: Path) -> None:
    run = _run(tmp_path, "model-a", "anxious-newcomer")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "broken.jsonl").write_text("not read\n", encoding="utf-8")
    text = bench.write_report(tmp_path, [run]).read_text(encoding="utf-8")
    assert "| model-a | anxious-newcomer |" in text


def test_run_bench_keeps_and_reports_the_finished_runs_when_one_fails(tmp_path: Path) -> None:
    personas = bench.load_personas(["anxious-newcomer"])
    kwargs: dict[str, Any] = {"model": "m", "out_dir": tmp_path, "days": 1, "turns_per_day": 1}
    with pytest.raises(bench.BenchRunError, match="m / anxious-newcomer: quota") as info:
        bench.run_bench(personas, _raising(LLMError("quota exceeded")), **kwargs)
    assert info.value.report is None  # nothing finished, so there is nothing to report
    assert isinstance(info.value.__cause__, LLMError)
    _run(tmp_path, "m", "boundary-tester")
    with pytest.raises(bench.BenchRunError) as info:
        bench.run_bench(personas, _raising(LLMError("quota exceeded")), **kwargs)
    assert info.value.report == tmp_path / "bench.md"
    assert "boundary-tester" in info.value.report.read_text(encoding="utf-8")


def test_model_labels() -> None:
    assert bench.model_label("fake", FakeProvider()) == "fake"
    assert bench.model_label("openai", Recorder()) == "openai-test-model-v1"


def _failing_after(first: int, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Let the first ``first`` bench runs succeed and fail the rest; returns the persona ids."""
    started: list[str] = []
    real = bench.run_one

    def run_one(persona: PersonaSeed, *args: Any, **kwargs: Any) -> bench.BenchRun:
        started.append(persona.id)
        if len(started) > first:
            raise LLMError("quota exceeded")
        return real(persona, *args, **kwargs)

    monkeypatch.setattr(bench, "run_one", run_one)
    return started


def test_a_failed_run_still_reports_the_finished_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _failing_after(1, monkeypatch)
    args = ["bench", "--persona", "anxious-newcomer", "--persona", "boundary-tester"]
    args += ["--persona", "grieving-after-loss", "--days", "1", "--turns", "1"]
    assert main([*args, "--out", str(tmp_path)]) == 1
    assert started == ["anxious-newcomer", "boundary-tester"]  # stops at the first failure
    err = capsys.readouterr().err
    assert "fake / boundary-tester: quota exceeded" in err
    assert "--skip-existing" in err
    assert "| fake | anxious-newcomer |" in (tmp_path / "bench.md").read_text(encoding="utf-8")


def test_skip_existing_resumes_only_runs_with_the_same_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = ["--out", str(tmp_path), "--turns", "1"]
    assert main(["bench", "--persona", "anxious-newcomer", "--days", "1", *out]) == 0
    started = _failing_after(10, monkeypatch)
    personas = ["--persona", "anxious-newcomer", "--persona", "boundary-tester"]
    assert main(["bench", *personas, "--days", "1", "--skip-existing", *out]) == 0
    assert started == ["boundary-tester"]
    assert "fake / anxious-newcomer: kept" in capsys.readouterr().out
    assert main(["bench", *personas, "--days", "2", "--skip-existing", *out]) == 0
    assert started == ["boundary-tester", "anxious-newcomer", "boundary-tester"]


def test_repeated_personas_run_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["bench", "--persona", "anxious-newcomer", "--persona", "anxious-newcomer"]
    assert main([*args, "--days", "1", "--turns", "1", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("fake / anxious-newcomer:") == 1


def test_bench_cli_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["bench", "--persona", "no-such", "--out", str(tmp_path)]) == 2
    assert "no bundled persona 'no-such'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        main(["bench", "--days", "0"])
    assert info.value.code == 2

    broken = _raising(LLMError("quota exceeded"))
    monkeypatch.setattr(bench, "make_provider", lambda name, model: broken)
    args = ["bench", "--persona", "anxious-newcomer", "--days", "1", "--out", str(tmp_path)]
    assert main(args) == 1
    assert "quota exceeded" in capsys.readouterr().err


def test_bench_cli_reports_a_missing_sdk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str, model: str | None) -> FakeProvider:
        raise LLMConfigurationError("The openai SDK is not installed")

    monkeypatch.setattr(bench, "make_provider", missing)
    assert main(["bench", "--provider", "openai", "--yes", "--out", str(tmp_path)]) == 1
    assert "SDK is not installed" in capsys.readouterr().err
