from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from moodsona import bench
from moodsona.cli import main
from moodsona.llm.base import ChatMessage, LLMConfigurationError, LLMError, LLMResponse
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import metrics_from_jsonl
from moodsona.personas import list_bundled_personas, load_bundled_persona
from moodsona.runner import Transcript


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
            bench.run_one(
                load_bundled_persona(persona_id),
                FakeProvider(),
                model=model,
                out_dir=tmp_path,
                days=1,
                turns_per_day=2,
            )
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    path = bench.write_report(tmp_path)
    assert path == tmp_path / "bench.md"
    text = path.read_text(encoding="utf-8")
    summary = [line for line in text.splitlines() if line.startswith("| model-")]
    assert [line.split(" | ")[0] for line in summary[:2]] == ["| model-a", "| model-b"]
    assert "| model-a | 2 |" in summary[0]  # two runs
    assert text.count("| model-a | anxious-newcomer |") == 1
    assert "## Per run" in text
    header = next(line for line in text.splitlines() if line.startswith("| model | runs"))
    assert len(header.split("|")) == len(summary[0].split("|"))


def test_write_report_needs_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no bench runs"):
        bench.write_report(tmp_path)


@given(st.lists(st.one_of(st.none(), st.floats(min_value=-100, max_value=100)), max_size=10))
def test_mean_of_runs_ignores_missing_values(values: list[float | None]) -> None:
    present = [v for v in values if v is not None]
    result = bench.mean_or_none(values)
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
    def forbidden(name: str, model: str | None) -> FakeProvider:
        raise AssertionError("no provider may be created without --yes")

    monkeypatch.setattr(bench, "make_provider", forbidden)
    assert main(["bench", "--provider", provider, "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "costs money" in err
    assert "up to 420 model calls" in err  # 7 personas x 3 days x (6 turns x 3 + 2 notes)
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


def test_bench_cli_reports_an_unreadable_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "other-model").mkdir()
    (tmp_path / "other-model" / "broken.jsonl").write_text("nope\n", encoding="utf-8")
    args = ["bench", "--persona", "anxious-newcomer", "--days", "1", "--turns", "1"]
    assert main([*args, "--out", str(tmp_path)]) == 1
    assert "broken.jsonl:1" in capsys.readouterr().err


def test_bench_cli_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["bench", "--persona", "no-such", "--out", str(tmp_path)]) == 2
    assert "no bundled persona 'no-such'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        main(["bench", "--days", "0"])
    assert info.value.code == 2

    class Broken(FakeProvider):
        def complete(
            self,
            messages: Sequence[ChatMessage],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            raise LLMError("quota exceeded")

    monkeypatch.setattr(bench, "make_provider", lambda name, model: Broken())
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
