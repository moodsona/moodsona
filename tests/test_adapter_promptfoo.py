"""The promptfoo adapter, called the way promptfoo calls a Python provider."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from moodsona import EchoCompanion, bench, run
from moodsona.adapters.promptfoo import DEFAULTS, call_api
from moodsona.llm.base import LLMConfigurationError
from moodsona.llm.fake import FakeProvider
from moodsona.metrics import compute_metrics
from moodsona.personas import bundled_persona_path, load_bundled_persona

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "promptfoo"


def _call(
    config: dict[str, Any] | None = None,
    test_vars: dict[str, Any] | None = None,
    *,
    prompt: str = "moodsona",
    base_path: str | None = None,
) -> dict[str, Any]:
    """Call the provider with promptfoo-shaped arguments."""
    config = dict(config or {})
    if base_path is not None:
        config["basePath"] = base_path  # promptfoo adds the config folder to the config
    options: dict[str, Any] = {"id": "file://provider.py", "config": config}
    return call_api(prompt, options, {"vars": test_vars or {}, "prompt": {"raw": prompt}})


def _small(**changes: Any) -> dict[str, Any]:
    """A short offline run: 2 days x 3 turns."""
    return {"days": 2, "turns_per_day": 3, **changes}


def _error(result: dict[str, Any]) -> str:
    assert set(result) == {"error"}, result
    message = result["error"]
    assert isinstance(message, str)
    return message


def _forbid_make_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(name: str, model: str | None) -> FakeProvider:
        raise AssertionError("no provider may be created")

    monkeypatch.setattr(bench, "make_provider", forbidden)


def test_defaults_run_the_bundled_persona_offline_against_the_echo_companion() -> None:
    result = _call()
    assert set(result) == {"output"}
    output = result["output"]
    assert output["persona"] == DEFAULTS["persona"] == "anxious-newcomer"
    assert output["companion"] == DEFAULTS["companion"] == "moodsona:EchoCompanion"
    assert output["model"] == "fake"
    assert (output["days"], output["turns_per_day"], output["seed"]) == (3, 6, 0)
    first, second = output["transcript"][:2]
    assert first["role"] == "user"
    assert (first["day"], first["turn"]) == (1, 1)
    assert second == {
        "day": 1,
        "turn": 1,
        "role": "assistant",
        "content": "You said: " + first["content"],
    }
    assert output["metrics"]["persona_id"] == "anxious-newcomer"


def test_output_is_json_and_matches_a_direct_run() -> None:
    output = _call(_small(seed=4))["output"]
    assert json.loads(json.dumps(output)) == output
    persona = load_bundled_persona("anxious-newcomer")
    transcript = run(persona, FakeProvider(), EchoCompanion(), 2, 3, 4)
    expected = compute_metrics(transcript, language=persona.language).to_dict()
    assert output["metrics"] == json.loads(json.dumps(expected))
    assert output["metrics"]["turns"] == len(transcript.turns)
    assert [(m["role"], m["content"]) for m in output["transcript"]] == [
        (m.role, m.content) for m in transcript.visible_history()
    ]
    assert output["trajectory"] == [
        {"day": p.day, "turn": p.turn, "emotion": p.emotion, "energy": p.energy, "stance": p.stance}
        for p in transcript.trajectory()
    ]
    assert output["skipped_days"] == transcript.skipped_days


def test_a_companion_import_path_with_arguments() -> None:
    config = _small(
        companion="moodsona:ScriptedCompanion",
        companion_args={"replies": ["Tell me more.", "I see."], "cycle": True},
    )
    output = _call(config)["output"]
    replies = [m["content"] for m in output["transcript"] if m["role"] == "assistant"]
    assert replies[:3] == ["Tell me more.", "I see.", "Tell me more."]
    assert output["companion"] == "moodsona:ScriptedCompanion"


def test_a_dotted_attribute_after_the_colon() -> None:
    output = _call(_small(companion="moodsona.companion:EchoCompanion"))["output"]
    assert output["transcript"][1]["content"].startswith("You said: ")


@pytest.mark.parametrize(
    ("companion", "problem"),
    [
        ("moodsona.EchoCompanion", "package.module:Name"),
        ("moodsona:", "package.module:Name"),
        ("no_such_module_for_moodsona_tests:Bot", "cannot import"),
        ("moodsona:NoSuchCompanion", "has no attribute"),
        ("moodsona:FakeProvider", "no reply() method"),
        ("moodsona:__version__", "not callable"),
        (42, "must be a string"),
    ],
)
def test_bad_companions_are_errors(companion: Any, problem: str) -> None:
    assert problem in _error(_call(_small(companion=companion)))


def test_companion_args_that_do_not_fit_are_an_error() -> None:
    message = _error(_call(_small(companion_args={"api_key": "s3cret", "no_such_argument": 1})))
    assert "moodsona:EchoCompanion" in message
    assert "no_such_argument" in message
    assert "s3cret" not in message  # argument values may be secrets


def test_companion_args_must_be_a_mapping() -> None:
    assert "companion_args" in _error(_call(_small(companion_args=["x"])))


def test_persona_by_bundled_id() -> None:
    output = _call(_small(persona="lonely-night-owl"))["output"]
    assert output["persona"] == "lonely-night-owl"
    assert output["metrics"]["persona_id"] == "lonely-night-owl"


def test_persona_by_path_absolute_and_relative_to_the_config(tmp_path: Path) -> None:
    source = bundled_persona_path("boundary-tester")
    copy = tmp_path / "personas" / "my-tester.yaml"
    copy.parent.mkdir()
    copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    absolute = _call(_small(persona=str(copy)))["output"]
    relative = _call(_small(persona="personas/my-tester.yaml"), base_path=str(tmp_path))["output"]
    assert absolute["persona"] == relative["persona"] == "boundary-tester"
    assert absolute["metrics"] == relative["metrics"]


def test_unknown_persona_id_and_bad_persona_file_are_errors(tmp_path: Path) -> None:
    assert "no bundled persona 'no-such-persona'" in _error(
        _call(_small(persona="no-such-persona"))
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: bad\n", encoding="utf-8")
    assert str(bad) in _error(_call(_small(persona=str(bad))))
    assert "missing.yaml" in _error(_call(_small(persona="missing.yaml")))


def test_vars_override_the_run_settings_and_other_vars_are_ignored() -> None:
    result = _call(
        _small(),
        {"persona": "lonely-night-owl", "days": "1", "turns_per_day": 2, "seed": 3, "topic": "x"},
    )
    output = result["output"]
    assert output["persona"] == "lonely-night-owl"
    assert (output["days"], output["turns_per_day"], output["seed"]) == (1, 2, 3)


@pytest.mark.parametrize("key", ["provider", "model", "allow_paid", "companion", "companion_args"])
def test_vars_cannot_change_the_provider_or_the_companion(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_make_provider(monkeypatch)
    message = _error(_call(_small(), {key: "anthropic"}))
    assert key in message
    assert "provider config" in message


def test_the_prompt_is_ignored() -> None:
    assert _call(_small(), prompt="anything")["output"] == _call(_small(), prompt="")["output"]


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        ({"days": 0}, "days must be an integer of at least 1"),
        ({"turns_per_day": "many"}, "turns_per_day must be an integer of at least 1"),
        ({"days": True}, "days must be an integer"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"turns": 3}, "unknown config key 'turns'"),
        ({"provider": "local"}, "provider must be one of fake, anthropic, openai"),
        ({"model": 5}, "model must be a string"),
        ({"persona": 7}, "persona must be a string"),
        ({"allow_paid": "yes"}, "allow_paid must be true or false"),
    ],
)
def test_bad_settings_are_errors(config: dict[str, Any], problem: str) -> None:
    assert problem in _error(_call(config))


def test_config_and_vars_must_be_mappings() -> None:
    assert "config must be a mapping" in _error(call_api("", {"config": ["days"]}, {}))
    assert "vars must be a mapping" in _error(call_api("", {}, {"vars": "persona"}))
    assert "output" in call_api("", {"config": None}, None)


@pytest.mark.parametrize(
    ("key", "meant"),
    [("turnsPerDay", "turns_per_day"), ("Persona", "persona"), ("companionArgs", "companion_args")],
)
def test_a_moodsona_key_in_another_case_is_an_error(key: str, meant: str) -> None:
    message = _error(_call(_small(**{key: "x"})))
    assert f"unknown config key {key!r}" in message
    assert f"did you mean {meant!r}" in message


def test_promptfoo_own_config_keys_are_accepted() -> None:
    config = _small(pythonExecutable=".venv/bin/python", workers=2, timeout=600000)
    assert "output" in _call(config)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_a_paid_provider_is_refused_without_allow_paid(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_make_provider(monkeypatch)
    for config in (_small(provider=provider), _small(provider=provider, allow_paid=False)):
        message = _error(_call(config))
        assert "costs money" in message
        assert "allow_paid: true" in message
        assert f"up to {bench.max_model_calls(1, 2, 3)} model calls" in message


def test_the_paid_refusal_comes_before_the_companion_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_make_provider(monkeypatch)
    config = _small(provider="openai", companion="no_such_module_for_moodsona_tests:Bot")
    assert "costs money" in _error(_call(config))


def test_allow_paid_creates_the_named_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[tuple[str, str | None]] = []

    class Labelled(FakeProvider):
        model = "test-model"

    def fake_make_provider(name: str, model: str | None) -> FakeProvider:
        made.append((name, model))
        return Labelled()

    monkeypatch.setattr(bench, "make_provider", fake_make_provider)
    config = _small(provider="anthropic", model="test-model", allow_paid=True)
    output = _call(config)["output"]
    assert made == [("anthropic", "test-model")]
    assert output["model"] == "anthropic-test-model"
    assert output["metrics"]["turns"] == 6


def test_a_provider_that_cannot_be_made_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(name: str, model: str | None) -> FakeProvider:
        raise LLMConfigurationError("install moodsona[anthropic]")

    monkeypatch.setattr(bench, "make_provider", missing_sdk)
    message = _error(_call(_small(provider="anthropic", allow_paid=True)))
    assert "install moodsona[anthropic]" in message


@pytest.mark.parametrize("error", [RuntimeError("no API key set"), KeyError()])
def test_any_exception_while_making_the_provider_is_an_error(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(name: str, model: str | None) -> FakeProvider:
        raise error

    monkeypatch.setattr(bench, "make_provider", failing)
    message = _error(_call(_small(provider="openai", allow_paid=True)))
    assert type(error).__name__ in message


def test_a_failing_companion_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def reply(self, history: Any) -> str:
            raise RuntimeError("the bot is down")

    monkeypatch.setattr("moodsona.EchoCompanion", Broken)
    assert "the run failed: RuntimeError: the bot is down" in _error(_call(_small()))


def test_a_companion_that_fails_to_construct_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_factory() -> EchoCompanion:
        raise RuntimeError("no connection")

    monkeypatch.setattr("moodsona.EchoCompanion", broken_factory)
    assert "RuntimeError: no connection" in _error(_call(_small()))


# ------------------------------------------------------------- the example ----

_ASSERTION = re.compile(r"^output((?:\.\w+)+) (===|<=|>=) (\S+)$")


def _evaluate(expression: str, output: Any) -> bool:
    """Evaluate the small subset of JavaScript the example's assertions use."""
    match = _ASSERTION.match(expression.strip())
    assert match, f"the example assertion {expression!r} is outside the checked subset"
    path, operator, literal = match.groups()
    value = output
    for name in path.strip(".").split("."):
        value = len(value) if name == "length" and isinstance(value, list) else value[name]
    expected = json.loads(literal)
    if operator == "===":
        return bool(value == expected)
    if operator == "<=":
        return bool(value <= expected)
    return bool(value >= expected)


def test_the_example_config_runs_and_its_assertions_hold() -> None:
    config = yaml.safe_load((EXAMPLE_DIR / "promptfooconfig.yaml").read_text(encoding="utf-8"))
    (provider,) = config["providers"]
    assert provider["id"] == "file://provider.py"
    assert provider["config"]["provider"] == "fake"
    prompt = config["prompts"][0]
    default_asserts = config.get("defaultTest", {}).get("assert", [])
    assert config["tests"]
    for test in config["tests"]:
        test_vars = test.get("vars", {})
        result = _call(provider["config"], test_vars, prompt=prompt, base_path=str(EXAMPLE_DIR))
        assert "output" in result, result
        asserts = default_asserts + test.get("assert", [])
        assert asserts
        for check in asserts:
            assert check["type"] == "javascript"
            assert _evaluate(check["value"], result["output"]), (test_vars, check["value"])


def test_the_example_provider_file_exports_call_api() -> None:
    path = EXAMPLE_DIR / "provider.py"
    spec = importlib.util.spec_from_file_location("moodsona_promptfoo_example_provider", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.call_api is call_api
