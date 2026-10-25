"""promptfoo provider: one promptfoo test case is one multi-day moodsona run.

Point promptfoo at a file that re-exports :func:`call_api`
(``examples/promptfoo/provider.py``). The provider config names the persona,
the companion under test (an import path) and the user model; the output is the
transcript plus the ``moodsona report --json`` metrics, and every problem comes
back as ``{"error": ...}``. A paid user model needs ``allow_paid: true``.
Experimental. See docs/integrations/promptfoo.md for the config keys, the
output and examples.
"""

from __future__ import annotations

import pkgutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Any

from moodsona import bench
from moodsona.companion import Companion
from moodsona.metrics import compute_metrics
from moodsona.personas import PersonaError, load_persona, resolve_persona_path
from moodsona.runner import run
from moodsona.state import PersonaSeed

__all__ = ["DEFAULTS", "RUN_KEYS", "call_api"]


@dataclass(frozen=True, slots=True)
class _Settings:
    """The provider config keys, with their defaults."""

    persona: str = "anxious-newcomer"  # a bundled persona id or a persona YAML path
    companion: str = "moodsona:EchoCompanion"  # import path of the companion under test
    companion_args: Mapping[str, Any] | None = None  # keyword arguments for the companion
    provider: str = "fake"  # the user model: one of moodsona.bench.PROVIDERS
    model: str | None = None  # model id for a real provider; None uses the adapter's default
    allow_paid: bool = False  # must be true for any provider but "fake"
    days: int = 3
    turns_per_day: int = 6
    seed: int = 0


#: The provider config keys and their defaults.
DEFAULTS: Mapping[str, Any] = MappingProxyType({f.name: f.default for f in fields(_Settings)})

#: The settings a test case's ``vars`` may override.
RUN_KEYS = ("persona", "days", "turns_per_day", "seed")

#: promptfoo's own Python provider keys that are all lowercase (camelCase keys
#: such as ``pythonExecutable`` are promptfoo's too); they are ignored here.
_PROMPTFOO_KEYS = frozenset({"workers", "timeout"})


def _fold(key: object) -> str:
    """``key`` without case, underscores and dashes: ``turnsPerDay`` matches ``turns_per_day``."""
    return str(key).replace("_", "").replace("-", "").lower()


_MOODSONA_KEYS_FOLDED = {_fold(key): key for key in DEFAULTS}


class _ConfigError(Exception):
    """A setting is wrong; the message is returned to promptfoo as the error."""


def call_api(
    prompt: str, options: Mapping[str, Any], context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Run one persona against the configured companion; promptfoo's provider entry point.

    Args:
        prompt: The rendered promptfoo prompt. Not used: the persona writes the
            user's lines.
        options: promptfoo's provider options; ``options["config"]`` holds the
            settings (see :data:`DEFAULTS`) and the ``basePath`` promptfoo adds
            (the folder of the promptfoo config), which anchors a relative
            persona path.
        context: promptfoo's call context; ``context["vars"]`` may override the
            keys in :data:`RUN_KEYS`.

    Returns:
        ``{"output": {...}}`` with ``persona``, ``companion``, ``model`` (the
        user-model label), ``days``, ``turns_per_day``, ``seed``,
        ``transcript`` (one ``{"day", "turn", "role", "content"}`` per visible
        message), ``trajectory`` (one ``{"day", "turn", "emotion", "energy",
        "stance"}`` per turn), ``skipped_days`` and ``metrics``
        (:meth:`moodsona.metrics.TranscriptMetrics.to_dict`); or
        ``{"error": "..."}`` when a setting is wrong, a paid provider is not
        allowed, or the run fails.
    """
    del prompt  # documented as unused
    try:
        settings = _settings(options.get("config"), (context or {}).get("vars"))
        if settings.provider != "fake" and not settings.allow_paid:
            return _fail(_paid_refusal(settings))  # before any user code or provider
        persona = _persona(settings.persona, (options.get("config") or {}).get("basePath"))
        companion = _companion(settings.companion, settings.companion_args or {})
    except (_ConfigError, PersonaError, FileNotFoundError) as exc:
        return _fail(str(exc))
    # Provider SDKs and the companion (user code) can raise anything: report it, do not crash.
    try:
        provider = bench.make_provider(settings.provider, settings.model)
    except Exception as exc:
        return _fail(f"cannot create provider {settings.provider!r}: {_describe(exc)}")
    try:
        transcript = run(
            persona, provider, companion, settings.days, settings.turns_per_day, settings.seed
        )
    except Exception as exc:
        return _fail(f"the run failed: {_describe(exc)}")
    output = {
        "persona": persona.id,
        "companion": settings.companion,
        "model": bench.model_label(settings.provider, provider),
        "days": settings.days,
        "turns_per_day": settings.turns_per_day,
        "seed": settings.seed,
        "transcript": [
            {"day": t.day, "turn": t.turn, "role": role, "content": content}
            for t in transcript.turns
            for role, content in (("user", t.visible), ("assistant", t.companion_reply))
        ],
        "trajectory": [asdict(point) for point in transcript.trajectory()],
        "skipped_days": list(transcript.skipped_days),
        "metrics": compute_metrics(transcript, language=persona.language).to_dict(),
    }
    return {"output": output}


def _fail(message: str) -> dict[str, Any]:
    return {"error": f"moodsona: {message}"}


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _paid_refusal(settings: _Settings) -> str:
    calls = bench.max_model_calls(1, settings.days, settings.turns_per_day)
    return (
        f"provider {settings.provider!r} calls a paid API and costs money: up to "
        f"{calls} model calls (1 persona x {settings.days} day(s) x {settings.turns_per_day} "
        "turn(s)) per test case. Set allow_paid: true in the provider config to run it."
    )


def _settings(config: object, test_vars: object) -> _Settings:
    config = {} if config is None else config
    test_vars = {} if test_vars is None else test_vars
    if not isinstance(config, Mapping):
        raise _ConfigError("the provider config must be a mapping")
    if not isinstance(test_vars, Mapping):
        raise _ConfigError("the test vars must be a mapping")
    for key in config:
        if key in DEFAULTS:
            continue
        meant = _MOODSONA_KEYS_FOLDED.get(_fold(key))
        if meant is not None:
            raise _ConfigError(f"unknown config key {key!r}; did you mean {meant!r}?")
        promptfoo_key = isinstance(key, str) and (key in _PROMPTFOO_KEYS or key != key.lower())
        if not promptfoo_key:
            raise _ConfigError(f"unknown config key {key!r}; moodsona keys: {', '.join(DEFAULTS)}")
    for key in test_vars:
        if key in DEFAULTS and key not in RUN_KEYS:
            raise _ConfigError(f"set {key} in the provider config, not in test vars")
        meant = _MOODSONA_KEYS_FOLDED.get(_fold(key))
        if key not in DEFAULTS and meant is not None:  # other vars may be for the prompt
            raise _ConfigError(f"unknown test var {key!r}; did you mean {meant!r}?")
    raw = {**DEFAULTS}
    raw.update((k, v) for k, v in config.items() if k in DEFAULTS)
    raw.update((k, test_vars[k]) for k in RUN_KEYS if k in test_vars)
    if raw["provider"] not in bench.PROVIDERS:
        raise _ConfigError(
            f"provider must be one of {', '.join(bench.PROVIDERS)}, got {raw['provider']!r}"
        )
    args = raw["companion_args"]
    if args is not None and not (
        isinstance(args, Mapping) and all(isinstance(k, str) for k in args)
    ):
        raise _ConfigError("companion_args must be a mapping of argument names to values")
    if not isinstance(raw["allow_paid"], bool):
        raise _ConfigError(f"allow_paid must be true or false, got {raw['allow_paid']!r}")
    for key in ("persona", "companion", "model"):
        if not isinstance(raw[key], str) and not (key == "model" and raw[key] is None):
            raise _ConfigError(f"{key} must be a string, got {raw[key]!r}")
    if raw["model"] is not None and raw["provider"] == "fake":
        raise _ConfigError(
            f"model is only used with provider anthropic or openai, but provider is fake "
            f"(got model {raw['model']!r}); set provider, or remove model"
        )
    for key, minimum in (("days", 1), ("turns_per_day", 1), ("seed", None)):
        raw[key] = _integer(key, raw[key], minimum=minimum)
    return _Settings(**raw)


def _integer(key: str, value: object, *, minimum: int | None) -> int:
    """An int, or a string of one (vars read from a CSV file are strings)."""
    try:
        number = (
            int(value) if isinstance(value, int | str) and not isinstance(value, bool) else None
        )
    except ValueError:  # not a number, or more digits than Python converts
        number = None
    if number is not None and (minimum is None or number >= minimum):
        return number
    need = "an integer" if minimum is None else f"an integer of at least {minimum}"
    raise _ConfigError(f"{key} must be {need}, got {value!r:.40}")


def _persona(value: str, base_path: object) -> PersonaSeed:
    """Load the persona ``value`` names; a relative path is read from promptfoo's ``basePath``."""
    base = base_path if isinstance(base_path, str) and base_path else None
    try:
        path = resolve_persona_path(value, base=base)
    except KeyError as exc:  # an unknown bundled id; the message lists the known ones
        raise _ConfigError(exc.args[0]) from exc
    return load_persona(path)


def _companion(spec: str, args: Mapping[str, Any]) -> Companion:
    """Import ``"package.module:Name"``, call it with ``args`` and check the result."""
    module_name, colon, attribute = spec.partition(":")  # the documented form only
    if not (colon and module_name and attribute):
        raise _ConfigError(f"companion must be an import path 'package.module:Name', got {spec!r}")
    # Importing and constructing run user code, so any exception is reported.
    try:
        target: Any = pkgutil.resolve_name(spec)
    except Exception as exc:
        raise _ConfigError(f"companion {spec!r}: cannot import it: {_describe(exc)}") from exc
    if not callable(target):
        raise _ConfigError(f"companion {spec!r} is not callable (use a class or a factory)")
    try:
        companion = target(**args)
    except Exception as exc:
        # Only the argument names: their values may be secrets, and promptfoo stores errors.
        names = ", ".join(args) or "none"
        raise _ConfigError(
            f"companion {spec!r} with companion_args ({names}): {_describe(exc)}"
        ) from exc
    if not isinstance(companion, Companion) or not callable(companion.reply):
        made = type(companion).__name__
        raise _ConfigError(f"companion {spec!r} made a {made} with no reply() method")
    return companion
