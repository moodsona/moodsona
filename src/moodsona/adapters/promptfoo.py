"""promptfoo provider: one promptfoo test case is one multi-day moodsona run.

`promptfoo <https://www.promptfoo.dev>`_ is a Node.js evaluation tool. It loads
a Python provider from a file (``id: file://provider.py``) and calls its
``call_api(prompt, options, context)`` once per test case. Point it at a file
that re-exports :func:`call_api` (``examples/promptfoo/provider.py`` in the
repository). promptfoo does not need to be installed for this module; nothing
here imports it. Experimental: the config keys and the output may still change.

Settings come from the provider's ``config`` (see :data:`DEFAULTS` for the keys
and their defaults). A test case's ``vars`` may override the run settings in
:data:`RUN_KEYS` (persona, days, turns per day, seed), so one provider can run
several personas. The provider, model, ``allow_paid`` and the companion can
only be set in the provider config, so a test case cannot switch to a paid
model. Other vars are ignored (they may be meant for the prompt), and so is the
prompt itself: the persona writes the user's lines.

The companion under test is named by an import path, ``"package.module:Name"``.
``Name`` (a class or a factory function) is called with ``companion_args`` as
keyword arguments, and must return an object with a ``reply(history)`` method
(see :class:`moodsona.Companion`). The module must be importable by the Python
that promptfoo runs.

The output is a JSON-ready dict: the visible transcript, the trajectory, the
skipped days and ``metrics`` (the dict that ``moodsona report --json`` prints).
Every problem is returned as promptfoo's ``{"error": "..."}`` instead of being
raised, including exceptions from the companion.

A provider other than ``"fake"`` calls a paid API. It is refused unless the
config sets ``allow_paid: true``; the refusal names the most model calls the
run could make (:func:`moodsona.bench.max_model_calls`).
"""

from __future__ import annotations

import pkgutil
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from moodsona import bench
from moodsona.companion import Companion
from moodsona.metrics import compute_metrics
from moodsona.personas import PersonaError, load_bundled_persona, load_persona
from moodsona.runner import run
from moodsona.state import PersonaSeed
from moodsona.validation import YAML_SUFFIXES

__all__ = ["DEFAULTS", "RUN_KEYS", "call_api"]

#: The provider config keys and their defaults.
DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "persona": "anxious-newcomer",  # a bundled persona id or a persona YAML path
        "companion": "moodsona:EchoCompanion",  # import path of the companion under test
        "companion_args": None,  # keyword arguments for the companion (a mapping)
        "provider": "fake",  # the user model: one of moodsona.bench.PROVIDERS
        "model": None,  # model id for a real provider; None uses the adapter's default
        "allow_paid": False,  # must be true for any provider but "fake"
        "days": 3,
        "turns_per_day": 6,
        "seed": 0,
    }
)

#: The settings a test case's ``vars`` may override.
RUN_KEYS = ("persona", "days", "turns_per_day", "seed")

#: promptfoo's own Python provider keys that are all lowercase (camelCase keys
#: such as ``pythonExecutable`` are promptfoo's too); they are ignored here.
_PROMPTFOO_KEYS = frozenset({"workers", "timeout"})

_INTEGER = re.compile(r"[+-]?[0-9]+")


def _fold(key: object) -> str:
    """``key`` without case and underscores: ``turnsPerDay`` and ``turns_per_day`` match."""
    return str(key).replace("_", "").lower()


_MOODSONA_KEYS_FOLDED = {_fold(key): key for key in DEFAULTS}


@dataclass(frozen=True, slots=True)
class _Settings:
    persona: str
    companion: str
    companion_args: Mapping[str, Any]
    provider: str
    model: str | None
    allow_paid: bool
    days: int
    turns_per_day: int
    seed: int
    base_path: str | None  # promptfoo's config folder, which anchors a relative persona path


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
            return {"error": _paid_refusal(settings)}  # before any user code or provider
        persona = _persona(settings.persona, settings.base_path)
        companion = _companion(settings.companion, settings.companion_args)
    except (_ConfigError, PersonaError, FileNotFoundError) as exc:
        return {"error": f"moodsona: {exc}"}
    # Provider SDKs and the companion (user code) can raise anything: report it, do not crash.
    try:
        provider = bench.make_provider(settings.provider, settings.model)
    except Exception as exc:
        return {
            "error": f"moodsona: cannot create provider {settings.provider!r}: {_describe(exc)}"
        }
    try:
        transcript = run(
            persona, provider, companion, settings.days, settings.turns_per_day, settings.seed
        )
    except Exception as exc:
        return {"error": f"moodsona: the run failed: {_describe(exc)}"}
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


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _paid_refusal(settings: _Settings) -> str:
    calls = bench.max_model_calls(1, settings.days, settings.turns_per_day)
    return (
        f"moodsona: provider {settings.provider!r} calls a paid API and costs money: up to "
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
    raw = {**DEFAULTS}
    raw.update((k, v) for k, v in config.items() if k in DEFAULTS)
    raw.update((k, test_vars[k]) for k in RUN_KEYS if k in test_vars)
    provider = raw["provider"]
    if provider not in bench.PROVIDERS:
        raise _ConfigError(
            f"provider must be one of {', '.join(bench.PROVIDERS)}, got {provider!r}"
        )
    args = {} if raw["companion_args"] is None else raw["companion_args"]
    if not isinstance(args, Mapping) or not all(isinstance(k, str) for k in args):
        raise _ConfigError("companion_args must be a mapping of argument names to values")
    allow_paid = raw["allow_paid"]
    if not isinstance(allow_paid, bool):
        raise _ConfigError(f"allow_paid must be true or false, got {allow_paid!r}")
    model, base_path = raw["model"], config.get("basePath")
    return _Settings(
        persona=_string("persona", raw["persona"]),
        companion=_string("companion", raw["companion"]),
        companion_args=args,
        provider=provider,
        model=None if model is None else _string("model", model),
        allow_paid=allow_paid,
        days=_integer("days", raw["days"], minimum=1),
        turns_per_day=_integer("turns_per_day", raw["turns_per_day"], minimum=1),
        seed=_integer("seed", raw["seed"]),
        base_path=base_path if isinstance(base_path, str) and base_path else None,
    )


def _string(key: str, value: object) -> str:
    if not isinstance(value, str):
        raise _ConfigError(f"{key} must be a string, got {value!r}")
    return value


def _integer(key: str, value: object, *, minimum: int | None = None) -> int:
    """An int, or a string of one (vars read from a CSV file are strings)."""
    if isinstance(value, str) and _INTEGER.fullmatch(value.strip()):
        number = int(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        number = value
    else:
        number = None
    if number is not None and (minimum is None or number >= minimum):
        return number
    need = "an integer" if minimum is None else f"an integer of at least {minimum}"
    raise _ConfigError(f"{key} must be {need}, got {value!r}")


def _persona(value: str, base_path: str | None) -> PersonaSeed:
    """Load a persona file, or a bundled persona when ``value`` is a bare id and no such file.

    The rule is the one ``moodsona smoke`` uses. A relative path is read from
    ``base_path`` (promptfoo's config folder) when given.

    Raises:
        PersonaError: The file is missing or invalid.
        _ConfigError: ``value`` is a bare name, but no bundled persona has that id.
        FileNotFoundError: The bundled personas are missing (a broken install).
    """
    path = Path(value)
    if not path.is_absolute() and base_path:
        path = Path(base_path) / path
    if path.is_file() or Path(value).name != value or path.suffix in YAML_SUFFIXES:
        return load_persona(path)
    try:
        return load_bundled_persona(value)
    except KeyError as exc:  # its message lists the bundled ids
        raise _ConfigError(exc.args[0]) from exc


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
