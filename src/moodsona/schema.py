"""A small JSON Schema validator for the persona schema.

``personas/schema.json`` is the public contract. To keep the core dependency
free (PyYAML only), this module implements exactly the keywords that schema
uses. Any other keyword raises :class:`UnsupportedSchemaError`, so a schema
change that needs more is caught by the tests instead of being silently
ignored. The test suite cross-checks this validator against the reference
``jsonschema`` package.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

#: Keywords that carry no validation.
_ANNOTATIONS = frozenset({"$schema", "$id", "$defs", "title", "description", "default"})
_SUPPORTED = _ANNOTATIONS | {
    "type",
    "enum",
    "const",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "$ref",
}

_TYPES: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer())
    ),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
}


class UnsupportedSchemaError(ValueError):
    """The schema uses a keyword this validator does not implement."""


def _path(parts: tuple[str | int, ...]) -> str:
    return "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts).lstrip(".") or "$"


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: 1 == 1.0 but True != 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return bool(a == b)


def _resolve(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise UnsupportedSchemaError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node  # type: ignore[no-any-return]


def iter_errors(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any] | None = None,
    path: tuple[str | int, ...] = (),
) -> Iterator[str]:
    """Yield one human-readable message per validation error.

    Args:
        instance: The data to check (as loaded from YAML or JSON).
        schema: The (sub)schema to check against.
        root: The full schema, for ``$ref`` resolution (defaults to ``schema``).
        path: Location of ``instance`` inside the document.

    Raises:
        UnsupportedSchemaError: The schema uses a keyword not implemented here.
    """
    root = schema if root is None else root
    unknown = set(schema) - _SUPPORTED
    if unknown:
        raise UnsupportedSchemaError(f"unsupported schema keywords: {sorted(unknown)}")
    where = _path(path)
    if "$ref" in schema:
        yield from iter_errors(instance, _resolve(schema["$ref"], root), root=root, path=path)
    if "type" in schema and not (isinstance(schema["type"], str) and schema["type"] in _TYPES):
        raise UnsupportedSchemaError(f"unsupported type {schema['type']!r}")
    if "type" in schema and not _TYPES[schema["type"]](instance):
        yield f"{where}: expected {schema['type']}, got {type(instance).__name__}"
        return
    if "const" in schema and not _equal(instance, schema["const"]):
        yield f"{where}: must be {schema['const']!r}"
    if "enum" in schema and not any(_equal(instance, v) for v in schema["enum"]):
        yield f"{where}: {instance!r} is not one of {schema['enum']}"
    if isinstance(instance, str):
        yield from _string_errors(instance, schema, where)
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            yield f"{where}: {instance} is below the minimum {schema['minimum']}"
        if "maximum" in schema and instance > schema["maximum"]:
            yield f"{where}: {instance} is above the maximum {schema['maximum']}"
    if isinstance(instance, list):
        yield from _array_errors(instance, schema, root, path)
    if isinstance(instance, dict):
        yield from _object_errors(instance, schema, root, path)


def _string_errors(value: str, schema: Mapping[str, Any], where: str) -> Iterator[str]:
    if "minLength" in schema and len(value) < schema["minLength"]:
        yield f"{where}: shorter than {schema['minLength']} characters"
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        yield f"{where}: longer than {schema['maxLength']} characters"
    if "pattern" in schema and re.search(schema["pattern"], value) is None:
        yield f"{where}: {value!r} does not match {schema['pattern']!r}"


def _array_errors(
    value: list[Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: tuple[str | int, ...],
) -> Iterator[str]:
    where = _path(path)
    if "minItems" in schema and len(value) < schema["minItems"]:
        yield f"{where}: needs at least {schema['minItems']} item(s)"
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        yield f"{where}: allows at most {schema['maxItems']} item(s)"
    if schema.get("uniqueItems"):
        for i, item in enumerate(value):
            if any(_equal(item, other) for other in value[:i]):
                yield f"{where}: items must be unique ({item!r} repeats)"
                break
    if "items" in schema:
        for i, item in enumerate(value):
            yield from iter_errors(item, schema["items"], root=root, path=(*path, i))


def _object_errors(
    value: dict[Any, Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: tuple[str | int, ...],
) -> Iterator[str]:
    where = _path(path)
    for name in schema.get("required", ()):
        if name not in value:
            yield f"{where}: missing required field {name!r}"
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                yield f"{where}: unknown field {name!r}"
    elif isinstance(schema.get("additionalProperties"), dict):
        raise UnsupportedSchemaError("additionalProperties must be a boolean here")
    for name, subschema in properties.items():
        if name in value:
            yield from iter_errors(value[name], subschema, root=root, path=(*path, name))


def validate(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Return every validation error of ``instance`` (an empty list when valid)."""
    return list(iter_errors(instance, schema))
