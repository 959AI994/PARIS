"""Resolved configuration loading and reproducibility helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    resolved = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValidationError(f"configuration override must be key=value: {item}")
        dotted_key, raw_value = item.split("=", 1)
        parts = [part for part in dotted_key.split(".") if part]
        if not parts:
            raise ValidationError(f"invalid override key: {dotted_key}")
        cursor: Any = resolved
        for position, part in enumerate(parts[:-1]):
            next_part = parts[position + 1]
            if isinstance(cursor, list):
                if not part.isdigit() or int(part) >= len(cursor):
                    raise ValidationError(f"invalid list index in config override: {part}")
                cursor = cursor[int(part)]
                continue
            if not isinstance(cursor, dict):
                raise ValidationError(f"cannot descend into non-object config key: {part}")
            child = cursor.get(part)
            if child is None:
                child = [] if next_part.isdigit() else {}
                cursor[part] = child
            cursor = child
        final = parts[-1]
        if isinstance(cursor, list):
            if not final.isdigit() or int(final) >= len(cursor):
                raise ValidationError(f"invalid list index in config override: {final}")
            cursor[int(final)] = _parse_scalar(raw_value)
        elif isinstance(cursor, dict):
            cursor[final] = _parse_scalar(raw_value)
        else:
            raise ValidationError(f"cannot assign config override at: {dotted_key}")
    return resolved


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValidationError(f"configuration file not found: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid YAML configuration {source}: {exc}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValidationError("configuration root must be an object")
    resolved = _merge(dict(defaults or {}), payload)
    resolved = apply_overrides(resolved, overrides)
    resolved["_source"] = str(source)
    resolved["_base_dir"] = str(source.parent)
    resolved["_hash"] = config_hash(resolved)
    return resolved


def config_hash(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()


def write_resolved_config(path: str | Path, config: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(config), stream, sort_keys=True, allow_unicode=True)
