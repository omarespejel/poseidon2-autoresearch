#!/usr/bin/env python3
"""Shared target configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TargetConfigError(ValueError):
    """Raised when the target config file does not match the expected schema."""


def load_targets_file(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TargetConfigError(f"Failed to parse targets file: {path}") from exc
    if not isinstance(data, dict):
        raise TargetConfigError(f"Targets file must contain a top-level JSON object: {path}")

    targets = data.get("targets")
    if not isinstance(targets, dict):
        raise TargetConfigError(f"Targets file must contain an object-valued 'targets' key: {path}")
    return targets
