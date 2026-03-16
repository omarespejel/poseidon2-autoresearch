#!/usr/bin/env python3
"""Shared target configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_targets_file(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["targets"]
