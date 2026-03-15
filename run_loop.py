#!/usr/bin/env python3
"""Backward-compatible alias for train.py.

Prefer: python3 train.py ...
"""

from __future__ import annotations

from train import main


if __name__ == "__main__":
    raise SystemExit(main())
