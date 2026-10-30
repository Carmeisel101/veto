"""Committed, invented parity fixtures (safe to publish; no private data)."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).with_name("parity_fixtures.json")


def load_fixtures() -> list[dict]:
    """Return ``[{"category": ..., "text": ...}, ...]``."""
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["examples"]
