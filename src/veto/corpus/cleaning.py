"""Deterministic, conservative text and timestamp normalization.

Normalization deliberately preserves capitalization, punctuation, emoji, URLs,
numbers and line structure; those may carry signal for later phases.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import pandas as pd

APPLE_EPOCH = pd.Timestamp("2001-01-01", tz="UTC")

_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: Any) -> str | None:
    """Return conservatively normalized text, or ``None`` for null input.

    Steps: Unicode NFC; ``\\r\\n``/``\\r`` -> ``\\n``; runs of horizontal whitespace
    (spaces, tabs, NBSP, ...) -> one space; per-line trailing/leading whitespace
    stripped; 3+ consecutive newlines -> 2; outer whitespace stripped.
    An all-whitespace input yields ``""``.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return None
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _HORIZONTAL_WS.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _EXCESS_NEWLINES.sub("\n\n", s)
    return s.strip()


def text_hash(normalized_text: str) -> str:
    """Stable SHA-256 hex digest of normalized text, for exact-duplicate analysis."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def apple_to_datetime(values: pd.Series) -> pd.Series:
    """Convert Apple Messages timestamps to UTC-aware datetimes.

    Accepts nanoseconds since 2001-01-01 UTC (modern ``chat.db``). Values that
    look like seconds (abs < 1e11) are interpreted as seconds, as in older
    databases.
    """
    nums = pd.to_numeric(values, errors="coerce")
    if nums.isna().any() and not values.isna().equals(nums.isna()):
        raise ValueError("malformed Apple timestamp values")
    seconds = nums.where(nums.abs() >= 1e11, nums * 1e9)
    delta = pd.to_timedelta(seconds.astype("float64").round(), unit="ns")
    return (APPLE_EPOCH + delta).astype("datetime64[ns, UTC]")


def to_received_at(values: pd.Series) -> pd.Series:
    """Convert the raw ``date`` column to timezone-aware UTC ``received_at``.

    Accepts timezone-aware datetimes (converted to UTC) or numeric Apple
    timestamps. Naive datetimes are rejected as ambiguous. Null or unparseable
    values raise ``ValueError``.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        if getattr(values.dt, "tz", None) is None:
            raise ValueError("naive datetimes are ambiguous; raw dates must be timezone-aware")
        out = values.dt.tz_convert("UTC")
    elif pd.api.types.is_numeric_dtype(values):
        out = apple_to_datetime(values)
    else:
        raise ValueError(f"unsupported date dtype: {values.dtype}")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} null or malformed timestamps")
    return out.astype("datetime64[ns, UTC]")
