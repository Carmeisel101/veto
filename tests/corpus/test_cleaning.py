import pandas as pd
import pytest

from veto.corpus.cleaning import (
    apple_to_datetime,
    normalize_text,
    text_hash,
    to_received_at,
)


def test_normal_text_unchanged():
    assert normalize_text("Hello, World!") == "Hello, World!"


def test_preserves_case_punctuation_urls_amounts():
    s = "URGENT!!! Donate $25 now: https://example.test/x?a=1 STOP to end"
    assert normalize_text(s) == s


def test_emoji_and_unicode_survive():
    assert normalize_text("Good morning 🩷 café Ñandú 你好") == "Good morning 🩷 café Ñandú 你好"


def test_unicode_nfc():
    assert normalize_text("café") == "café"


def test_multiline_structure_kept_and_normalized():
    assert normalize_text("a  b\r\nc \r\n\r\n\r\n\r\nd") == "a b\nc\n\nd"


def test_strips_outer_whitespace():
    assert normalize_text("  \n hi \t") == "hi"


def test_null_and_empty():
    assert normalize_text(None) is None
    assert normalize_text(float("nan")) is None
    assert normalize_text("") == ""
    assert normalize_text("  \n\t ") == ""


def test_hash_stable_and_sensitive():
    assert text_hash("a") == text_hash("a")
    assert text_hash("a") != text_hash("A")
    assert len(text_hash("a")) == 64


def test_apple_timestamp_nanoseconds():
    s = pd.Series([0, 86_400 * 10**9])
    out = apple_to_datetime(s)
    assert out.iloc[0] == pd.Timestamp("2001-01-01", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("2001-01-02", tz="UTC")


def test_apple_timestamp_seconds_legacy():
    out = apple_to_datetime(pd.Series([86_400]))
    assert out.iloc[0] == pd.Timestamp("2001-01-02", tz="UTC")


def test_received_at_is_utc_aware():
    s = pd.to_datetime(pd.Series(["2026-01-01 12:00"])).dt.tz_localize("America/Chicago")
    out = to_received_at(s)
    assert str(out.dt.tz) == "UTC"
    assert out.iloc[0] == pd.Timestamp("2026-01-01 18:00", tz="UTC")


def test_naive_datetimes_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        to_received_at(pd.to_datetime(pd.Series(["2026-01-01"])))


def test_null_timestamps_rejected():
    with pytest.raises(ValueError):
        to_received_at(pd.Series([1.0, None]))
