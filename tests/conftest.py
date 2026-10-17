"""Shared synthetic fixtures. No real message data appears in tests."""

import pandas as pd
import pytest


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message_id": [3, 1, 2, 4],
            "text": ["Hello  there\r\nfriend", "  Vote for Jane Doe 🗳️ now!!  ", "", None],
            "is_from_me": [0, 0, 0, 0],
            "date": pd.to_datetime(
                ["2026-01-02 10:00", "2026-01-01 09:00", "2026-01-03 08:00", "2026-01-04 08:00"]
            ).tz_localize("America/Chicago"),
            "sender": ["+15550000001", "+15550000002", "a@example.com", "+15550000003"],
            "chat_id": [10.0, 11.0, None, 12.0],
            "chat_identifier": ["+15550000001", "chat111", None, "+15550000003"],
            "chat_name": ["", "Fake Group", None, ""],
        }
    )
