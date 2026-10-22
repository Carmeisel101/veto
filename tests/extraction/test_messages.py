import sqlite3

import pandas as pd
import pytest

from veto.corpus import build_corpus
from veto.extraction.messages import decode_attributed_body, extract_raw_messages

APPLE_EPOCH = pd.Timestamp("2001-01-01", tz="UTC")


def fake_blob(text: str) -> bytes:
    body = text.encode()
    n = len(body)
    length = bytes([n]) if n < 0x80 else b"\x81" + n.to_bytes(2, "little")
    return (b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString"
            b"\x00\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+"
            + length + body + b"\x86\x84\x02iI\x01")


def ns(ts: str) -> int:
    return int((pd.Timestamp(ts, tz="UTC") - APPLE_EPOCH).total_seconds() * 1e9)


@pytest.mark.parametrize("text", ["hi 🩷", "x" * 300, "multi\nline"])
def test_decoder_roundtrip(text):
    assert decode_attributed_body(fake_blob(text)) == text


def test_decoder_handles_missing_and_garbage():
    assert decode_attributed_body(None) is None
    assert decode_attributed_body(b"no marker here") is None
    assert decode_attributed_body(b"NSString") is None


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "chat.db"
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,
            is_from_me INTEGER, group_title TEXT, date INTEGER, handle_id INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+15550000001');
        INSERT INTO chat VALUES (1, '+15550000001', ''), (2, 'chat999', 'Fake Group');
    """)
    rows = [
        (1, "plain text", None, 0, ns("2026-06-01")),
        (2, None, fake_blob("only in blob"), 0, ns("2026-06-02")),
        (3, None, None, 0, ns("2026-06-03")),                 # attachment, no text anywhere
        (4, "sent by me", None, 1, ns("2026-06-04")),         # outgoing: excluded
        (5, "old", None, 0, ns("2024-01-01")),                # outside window
        (6, "in two chats", None, 0, ns("2026-06-05")),
    ]
    c.executemany("INSERT INTO message VALUES (?, ?, ?, ?, NULL, ?, 1)", rows)
    c.executemany("INSERT INTO chat_message_join VALUES (?, ?)",
                  [(1, 1), (1, 2), (1, 3), (1, 5), (2, 6), (1, 6)])
    c.commit()
    c.close()
    return path


def test_extract_recovers_blob_text_and_keeps_null_text_rows(db):
    now = pd.Timestamp("2026-07-01", tz="UTC")
    df = extract_raw_messages(db, months=6, now=now)
    by_id = df.set_index("message_id")["text"]
    assert by_id[1] == "plain text"
    assert by_id[2] == "only in blob"
    assert pd.isna(by_id[3])  # not filtered out by a text IS NOT NULL clause
    assert 4 not in by_id.index and 5 not in by_id.index


def test_extract_unique_ids_and_schema(db):
    df = extract_raw_messages(db, months=None)
    assert df["message_id"].is_unique
    assert 5 in set(df["message_id"])  # months=None keeps old messages
    assert df.set_index("message_id").loc[6, "chat_id"] == 1  # lowest chat id wins
    assert str(df["date"].dt.tz) == "America/Chicago"


def test_extract_output_feeds_corpus(db):
    raw = extract_raw_messages(db, months=None)
    corpus, report = build_corpus(raw)
    assert report.removed_null_text == 1
    assert "only in blob" in set(corpus["text"])
