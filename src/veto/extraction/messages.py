"""Read incoming messages from ``chat.db`` into the raw-export schema.

Recent macOS versions often leave ``message.text`` NULL and store the body only in the
``attributedBody`` blob (a typedstream-serialized ``NSAttributedString``), so text is
recovered from the blob when the column is empty. Messages are never filtered on
``text IS NOT NULL``; empty-text rows are kept here and handled by corpus construction.

Logging reports aggregate counts only.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from veto.corpus.cleaning import apple_to_datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / "Library/Messages/chat.db"
DEFAULT_TIMEZONE = "America/Chicago"

_QUERY = """
    SELECT
        message.ROWID AS message_id,
        message.text,
        message.attributedBody,
        message.is_from_me,
        message.group_title,
        message.date,
        handle.id AS sender,
        chat.ROWID AS chat_id,
        chat.chat_identifier,
        chat.display_name AS chat_name
    FROM message
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    LEFT JOIN chat ON chat_message_join.chat_id = chat.ROWID
    WHERE message.is_from_me = 0
    ORDER BY message.ROWID, chat.ROWID
"""

RAW_COLUMNS = (
    "message_id", "text", "is_from_me", "group_title", "date",
    "sender", "chat_id", "chat_identifier", "chat_name",
)


def decode_attributed_body(blob: bytes | None) -> str | None:
    """Extract plain text from a Messages ``attributedBody`` blob; None if not found.

    The string follows the ``NSString`` class marker and 5 bytes of typedstream framing,
    prefixed by a length: one byte if < 0x80, else ``0x81`` + 2 bytes or ``0x82`` + 4 bytes
    (little-endian).
    """
    if blob is None:
        return None
    i = blob.find(b"NSString")
    if i < 0:
        return None
    rest = blob[i + len(b"NSString") + 5:]
    if not rest:
        return None
    n = rest[0]
    if n == 0x81:
        n, start = int.from_bytes(rest[1:3], "little"), 3
    elif n == 0x82:
        n, start = int.from_bytes(rest[1:5], "little"), 5
    else:
        start = 1
    return rest[start:start + n].decode("utf-8", errors="replace")


def extract_raw_messages(
    db_path: Path = DEFAULT_DB_PATH,
    months: int | None = 6,
    timezone: str = DEFAULT_TIMEZONE,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return incoming messages in the raw-export schema (see ``RAW_COLUMNS``).

    Args:
        db_path: path to ``chat.db``; opened read-only.
        months: keep only messages from the last N months; ``None`` or 0 keeps everything.
        timezone: timezone for the ``date`` column (timezone-aware output).
        now: reference time for the cutoff (defaults to the current time).

    A message that belongs to several chats yields one row (its lowest chat id), so
    ``message_id`` is unique.
    """
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(_QUERY, conn)
    finally:
        conn.close()
    return _to_raw_schema(df, months, timezone, now)


def _to_raw_schema(df: pd.DataFrame, months: int | None, timezone: str,
                   now: pd.Timestamp | None) -> pd.DataFrame:
    logger.info("Read %d incoming message rows.", len(df))
    n = len(df)
    df = df.drop_duplicates("message_id", keep="first")
    logger.info("Collapsed %d rows for messages in multiple chats.", n - len(df))

    null_text = df["text"].isna()
    decoded = df["attributedBody"].map(decode_attributed_body)
    recovered = null_text & decoded.notna() & decoded.str.strip().ne("")
    df = df.assign(text=df["text"].where(~null_text, decoded)).drop(columns="attributedBody")
    logger.info("Text NULL for %d messages; recovered %d from attributedBody.",
                int(null_text.sum()), int(recovered.sum()))

    df["date"] = apple_to_datetime(df["date"]).dt.tz_convert(timezone)
    if months:
        ref = now if now is not None else pd.Timestamp.now(tz="UTC")
        cutoff = ref - pd.DateOffset(months=months)
        before = len(df)
        df = df[df["date"] >= cutoff]
        logger.info("Dropped %d messages older than %d months.", before - len(df), months)
    return df[list(RAW_COLUMNS)].reset_index(drop=True)
