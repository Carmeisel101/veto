"""Canonical corpus schema and validation.

Validation functions raise :class:`ValidationError` with an aggregate message.
Messages never include message text or sender identifiers.
"""

from __future__ import annotations

import pandas as pd

LABEL_POLITICAL = "political"
LABEL_NON_POLITICAL = "non_political"
LABEL_UNCERTAIN = "uncertain"
LABELS: tuple[str, ...] = (LABEL_POLITICAL, LABEL_NON_POLITICAL, LABEL_UNCERTAIN)

LABEL_SOURCE_HUMAN = "human"
LABEL_SOURCES: tuple[str, ...] = (LABEL_SOURCE_HUMAN,)

#: Columns the raw export must contain.
REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "message_id",
    "text",
    "is_from_me",
    "date",
    "sender",
    "chat_id",
    "chat_identifier",
    "chat_name",
)

#: Columns of the corpus, in output order.
CORPUS_COLUMNS: tuple[str, ...] = (
    "message_id",
    "text",
    "normalized_text",
    "text_hash",
    "sender",
    "received_at",
    "chat_id",
    "chat_identifier",
    "chat_name",
    "is_group_chat",
    "is_from_me",
    "suggested_label",
    "label",
    "label_source",
    "label_confidence",
)

#: Columns an annotation table may contain. ``message_id`` and ``label`` are required.
ANNOTATION_COLUMNS: tuple[str, ...] = ("message_id", "label", "label_confidence")


class ValidationError(ValueError):
    """Raised when input or output data violates the schema."""


def require_columns(df: pd.DataFrame, required: tuple[str, ...], what: str) -> None:
    """Raise if ``df`` lacks any of ``required``."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"{what} is missing required columns: {missing}")


def validate_labels(values: pd.Series, column: str = "label", allow_null: bool = True) -> None:
    """Raise if ``values`` contains anything outside :data:`LABELS`."""
    nulls = values.isna()
    if not allow_null and nulls.any():
        raise ValidationError(f"{column}: {int(nulls.sum())} null values not allowed")
    bad = sorted({str(v) for v in values[~nulls].unique() if v not in LABELS})
    if bad:
        raise ValidationError(f"{column}: unsupported values {bad}; expected one of {list(LABELS)}")


def validate_raw(raw: pd.DataFrame) -> None:
    """Validate structural requirements of the raw export.

    Row-level problems that are expected and handled (null/empty text) are not
    errors here; they are reported by the builder.
    """
    require_columns(raw, REQUIRED_RAW_COLUMNS, "raw data")
    ids = raw["message_id"]
    if ids.isna().any():
        raise ValidationError(f"raw data has {int(ids.isna().sum())} missing message_id values")
    dup = int(ids.duplicated().sum())
    if dup:
        raise ValidationError(
            f"raw data has {dup} duplicate message_id values; message_id must be unique "
            "(a message joined to multiple chats must be resolved before corpus construction)"
        )


def validate_corpus(corpus: pd.DataFrame) -> None:
    """Validate a built corpus."""
    require_columns(corpus, CORPUS_COLUMNS, "corpus")
    extra = [c for c in corpus.columns if c not in CORPUS_COLUMNS]
    if extra:
        raise ValidationError(f"corpus has unexpected columns: {extra}")
    if corpus["message_id"].isna().any() or corpus["message_id"].duplicated().any():
        raise ValidationError("corpus message_id must be non-null and unique")
    for col in ("text", "normalized_text"):
        if corpus[col].isna().any() or (corpus[col] == "").any():
            raise ValidationError(f"corpus {col} must be non-null and non-empty")
    if corpus["received_at"].isna().any():
        raise ValidationError("corpus received_at must be non-null")
    if getattr(corpus["received_at"].dt, "tz", None) is None:
        raise ValidationError("corpus received_at must be timezone-aware")
    validate_labels(corpus["label"], "label")
    validate_labels(corpus["suggested_label"], "suggested_label")
    labeled = corpus["label"].notna()
    sources = corpus["label_source"]
    if (labeled & sources.isna()).any():
        raise ValidationError("every labeled row must have a label_source")
    if (~labeled & sources.notna()).any():
        raise ValidationError("label_source is set on rows without a label")
    bad_src = sorted({str(v) for v in sources.dropna().unique() if v not in LABEL_SOURCES})
    if bad_src:
        raise ValidationError(f"label_source: unsupported values {bad_src}")
    conf = corpus["label_confidence"].dropna()
    if ((conf < 0) | (conf > 1)).any():
        raise ValidationError("label_confidence must be within [0, 1]")
