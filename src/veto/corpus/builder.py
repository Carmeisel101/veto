"""Corpus construction: validate, normalize, annotate, write.

Transformation logic (:func:`build_corpus`) is pure and separate from I/O
(:func:`build_corpus_from_files`). Logging reports aggregate counts only.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from veto.corpus import cleaning, labeling, schema

logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = Path("data/data.parquet")
DEFAULT_CORPUS_PATH = Path("data/corpus/corpus.parquet")

#: Messages group chats use identifiers like ``chat123456789``; 1:1 chats use the
#: counterpart's phone number or email.
_GROUP_CHAT_IDENTIFIER = re.compile(r"^chat\d+$")


@dataclass(frozen=True)
class BuildReport:
    """Aggregate statistics about a build. Contains no message content."""

    loaded: int
    removed_null_text: int
    removed_empty_text: int
    written: int
    exact_duplicate_texts: int
    group_chat_messages: int
    unknown_chat_messages: int
    unmatched_annotations: int
    labeled: int


def derive_is_group_chat(chat_identifier: pd.Series) -> pd.Series:
    """``True`` when ``chat_identifier`` matches ``chat<digits>``; null if no chat is known."""
    known = chat_identifier.notna()
    is_group = chat_identifier.where(known, "").astype(str).str.match(_GROUP_CHAT_IDENTIFIER)
    return is_group.astype("boolean").where(known, pd.NA)


def build_corpus(
    raw: pd.DataFrame, annotations: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, BuildReport]:
    """Build the corpus from a raw export. Does not modify ``raw``.

    Raises:
        ValidationError: for missing columns, missing/duplicate message ids,
            invalid labels, or other schema violations.
        ValueError: for malformed timestamps.

    Rows with null or empty (after normalization) text are removed and counted.
    Output is sorted by ``message_id`` so identical input gives identical output.
    """
    schema.validate_raw(raw)
    loaded = len(raw)
    df = raw.copy()

    normalized = df["text"].map(cleaning.normalize_text)
    null_mask = normalized.isna()
    empty_mask = normalized.eq("").fillna(False)
    keep = ~(null_mask | empty_mask)
    df, normalized = df[keep], normalized[keep]

    out = pd.DataFrame(index=df.index)
    out["message_id"] = df["message_id"].astype("int64")
    out["text"] = df["text"].astype(object)
    out["normalized_text"] = normalized.astype(object)
    out["text_hash"] = out["normalized_text"].map(cleaning.text_hash)
    out["sender"] = df["sender"].astype(object)
    try:
        out["received_at"] = cleaning.to_received_at(df["date"])
    except ValueError as exc:
        raise schema.ValidationError(f"invalid date column: {exc}") from exc
    out["chat_id"] = pd.to_numeric(df["chat_id"]).astype("Int64")
    out["chat_identifier"] = df["chat_identifier"].astype(object)
    out["chat_name"] = df["chat_name"].astype(object).replace("", None)
    out["is_group_chat"] = derive_is_group_chat(df["chat_identifier"])
    out["is_from_me"] = df["is_from_me"].astype("bool")
    out["suggested_label"] = pd.Series([None] * len(out), index=out.index, dtype="object")

    out, unmatched = labeling.apply_annotations(out, annotations)
    out = out.sort_values("message_id").reset_index(drop=True)
    out = out[list(schema.CORPUS_COLUMNS)]
    schema.validate_corpus(out)

    report = BuildReport(
        loaded=loaded,
        removed_null_text=int(null_mask.sum()),
        removed_empty_text=int(empty_mask.sum()),
        written=len(out),
        exact_duplicate_texts=int(out["text_hash"].duplicated(keep=False).sum()),
        group_chat_messages=int(out["is_group_chat"].eq(True).sum()),
        unknown_chat_messages=int(out["is_group_chat"].isna().sum()),
        unmatched_annotations=unmatched,
        labeled=int(out["label"].notna().sum()),
    )
    return out, report


def log_report(report: BuildReport) -> None:
    """Log aggregate build statistics."""
    logger.info("Loaded %d messages.", report.loaded)
    logger.info(
        "Removed %d null-text and %d empty-text records.",
        report.removed_null_text,
        report.removed_empty_text,
    )
    logger.info(
        "%d records share their normalized text with another record (kept; not deduplicated).",
        report.exact_duplicate_texts,
    )
    logger.info(
        "%d group-chat records, %d with unknown chat.",
        report.group_chat_messages,
        report.unknown_chat_messages,
    )
    logger.info("%d human-labeled records.", report.labeled)
    if report.unmatched_annotations:
        logger.warning(
            "%d annotations did not match any corpus record.", report.unmatched_annotations
        )
    logger.info("Wrote %d corpus records.", report.written)


def build_corpus_from_files(
    raw_path: Path = DEFAULT_RAW_PATH,
    out_path: Path = DEFAULT_CORPUS_PATH,
    annotations_path: Path | None = None,
) -> BuildReport:
    """Read the raw Parquet file (never modified), build the corpus, write Parquet."""
    raw_path, out_path = Path(raw_path), Path(out_path)
    if raw_path.resolve() == out_path.resolve():
        raise ValueError("output path must differ from the raw input path")
    raw = pd.read_parquet(raw_path)
    annotations = labeling.load_annotations(annotations_path) if annotations_path else None
    corpus, report = build_corpus(raw, annotations)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_path, index=False)
    log_report(report)
    return report


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (``veto-build-corpus``)."""
    parser = argparse.ArgumentParser(description="Build the private Veto corpus.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--annotations", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_corpus_from_files(args.raw, args.out, args.annotations)


if __name__ == "__main__":
    main()
