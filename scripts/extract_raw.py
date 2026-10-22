#!/usr/bin/env python
"""Generate data/raw/data.parquet from the macOS Messages database.

Reads ~/Library/Messages/chat.db read-only (the terminal needs Full Disk Access).
The raw file is treated as immutable, so an existing output is only replaced with --overwrite.
Output is private: it lives under data/, which is gitignored.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from veto.extraction.messages import DEFAULT_DB_PATH, DEFAULT_TIMEZONE, extract_raw_messages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="path to chat.db")
    p.add_argument("--out", type=Path, default=Path("data/raw/data.parquet"))
    p.add_argument("--months", type=int, default=6,
                   help="keep the last N months of messages; 0 keeps everything (default 6)")
    p.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    p.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.out.exists() and not args.overwrite:
        p.error(f"{args.out} exists; raw data is immutable. Pass --overwrite to replace it.")

    df = extract_raw_messages(args.db, months=args.months, timezone=args.timezone)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logging.getLogger(__name__).info("Saved %d messages to %s.", len(df), args.out)


if __name__ == "__main__":
    main()
