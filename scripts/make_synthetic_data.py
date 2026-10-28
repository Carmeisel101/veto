#!/usr/bin/env python
"""Generate the synthetic example dataset (no private data; everything is invented).

Writes examples/synthetic/raw.parquet and examples/synthetic/solicitation_labels.csv.
The output is deterministic for a given seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from veto.synthetic import generate_synthetic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("examples/synthetic"))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--positive-rate", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw, ann = generate_synthetic(args.n, args.positive_rate, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.out_dir / "raw.parquet", index=False)
    ann.to_csv(args.out_dir / "solicitation_labels.csv", index=False)
    pos = int((ann["label"] == 1).sum())
    print(f"Wrote {len(raw)} synthetic messages ({pos} positive) to {args.out_dir}/")


if __name__ == "__main__":
    main()
