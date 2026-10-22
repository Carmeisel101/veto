"""Dataset report: aggregate statistics only (no message text)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_dataset_report(labeled: pd.DataFrame) -> dict:
    """Summarise a labeled dataset (must contain group_id after grouping)."""
    resolved = labeled[~labeled["review_required"] & labeled["label"].notna()]
    n_pos = int((resolved["label"] == 1).sum())
    n_neg = int((resolved["label"] == 0).sum())
    lengths = labeled["normalized_text"].str.len()
    report = {
        "total_messages": len(labeled),
        "labeled": len(resolved),
        "review_required": int(labeled["review_required"].sum()),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "positive_pct": round(100 * n_pos / max(len(resolved), 1), 2),
        "negative_pct": round(100 * n_neg / max(len(resolved), 1), 2),
        "subtype_counts": {k: int(v) for k, v in resolved["subtype"].value_counts().items()},
        "review_reasons": {
            k: int(v) for k, v in labeled.loc[labeled["review_required"], "review_reason"]
            .value_counts().items()
        },
        "duplicates": {
            "messages_sharing_exact_normalized_text": int(
                labeled["text_hash"].duplicated(keep=False).sum()
            ),
            "duplicate_groups_with_multiple_messages": int(
                (labeled.groupby("group_id").size() > 1).sum()
            ) if "group_id" in labeled else None,
            "messages_in_multi_message_groups": int(
                labeled["group_id"].duplicated(keep=False).sum()
            ) if "group_id" in labeled else None,
        },
        "null_or_empty_text": int(
            (labeled["text"].isna() | labeled["normalized_text"].fillna("").eq("")).sum()
        ),
        "message_length_chars": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": round(float(lengths.mean()), 1),
            "p90": float(lengths.quantile(0.9)),
            "max": int(lengths.max()),
        },
    }
    if "split" in labeled:
        report["splits"] = {
            s: {
                "n": int((resolved["split"] == s).sum()),
                "positives": int(((resolved["split"] == s) & (resolved["label"] == 1)).sum()),
            }
            for s in ("train", "validation", "test")
        }
    warnings = []
    if n_pos < 30:
        warnings.append(
            f"Only {n_pos} positive examples: all metrics are anecdotal, not statistically "
            "meaningful. Collect more political-outreach messages before trusting results."
        )
    report["warnings"] = warnings
    return report


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Dataset report", ""]
    for k, v in report.items():
        lines.append(f"- **{k}**: `{json.dumps(v)}`")
    (out_dir / "dataset_report.md").write_text("\n".join(lines) + "\n")
