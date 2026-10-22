"""Decision-threshold selection (use validation data only)."""

from __future__ import annotations

import numpy as np


def threshold_table(y_true, proba) -> list[dict]:
    """Precision/recall at each distinct score (candidate thresholds), ascending."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    rows = []
    for t in sorted(set(p.tolist())):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        rows.append({
            "threshold": float(t),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "fp": fp,
            "fn": fn,
        })
    return rows


def select_threshold(y_true, proba, target_precision: float = 0.95,
                     default: float = 0.5) -> dict:
    """Pick the threshold with precision >= target and the highest recall.

    Among qualifying thresholds, prefer the one with highest recall, then the highest
    threshold (more conservative). If validation lacks both classes or nothing qualifies,
    fall back to ``default`` and say so in ``reason``.
    """
    y = np.asarray(y_true, dtype=int)
    if len(set(y.tolist())) < 2:
        return {"threshold": default, "reason": "validation lacks both classes; using default",
                "table": []}
    table = threshold_table(y, proba)
    ok = [r for r in table if r["precision"] >= target_precision and r["recall"] > 0]
    if not ok:
        return {"threshold": default,
                "reason": f"no threshold reaches precision >= {target_precision}; using default",
                "table": table}
    best = max(ok, key=lambda r: (r["recall"], r["threshold"]))
    # Midpoint to the next-lower score is overfit-prone; use the score itself.
    return {"threshold": best["threshold"],
            "reason": f"max recall subject to validation precision >= {target_precision}",
            "table": table}
