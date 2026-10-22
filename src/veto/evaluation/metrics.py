"""Binary classification metrics; safe on degenerate (single-class) inputs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Metrics at ``threshold``. ROC/PR-AUC are None when only one class is present."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_ids_count": int(fp),
        "false_negative_ids_count": int(fn),
    }
