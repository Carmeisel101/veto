"""Error analysis: per-mistake feature flags. Output is local/private (gitignored)."""

from __future__ import annotations

import re

import pandas as pd

_FLAGS = {
    "has_url": re.compile(r"https?://|www\.|\b[\w-]+\.(com|co|org|net|us)\b|\b\w+\.\w+/\S+", re.I),
    "has_stop_optout": re.compile(r"\b(stop|end|unsubscribe)\b", re.I),
    "has_donation_cue": re.compile(r"\b(donat\w*|contribut\w*|chip in|\$\d+)", re.I),
    "has_election_vocab": re.compile(
        r"\b(vote|voting|election|ballot|polls?|early voting|primary|candidate)\b", re.I),
    "has_political_name_or_office": re.compile(
        r"\b(senator|governor|comptroller|president|congress\w*|mayor|rep\.)\b", re.I),
}


def analyze_errors(df: pd.DataFrame, proba, threshold: float, split: str) -> list[dict]:
    """Return one record per misclassified row of ``df`` (needs message_id, label, text)."""
    out = []
    pred = (pd.Series(proba, index=df.index) >= threshold).astype(int)
    for idx, row in df.iterrows():
        if int(row["label"]) == int(pred[idx]):
            continue
        text = row["normalized_text"]
        rec = {
            "split": split,
            "message_id": int(row["message_id"]),
            "error_type": "false_positive" if pred[idx] == 1 else "false_negative",
            "true_subtype": row.get("subtype"),
            "probability": round(float(proba[df.index.get_loc(idx)]), 4),
            "char_length": len(text),
            "word_count": len(text.split()),
            "text": text,
        }
        rec.update({k: bool(rx.search(text)) for k, rx in _FLAGS.items()})
        out.append(rec)
    return out
