"""Label handling.

Ground-truth ``label`` values come only from human annotations. Veto does not
generate labels from rules or models; ``suggested_label`` exists as a separate,
initially empty column so that future tooling can propose labels without
contaminating ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from veto.corpus.schema import (
    ANNOTATION_COLUMNS,
    LABEL_SOURCE_HUMAN,
    ValidationError,
    require_columns,
    validate_labels,
)


def load_annotations(path: Path) -> pd.DataFrame:
    """Load human annotations from Parquet or CSV."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def validate_annotations(ann: pd.DataFrame) -> None:
    """Validate an annotation table (columns, labels, unique ids, confidence)."""
    require_columns(ann, ("message_id", "label"), "annotations")
    extra = [c for c in ann.columns if c not in ANNOTATION_COLUMNS]
    if extra:
        raise ValidationError(f"annotations has unexpected columns: {extra}")
    if ann["message_id"].isna().any() or ann["message_id"].duplicated().any():
        raise ValidationError("annotation message_id must be non-null and unique")
    validate_labels(ann["label"], "annotation label", allow_null=False)
    if "label_confidence" in ann and ann["label_confidence"].notna().any():
        conf = ann["label_confidence"].dropna()
        if ((conf < 0) | (conf > 1)).any():
            raise ValidationError("annotation label_confidence must be within [0, 1]")


def apply_annotations(corpus: pd.DataFrame, ann: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Attach human labels to ``corpus``; returns (corpus, number of unmatched annotations).

    Annotations for message_ids absent from the corpus are counted, not applied.
    """
    out = corpus.copy()
    out["label"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["label_source"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["label_confidence"] = pd.Series(float("nan"), index=out.index, dtype="float64")
    if ann is None or ann.empty:
        return out, 0
    validate_annotations(ann)
    ann = ann.copy()
    if "label_confidence" not in ann:
        ann["label_confidence"] = float("nan")
    ann = ann.set_index(ann["message_id"].astype("int64"))
    ids = out["message_id"].astype("int64")
    hit = ids.isin(ann.index)
    out.loc[hit, "label"] = ids[hit].map(ann["label"]).to_numpy()
    out.loc[hit, "label_source"] = LABEL_SOURCE_HUMAN
    out.loc[hit, "label_confidence"] = ids[hit].map(ann["label_confidence"]).to_numpy()
    return out, int((~ann.index.isin(ids)).sum())
