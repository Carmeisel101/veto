"""Binary political-solicitation labels merged onto the corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from veto.corpus.schema import ValidationError, require_columns

POSITIVE_NAME = "political_solicitation"
NEGATIVE_NAME = "not_political_solicitation"
LABEL_NAMES = {1: POSITIVE_NAME, 0: NEGATIVE_NAME}

SUBTYPES = {
    POSITIVE_NAME: (
        "campaign",
        "fundraising",
        "advocacy",
        "election_information",
        "polling_or_survey",
        "petition",
        "political_event",
        "political_other",
    ),
    NEGATIVE_NAME: ("personal", "transactional", "commercial", "nonpolitical_spam", "other"),
}

DEFAULT_ANNOTATIONS_PATH = Path("data/annotations/solicitation_labels.csv")
ANNOTATION_COLUMNS = ("message_id", "label", "subtype", "review_required", "review_reason")


def load_solicitation_annotations(path: Path) -> pd.DataFrame:
    """Load and validate the binary-label annotation table (CSV or Parquet)."""
    path = Path(path)
    ann = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    validate_solicitation_annotations(ann)
    return ann


def validate_solicitation_annotations(ann: pd.DataFrame) -> None:
    require_columns(ann, ANNOTATION_COLUMNS, "solicitation annotations")
    if ann["message_id"].isna().any() or ann["message_id"].duplicated().any():
        raise ValidationError("annotation message_id must be non-null and unique")
    review = ann["review_required"].astype(bool)
    decided = ann.loc[~review]
    bad = sorted({str(v) for v in decided["label"].unique() if v not in (0, 1, "0", "1", 0.0, 1.0)})
    if bad or decided["label"].isna().any():
        raise ValidationError(f"non-review rows need label 0 or 1; found {bad or 'nulls'}")
    for _, row in decided.iterrows():
        name = LABEL_NAMES[int(row["label"])]
        if row["subtype"] not in SUBTYPES[name]:
            raise ValidationError(f"subtype {row['subtype']!r} invalid for {name}")
    undocumented = review & ann["review_reason"].fillna("").eq("")
    if undocumented.any():
        raise ValidationError("review_required rows must carry a review_reason")


def build_labeled_dataset(corpus: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """Left-join annotations onto the corpus; every corpus message must be annotated."""
    validate_solicitation_annotations(ann)
    missing = set(corpus["message_id"]) - set(ann["message_id"])
    if missing:
        raise ValidationError(f"{len(missing)} corpus messages have no annotation")
    a = ann.assign(review_required=ann["review_required"].astype(bool))
    a["label"] = pd.to_numeric(a["label"], errors="coerce").astype("Int64")
    a.loc[a["review_required"], ["label", "subtype"]] = [pd.NA, pd.NA]
    # The corpus's own label columns use the earlier political/non_political/uncertain scheme;
    # they are superseded here by the binary solicitation labels.
    legacy = ["label", "label_source", "label_confidence", "suggested_label"]
    base = corpus.drop(columns=[c for c in legacy if c in corpus.columns])
    out = base.merge(a.drop(columns=["label_source"], errors="ignore"),
                     on="message_id", how="left", validate="one_to_one")
    out["label_name"] = out["label"].map(LABEL_NAMES)
    if "label_source" in ann:
        out["label_source"] = out["message_id"].map(ann.set_index("message_id")["label_source"])
    return out


def trainable(labeled: pd.DataFrame) -> pd.DataFrame:
    """Rows with a resolved label (review_required excluded)."""
    return labeled[~labeled["review_required"] & labeled["label"].notna()].reset_index(drop=True)
