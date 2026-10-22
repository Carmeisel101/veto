import pandas as pd
import pytest

from veto.corpus import ValidationError
from veto.dataset.grouping import assign_groups, dedup_key_text
from veto.dataset.labels import build_labeled_dataset, trainable, validate_solicitation_annotations
from veto.dataset.report import build_dataset_report
from veto.dataset.splits import grouped_stratified_folds, train_val_test_split


def _ann(rows):
    return pd.DataFrame(rows, columns=["message_id", "label", "subtype", "review_required",
                                       "review_reason"])


def _corpus(n=3):
    return pd.DataFrame({"message_id": range(1, n + 1),
                         "text": [f"t{i}" for i in range(n)],
                         "normalized_text": [f"t{i}" for i in range(n)],
                         "text_hash": [f"h{i}" for i in range(n)],
                         "label": [None] * n})


def test_label_validation_accepts_good():
    validate_solicitation_annotations(_ann([(1, 1, "campaign", False, ""),
                                            (2, 0, "personal", False, ""),
                                            (3, "", "", True, "ambiguous")]))


@pytest.mark.parametrize("row", [(1, 2, "campaign", False, ""),
                                 (1, 1, "personal", False, ""),
                                 (1, "", "", True, "")])
def test_label_validation_rejects_bad(row):
    with pytest.raises(ValidationError):
        validate_solicitation_annotations(_ann([row]))


def test_duplicate_annotation_ids_rejected():
    with pytest.raises(ValidationError):
        validate_solicitation_annotations(_ann([(1, 0, "personal", False, ""),
                                                (1, 1, "campaign", False, "")]))


def test_build_labeled_and_trainable_excludes_review():
    ann = _ann([(1, 1, "campaign", False, ""), (2, 0, "personal", False, ""),
                (3, "", "", True, "no content")])
    df = build_labeled_dataset(_corpus(), ann)
    assert df.set_index("message_id")["label_name"][1] == "political_solicitation"
    assert sorted(trainable(df)["message_id"]) == [1, 2]


def test_unannotated_corpus_rows_rejected():
    with pytest.raises(ValidationError, match="no annotation"):
        build_labeled_dataset(_corpus(3), _ann([(1, 0, "personal", False, "")]))


def test_dedup_key_masks_urls_phones_numbers():
    a = dedup_key_text("Vote now! https://x.co/AbC123 call 555-123-4567 by 10")
    b = dedup_key_text("vote  NOW! https://x.co/zzz999 call 555-987-6543 by 12")
    assert a == b


def test_near_duplicates_grouped_and_distinct_kept_apart():
    texts = pd.Series([
        "Carlos we need your help to defeat Smith this November chip in today https://a.co/1",
        "Maria we need your help to defeat Smith this November chip in today https://a.co/2",
        "what time are we meeting for dinner tonight",
        "ok",
        "ok",
    ])
    g = assign_groups(texts)
    assert g[0] == g[1] and g[3] == g[4]
    assert len({g[0], g[2], g[3]}) == 3


def _split_df():
    n = 60
    df = pd.DataFrame({"label": [1 if i % 12 == 0 else 0 for i in range(n)],
                       "group_id": [i // 2 for i in range(n)]})
    return df


def test_splits_deterministic_and_group_safe():
    df = _split_df()
    a, b = train_val_test_split(df, 7), train_val_test_split(df, 7)
    assert a.equals(b)
    per_group = df.assign(s=a).groupby("group_id")["s"].nunique()
    assert (per_group == 1).all()
    assert set(a) == {"train", "validation", "test"}


def test_splits_distribute_positives():
    s = train_val_test_split(_split_df(), 1)
    pos = _split_df()["label"] == 1
    assert {"validation", "test"} <= set(s[pos])


def test_folds_cover_all_rows():
    f = grouped_stratified_folds(_split_df(), 3, 0)
    assert f.notna().all() and set(f) == {0, 1, 2}


def test_report_counts():
    ann = _ann([(1, 1, "campaign", False, ""), (2, 0, "personal", False, ""),
                (3, "", "", True, "no content")])
    df = build_labeled_dataset(_corpus(), ann)
    df["group_id"] = [0, 1, 2]
    rep = build_dataset_report(df)
    assert (rep["total_messages"], rep["labeled"], rep["review_required"]) == (3, 2, 1)
    assert (rep["positive_count"], rep["negative_count"]) == (1, 1)
    assert rep["warnings"]
