import pandas as pd
import pytest

from veto.corpus import ValidationError, build_corpus
from veto.corpus.schema import LABELS, validate_labels


@pytest.mark.parametrize("label", ["political", "non_political", "uncertain"])
def test_supported_labels_accepted(label):
    validate_labels(pd.Series([label]))


@pytest.mark.parametrize("label", ["spam", "Political", "1", ""])
def test_unsupported_labels_rejected(label):
    with pytest.raises(ValidationError, match="unsupported"):
        validate_labels(pd.Series([label]))


def test_label_set():
    assert set(LABELS) == {"political", "non_political", "uncertain"}


def test_annotations_applied_as_human(raw):
    ann = pd.DataFrame({"message_id": [1, 99], "label": ["political", "uncertain"]})
    corpus, report = build_corpus(raw, ann)
    row = corpus.set_index("message_id").loc[1]
    assert row["label"] == "political" and row["label_source"] == "human"
    assert corpus["label"].notna().sum() == 1
    assert report.unmatched_annotations == 1


def test_unlabeled_by_default_and_no_suggestions(raw):
    corpus, _ = build_corpus(raw)
    assert corpus["label"].isna().all()
    assert corpus["label_source"].isna().all()
    assert corpus["suggested_label"].isna().all()


def test_invalid_annotation_label_fails(raw):
    ann = pd.DataFrame({"message_id": [1], "label": ["spam"]})
    with pytest.raises(ValidationError, match="unsupported"):
        build_corpus(raw, ann)


def test_duplicate_annotation_ids_fail(raw):
    ann = pd.DataFrame({"message_id": [1, 1], "label": ["political", "non_political"]})
    with pytest.raises(ValidationError, match="unique"):
        build_corpus(raw, ann)
