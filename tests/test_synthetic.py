import pandas as pd
import pytest

from veto.corpus import build_corpus
from veto.dataset.labels import build_labeled_dataset, trainable, validate_solicitation_annotations
from veto.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def data():
    return generate_synthetic(1000, 0.10, seed=42)


def test_size_and_positive_rate(data):
    raw, ann = data
    assert len(raw) == len(ann) == 1000
    assert (ann["label"] == 1).sum() == 100


def test_deterministic(data):
    raw2, ann2 = generate_synthetic(1000, 0.10, seed=42)
    pd.testing.assert_frame_equal(data[0], raw2)
    pd.testing.assert_frame_equal(data[1], ann2)
    raw3, _ = generate_synthetic(1000, 0.10, seed=7)
    assert not data[0]["text"].equals(raw3["text"])


def test_annotations_valid_and_pipeline_compatible(data):
    raw, ann = data
    validate_solicitation_annotations(ann)
    corpus, report = build_corpus(raw)
    labeled = build_labeled_dataset(corpus, ann)
    ok = trainable(labeled)
    assert report.removed_empty_text == 0 and len(corpus) == 1000
    assert ok["label"].sum() == 100
    assert 0 < labeled["review_required"].sum() < 40


def test_all_invented(data):
    raw, _ = data
    senders = raw["sender"].astype(str)
    phones = senders[senders.str.startswith("+")]
    assert phones.str.contains(r"^\+1\d{3}5550\d{3}$").all()  # reserved fictional range
    urls = raw["text"].str.findall(r"\b[\w-]+\.example/\w+").explode().dropna()
    assert len(urls) > 0
    assert not raw["text"].str.contains(r"https?://|\.com\b", regex=True).any()


def test_other_sizes():
    raw, ann = generate_synthetic(200, 0.10, seed=1)
    assert len(raw) == 200 and (ann["label"] == 1).sum() == 20
