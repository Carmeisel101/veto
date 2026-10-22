import json

import pytest

from veto.inference import VetoClassifier


def test_predict_schema_and_threshold():
    clf = VetoClassifier(lambda texts: [0.9 if "vote" in t else 0.1 for t in texts], 0.5)
    hit, miss = clf.predict("please vote"), clf.predict("hello")
    assert set(hit) == {"label", "probability", "threshold", "should_filter"}
    assert hit["label"] == "political_solicitation" and hit["should_filter"] is True
    assert miss["label"] == "not_political_solicitation" and miss["should_filter"] is False


def test_threshold_behavior_inclusive():
    clf = VetoClassifier(lambda t: [0.7] * len(t), 0.7)
    assert clf.predict("x")["should_filter"] is True
    assert VetoClassifier(lambda t: [0.7] * len(t), 0.71).predict("x")["should_filter"] is False


def test_invalid_threshold():
    with pytest.raises(ValueError):
        VetoClassifier(lambda t: [], 1.5)


def test_roundtrip_tiny_finetuned_model(tmp_path):
    """Serialization round trip with a tiny randomly initialised BERT (no downloads)."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from veto.inference.persistence import save_finetuned

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "vote", "hello", "today"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, num_labels=2)
    model = BertForSequenceClassification(cfg)
    out = tmp_path / "m"
    save_finetuned(model, tok, out, {"threshold": 0.3, "hyperparameters": {"max_length": 16}})
    assert json.loads((out / "metadata.json").read_text())["model_type"] == "finetuned"
    clf = VetoClassifier.load(out, device="cpu")
    r = clf.predict("vote today")
    assert r["threshold"] == 0.3 and 0.0 <= r["probability"] <= 1.0
    assert clf.predict_proba(["vote today"]) == clf.predict_proba(["vote today"])
