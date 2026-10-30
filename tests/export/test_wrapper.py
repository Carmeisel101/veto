import numpy as np
import pytest
import torch
from export_tiny import SEQ
from torch.nn import functional as F

from veto.export.tokenizer import encode_fixed
from veto.export.wrapper import VetoCoreMLModel, load_source_model, sha256_of

TEXTS = ["vote early now", "hello friend dinner tonight lol", "chip in to the campaign and stop",
         "reply stop " * 10]


def tensors(tokenizer, texts, seq=SEQ):
    ids, mask = encode_fixed(tokenizer, texts, seq)
    return torch.tensor(ids).long(), torch.tensor(mask).long()


def test_encoder_matches_huggingface(tiny_bert, tokenizer):
    """The explicit encoder forward equals BertModel.last_hidden_state at real tokens."""
    fc = torch.nn.Linear(32, 1)
    w = VetoCoreMLModel(tiny_bert, "baseline", fc, SEQ).eval()
    ids, mask = tensors(tokenizer, TEXTS)
    for i in range(len(TEXTS)):
        with torch.no_grad():
            mine = w.encode(ids[i:i + 1], mask[i:i + 1])
            ref = tiny_bert(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1]).last_hidden_state
        keep = mask[i].bool()
        assert torch.allclose(mine[0][keep], ref[0][keep], atol=1e-5)


def test_baseline_pooling_normalization_and_classifier(tiny_bert, tokenizer):
    """mean pool -> L2 normalize -> linear -> sigmoid, built independently from HF outputs."""
    torch.manual_seed(3)
    fc = torch.nn.Linear(32, 1)
    w = VetoCoreMLModel(tiny_bert, "baseline", fc, SEQ).eval()
    ids, mask = tensors(tokenizer, TEXTS)
    for i in range(len(TEXTS)):
        with torch.no_grad():
            out = w(ids[i:i + 1], mask[i:i + 1])
            h = tiny_bert(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1]).last_hidden_state
            m = mask[i:i + 1].unsqueeze(-1).float()
            emb = F.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            expected = torch.sigmoid(fc(emb)).reshape(1)
        assert torch.allclose(out, expected, atol=1e-5)
        assert out.shape == (1,) and 0.0 <= out.item() <= 1.0


def test_finetuned_head_matches_sequence_classifier(tokenizer):
    from export_tiny import tiny_config
    from transformers import BertForSequenceClassification

    torch.manual_seed(4)
    model = BertForSequenceClassification(tiny_config()).eval()
    w = VetoCoreMLModel(model.bert, "finetuned", model.classifier, SEQ).eval()
    ids, mask = tensors(tokenizer, TEXTS)
    for i in range(len(TEXTS)):
        with torch.no_grad():
            got = w(ids[i:i + 1], mask[i:i + 1]).item()
            ref = torch.softmax(model(input_ids=ids[i:i + 1],
                                      attention_mask=mask[i:i + 1]).logits, dim=-1)[0, 1].item()
        assert got == pytest.approx(ref, abs=1e-5)


def test_padding_length_does_not_change_the_result(tiny_bert, tokenizer):
    fc = torch.nn.Linear(32, 1)
    a = VetoCoreMLModel(tiny_bert, "baseline", fc, 16).eval()
    b = VetoCoreMLModel(tiny_bert, "baseline", fc, 32).eval()
    for text in TEXTS[:3]:  # these fit in 16 tokens, so only the padding differs
        with torch.no_grad():
            ia, ma = tensors(tokenizer, [text], 16)
            ib, mb = tensors(tokenizer, [text], 32)
            assert a(ia, ma).item() == pytest.approx(b(ib, mb).item(), abs=1e-5)


def test_load_baseline_transfers_classifier_weights(saved_baseline):
    import joblib

    src = load_source_model(saved_baseline)
    lr = joblib.load(saved_baseline / "model" / "logreg.joblib")
    assert src.model_type == "baseline" and src.wrapper.seq_len == SEQ
    assert np.allclose(src.wrapper.classifier.weight.detach().numpy(), lr.coef_, atol=1e-7)
    assert np.allclose(src.wrapper.classifier.bias.detach().numpy(), lr.intercept_, atol=1e-7)
    assert not src.wrapper.training


def test_load_baseline_matches_sklearn_on_embeddings(saved_baseline, tokenizer):
    """End to end: wrapper probability == sklearn predict_proba on the normalized mean embedding."""
    import joblib
    from transformers import AutoModel

    src = load_source_model(saved_baseline)
    lr = joblib.load(saved_baseline / "model" / "logreg.joblib")
    bert = AutoModel.from_pretrained(saved_baseline / "model" / "encoder").eval()
    ids, mask = tensors(tokenizer, TEXTS)
    for i in range(len(TEXTS)):
        with torch.no_grad():
            h = bert(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1]).last_hidden_state
            m = mask[i:i + 1].unsqueeze(-1).float()
            emb = F.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy()
            got = src.wrapper(ids[i:i + 1], mask[i:i + 1]).item()
        assert got == pytest.approx(lr.predict_proba(emb)[0, 1], abs=1e-5)


def test_load_finetuned_and_unknown_type(saved_finetuned, tmp_path):
    src = load_source_model(saved_finetuned)
    assert src.model_type == "finetuned" and src.wrapper.pooler is not None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metadata.json").write_text(
        '{"model_type": "mystery", "hyperparameters": {"max_length": 8}}')
    with pytest.raises(ValueError, match="unknown model_type"):
        load_source_model(bad)


def test_seq_len_cannot_exceed_position_embeddings(tiny_bert):
    with pytest.raises(ValueError, match="position embeddings"):
        VetoCoreMLModel(tiny_bert, "baseline", torch.nn.Linear(32, 1), 128)


def test_weight_hash_is_stable_and_sensitive(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    assert sha256_of([a, b]) == sha256_of([b, a])
    b.write_bytes(b"3")
    assert sha256_of([a, b]) != sha256_of([a])
