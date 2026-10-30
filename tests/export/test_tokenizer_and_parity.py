import json

import numpy as np
import pytest
from export_tiny import SEQ

from veto.export import parity
from veto.export.tokenizer import (
    describe_tokenizer,
    encode_fixed,
    export_tokenizer_assets,
    load_tokenizer,
)


def test_encode_fixed_shapes_dtype_padding_and_specials(tokenizer):
    ids, mask = encode_fixed(tokenizer, ["Vote EARLY now", "hello"], SEQ)
    assert ids.shape == mask.shape == (2, SEQ) and ids.dtype == mask.dtype == np.int32
    assert ids[0, 0] == tokenizer.cls_token_id
    n = int(mask[0].sum())
    assert ids[0, n - 1] == tokenizer.sep_token_id and n == 5  # [CLS] vote early now [SEP]
    assert (ids[0, n:] == tokenizer.pad_token_id).all() and (mask[0, n:] == 0).all()
    # Lower-casing happens inside the tokenizer.
    assert (ids[0] == encode_fixed(tokenizer, ["vote early now"], SEQ)[0][0]).all()


def test_truncation_keeps_cls_and_sep(tokenizer):
    ids, mask = encode_fixed(tokenizer, ["vote " * 100], SEQ)
    assert mask.sum() == SEQ and ids[0, 0] == tokenizer.cls_token_id
    assert ids[0, SEQ - 1] == tokenizer.sep_token_id


def test_unknown_characters_map_to_unk(tokenizer):
    ids, _ = encode_fixed(tokenizer, ["vote 🩷"], SEQ)
    assert tokenizer.unk_token_id in ids[0]


def test_tokenizer_assets_and_contract(tokenizer, tmp_path):
    src = tmp_path / "tok"
    tokenizer.save_pretrained(src)
    written = export_tokenizer_assets(src, tmp_path / "out")
    names = sorted(p.name for p in written)
    assert names == ["tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    vocab = (tmp_path / "out" / "vocab.txt").read_text().split("\n")[:-1]
    assert vocab[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    assert len(vocab) == tokenizer.vocab_size
    c = describe_tokenizer(tmp_path / "out", SEQ)
    assert c["lowercase"] is True and c["max_sequence_length"] == SEQ
    assert c["special_tokens"]["[CLS]"] == tokenizer.cls_token_id
    assert c["input_dtype"] == "int32"
    # The exported copy tokenizes identically to the original.
    again = load_tokenizer(tmp_path / "out")
    t = "Vote early 🩷 now"
    assert (encode_fixed(again, [t], SEQ)[0] == encode_fixed(tokenizer, [t], SEQ)[0]).all()


# -- parity helpers ------------------------------------------------------------------------------
def test_compare_statistics_and_noise_explanation():
    py = np.array([0.10, 0.50, 0.8700, 0.99, 0.30])
    ml = np.array([0.101, 0.502, 0.8670, 0.989, 0.30])
    s = parity.compare(py, ml, threshold=0.869)
    assert s["n"] == 5 and s["classification_disagreements"] == 1
    assert s["disagreement_indices"] == [2]
    assert s["max_abs_diff"] == pytest.approx(0.003)
    assert s["disagreements_within_precision_noise"] == 1  # margin 0.001 <= max diff 0.003
    assert s["mean_abs_diff"] == pytest.approx((0.001 + 0.002 + 0.003 + 0.001 + 0) / 5)


def test_verdict_logic():
    ok = {"fixtures": {"passes": True}, "threshold_bands": {}}
    assert parity.verdict(ok) == "PASS"
    assert parity.verdict({**ok, "fixtures": {"passes": False}}) == "FAIL"
    explained = {**ok, "local_corpus": {"classification_disagreements": 1,
                                        "disagreements_within_precision_noise": 1}}
    assert parity.verdict(explained) == "PASS_WITH_NOTES"
    unexplained = {**ok, "local_corpus": {"classification_disagreements": 2,
                                          "disagreements_within_precision_noise": 1}}
    assert parity.verdict(unexplained) == "FAIL"


def test_error_counts():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.2, 0.8, 0.1])
    assert parity.error_counts(labels, probs, 0.5) == {"false_positives": 1, "false_negatives": 1}


def test_committed_fixtures_cover_required_categories_and_are_invented():
    fixtures = parity.load_fixtures()
    cats = {f["category"] for f in fixtures}
    required = {
        "campaign_solicitation", "political_fundraising", "advocacy", "election_reminder",
        "candidate_update", "ballot_measure_advocacy", "personal_conversation",
        "personal_political_conversation", "transactional", "commercial_marketing",
        "ordinary_spam", "short_message", "long_message", "contains_url", "stop_end_language",
    }
    assert required <= cats and len(fixtures) >= 60
    blob = json.dumps(fixtures)
    assert "http://" not in blob and "https://" not in blob.replace("https://vote-now.example", "")
    import re
    phones = re.findall(r"\+?1?\d{3}[\s.-]?\d{3}[\s.-]?\d{4}", blob)
    assert all("5550" in p for p in phones)  # only reserved fictional numbers
