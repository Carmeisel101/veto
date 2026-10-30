"""Shared constants and builders for the export tests (tiny offline models)."""

from transformers import BertConfig, BertTokenizerFast

SEQ = 16
WORDS = ["vote", "early", "donate", "now", "hello", "friend", "dinner", "tonight", "chip", "in",
         "the", "a", "to", "and", "stop", "reply", "campaign", "you", "me", "lol"]


def tiny_config() -> BertConfig:
    return BertConfig(vocab_size=5 + len(WORDS), hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
                      num_labels=2)


def tiny_tokenizer(tmp_path) -> BertTokenizerFast:
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    return BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=True, model_max_length=SEQ)
