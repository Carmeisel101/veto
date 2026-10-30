"""The tokenizer contract: how text becomes ``input_ids`` and ``attention_mask``.

The contract is part of the deployed model. This module (a) produces the fixed-shape tensors the
Core ML model consumes, using the same Hugging Face tokenizer as Python inference, and (b) copies
the assets another implementation (for example Swift) needs to reproduce tokenization.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ASSET_FILES = ("tokenizer.json", "tokenizer_config.json")


def load_tokenizer(tokenizer_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir)


def encode_fixed(tokenizer, texts: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize to int32 ``input_ids`` and ``attention_mask`` of shape ``[n, seq_len]``.

    Truncates to ``seq_len`` tokens (CLS/SEP included, longest_first) and right-pads with the
    pad token. Padding is masked, so results match unpadded Python inference.
    """
    enc = tokenizer(list(texts), truncation=True, max_length=seq_len, padding="max_length",
                    return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


def describe_tokenizer(tokenizer_dir: Path, seq_len: int) -> dict:
    """Machine-readable summary of the tokenizer contract (for ``metadata.json``)."""
    tj = json.loads((Path(tokenizer_dir) / "tokenizer.json").read_text())
    tk = load_tokenizer(tokenizer_dir)
    norm = tj.get("normalizer") or {}
    return {
        "type": "BertTokenizer (WordPiece)",
        "implementation": "Hugging Face `tokenizers` fast tokenizer (tokenizer.json)",
        "vocab_size": len(tj["model"]["vocab"]),
        "lowercase": bool(norm.get("lowercase", False)),
        "strip_accents": norm.get("strip_accents"),  # null means: follow lowercase
        "clean_text": bool(norm.get("clean_text", False)),
        "handle_chinese_chars": bool(norm.get("handle_chinese_chars", False)),
        "unknown_token": "[UNK]",
        "unknown_note": "Characters outside the vocabulary, including emoji, map to [UNK] "
                        "(id 100).",
        "special_tokens": {
            "[PAD]": tk.pad_token_id, "[UNK]": tk.unk_token_id, "[CLS]": tk.cls_token_id,
            "[SEP]": tk.sep_token_id, "[MASK]": tk.mask_token_id,
        },
        "template": "[CLS] <tokens> [SEP]",
        "max_sequence_length": seq_len,
        "truncation": "longest_first, right side, special tokens included in the budget",
        "padding": f"right-pad to exactly {seq_len} tokens with [PAD] (id {tk.pad_token_id})",
        "attention_mask": "1 for real tokens (including [CLS] and [SEP]), 0 for padding",
        "token_type_ids": "not an input; the model uses a single segment (all zeros)",
        "input_dtype": "int32",
    }


def export_tokenizer_assets(tokenizer_dir: Path, dest: Path) -> list[Path]:
    """Copy the tokenizer files a consumer needs and write ``vocab.txt`` (id order)."""
    tokenizer_dir, dest = Path(tokenizer_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ASSET_FILES:
        shutil.copyfile(tokenizer_dir / name, dest / name)
        written.append(dest / name)
    vocab = json.loads((tokenizer_dir / "tokenizer.json").read_text())["model"]["vocab"]
    ordered = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    (dest / "vocab.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    written.append(dest / "vocab.txt")
    return written


def export_golden_tokenization(tokenizer, texts: list[str], seq_len: int, dest: Path) -> Path:
    """Write ``golden_tokenization.json``: texts with their exact (unpadded) ``input_ids``.

    A non-Python tokenizer implementation can use it to check that it reproduces Python
    tokenization exactly. The texts are the committed, invented parity fixtures.
    """
    ids, mask = encode_fixed(tokenizer, texts, seq_len)
    examples = []
    for text, row, m in zip(texts, ids, mask, strict=True):
        n = int(m.sum())
        examples.append({"text": text, "length": n, "input_ids": [int(x) for x in row[:n]]})
    path = Path(dest) / "golden_tokenization.json"
    path.write_text(json.dumps({
        "note": "Invented example texts with the exact token ids Python produces. Padding to "
                f"{seq_len} with id {tokenizer.pad_token_id} is omitted here (attention_mask is "
                "1 for exactly `length` positions).",
        "max_sequence_length": seq_len, "examples": examples}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    return path
