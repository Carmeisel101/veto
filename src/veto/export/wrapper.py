"""A single traceable torch module that reproduces Veto's inference path.

``VetoCoreMLModel`` maps ``input_ids`` and ``attention_mask`` (fixed shape ``[1, L]``) to the
probability of ``political_solicitation``, using the *already trained* parameters of the saved
Veto model. Nothing is retrained.

Two source models are supported, matching ``models/veto/metadata.json``:

* ``baseline``: MiniLM encoder -> mean pooling -> L2 normalization -> logistic regression
  (mirrors ``SentenceTransformer.encode(..., normalize_embeddings=True)`` followed by
  ``LogisticRegression.predict_proba``).
* ``finetuned``: MiniLM encoder -> pooler -> 2-class head -> softmax
  (mirrors ``BertForSequenceClassification``).

The encoder forward pass is written out explicitly from the trained layers instead of calling the
Hugging Face ``BertModel`` forward. The stock forward builds attention masks with data-dependent
Python integer ops that ``torch.jit.trace`` / Core ML cannot represent. Because the explicit
version reuses the trained ``Linear`` and ``LayerNorm`` modules, it computes the same function,
and ``tests/export`` and ``verify_wrapper`` check that against the original implementation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MASK_FILL = -1.0e4  # additive attention mask; large enough to zero the softmax, safe in fp16


@dataclass
class SourceModel:
    """What was loaded from ``models/veto`` and the facts the exporter needs about it."""

    wrapper: VetoCoreMLModel
    metadata: dict
    model_type: str
    tokenizer_dir: Path
    weight_files: list[Path]


class VetoCoreMLModel(nn.Module):
    """``(input_ids, attention_mask) -> P(political_solicitation)``, shape ``[1]``."""

    def __init__(self, bert: nn.Module, head: str, classifier: nn.Module, seq_len: int) -> None:
        super().__init__()
        if head not in ("baseline", "finetuned"):
            raise ValueError(f"unknown head {head!r}")
        cfg = bert.config
        self.head, self.seq_len = head, seq_len
        self.hidden = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        if seq_len > cfg.max_position_embeddings:
            raise ValueError("seq_len exceeds the model's position embeddings")
        self.embeddings = bert.embeddings
        self.layers = bert.encoder.layer
        self.pooler = getattr(bert, "pooler", None) if head == "finetuned" else None
        self.classifier = classifier
        # Constant position ids and token types (single segment) for the fixed length.
        self.register_buffer("position_ids", torch.arange(seq_len).unsqueeze(0), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros(1, seq_len, dtype=torch.long),
                             persistent=False)

    # -- encoder ------------------------------------------------------------------------------
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Token embeddings ``[1, L, H]`` (equivalent to ``BertModel.last_hidden_state``)."""
        e = self.embeddings
        x = (e.word_embeddings(input_ids) + e.position_embeddings(self.position_ids)
             + e.token_type_embeddings(self.token_type_ids))
        x = e.LayerNorm(x)
        ext = (1.0 - attention_mask.to(x.dtype)).reshape(1, 1, 1, self.seq_len) * MASK_FILL
        for layer in self.layers:
            att = layer.attention
            q = self._split(att.self.query(x))
            k = self._split(att.self.key(x))
            v = self._split(att.self.value(x))
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) + ext
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
            ctx = ctx.permute(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden)
            x = att.output.LayerNorm(att.output.dense(ctx) + x)
            h = F.gelu(layer.intermediate.dense(x))
            x = layer.output.LayerNorm(layer.output.dense(h) + x)
        return x

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(1, self.seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    # -- full model ---------------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encode(input_ids, attention_mask)
        if self.head == "baseline":
            m = attention_mask.to(h.dtype).unsqueeze(-1)
            emb = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # mean pooling
            emb = F.normalize(emb, p=2.0, dim=-1)
            return torch.sigmoid(self.classifier(emb)).reshape(1)
        pooled = torch.tanh(self.pooler.dense(h[:, 0]))
        logits = self.classifier(pooled)
        return torch.sigmoid(logits[:, 1] - logits[:, 0]).reshape(1)  # == softmax(logits)[:, 1]


def sha256_of(paths: list[Path]) -> str:
    """Stable digest of the source weight files (for traceability in metadata)."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_source_model(model_dir: Path, seq_len: int | None = None) -> SourceModel:
    """Load the model saved by ``veto train`` and build the export wrapper (eval mode)."""
    from transformers import AutoModel, AutoModelForSequenceClassification

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    kind = metadata["model_type"]
    seq_len = seq_len or int(metadata["hyperparameters"]["max_length"])

    if kind == "baseline":
        import joblib

        enc_dir = model_dir / "model" / "encoder"
        bert = AutoModel.from_pretrained(enc_dir, attn_implementation="eager").eval()
        lr = joblib.load(model_dir / "model" / "logreg.joblib")
        if lr.coef_.shape != (1, bert.config.hidden_size):
            raise ValueError(f"unexpected classifier shape {lr.coef_.shape}")
        fc = nn.Linear(bert.config.hidden_size, 1)
        fc.weight.data = torch.tensor(lr.coef_, dtype=torch.float32)
        fc.bias.data = torch.tensor(lr.intercept_, dtype=torch.float32)
        weights = [enc_dir / "model.safetensors", model_dir / "model" / "logreg.joblib"]
        tok_dir = enc_dir
    elif kind == "finetuned":
        full = AutoModelForSequenceClassification.from_pretrained(
            model_dir / "model", attn_implementation="eager").eval()
        bert, fc = full.bert, full.classifier
        weights = [model_dir / "model" / "model.safetensors"]
        tok_dir = model_dir / "tokenizer"
    else:
        raise ValueError(f"unknown model_type {kind!r}")

    wrapper = VetoCoreMLModel(bert, kind, fc, seq_len).eval()
    return SourceModel(wrapper, metadata, kind, tok_dir, weights)
