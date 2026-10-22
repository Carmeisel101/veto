"""Saving and loading model artifacts (finetuned transformer or embedding baseline)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from veto.inference.classifier import METADATA_FILE
from veto.training.config import TrainConfig, detect_device


def save_finetuned(model, tokenizer, model_dir: Path, metadata: dict) -> None:
    model_dir = Path(model_dir)
    model.save_pretrained(model_dir / "model")
    tokenizer.save_pretrained(model_dir / "tokenizer")
    _write_meta(model_dir, {**metadata, "model_type": "finetuned"})


def save_baseline(encoder, clf, model_dir: Path, metadata: dict) -> None:
    import joblib

    model_dir = Path(model_dir)
    (model_dir / "model").mkdir(parents=True, exist_ok=True)
    encoder.save(str(model_dir / "model" / "encoder"))
    joblib.dump(clf, model_dir / "model" / "logreg.joblib")
    _write_meta(model_dir, {**metadata, "model_type": "baseline"})


def _write_meta(model_dir: Path, metadata: dict) -> None:
    metadata = {"saved_at": datetime.now(UTC).isoformat(), **metadata}
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))


def load_scorer(model_dir: Path, metadata: dict, device: str | None = None):
    device = device or detect_device()
    kind = metadata["model_type"]
    if kind == "finetuned":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from veto.training.finetune import predict_proba

        tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model").to(device)
        cfg = TrainConfig(max_length=metadata["hyperparameters"]["max_length"])
        return lambda texts: predict_proba(model, tok, list(texts), device, cfg)
    if kind == "baseline":
        import joblib

        from veto.training.baseline import embed, load_encoder

        enc = load_encoder(str(model_dir / "model" / "encoder"), device)
        clf = joblib.load(model_dir / "model" / "logreg.joblib")
        return lambda texts: clf.predict_proba(embed(list(texts), enc))[:, 1]
    raise ValueError(f"unknown model_type {kind!r}")
