"""Public inference interface, independent of Hugging Face / sklearn details."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from veto.dataset.labels import NEGATIVE_NAME, POSITIVE_NAME

DEFAULT_MODEL_DIR = Path("models/veto")
METADATA_FILE = "metadata.json"


class VetoClassifier:
    """Scores messages for political solicitation.

    ``scorer`` maps a list of texts to a list of positive-class probabilities. Use
    :meth:`load` to restore a saved model.
    """

    def __init__(self, scorer: Callable[[Sequence[str]], Sequence[float]], threshold: float,
                 metadata: dict | None = None) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._scorer = scorer
        self.threshold = float(threshold)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, model_dir: Path = DEFAULT_MODEL_DIR, device: str | None = None):
        """Load a model saved by ``veto train``."""
        from veto.inference.persistence import load_scorer

        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / METADATA_FILE).read_text())
        scorer = load_scorer(model_dir, metadata, device)
        return cls(scorer, metadata["threshold"], metadata)

    def predict_proba(self, messages: Sequence[str]) -> list[float]:
        return [float(p) for p in self._scorer(list(messages))]

    def predict(self, message: str) -> dict:
        """Classify one message."""
        return self.predict_batch([message])[0]

    def predict_batch(self, messages: Sequence[str]) -> list[dict]:
        out = []
        for p in self.predict_proba(messages):
            flag = p >= self.threshold
            out.append({
                "label": POSITIVE_NAME if flag else NEGATIVE_NAME,
                "probability": round(p, 4),
                "threshold": self.threshold,
                "should_filter": bool(flag),
            })
        return out
