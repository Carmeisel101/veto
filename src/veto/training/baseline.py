"""Baseline: frozen MiniLM sentence embeddings -> logistic regression."""

from __future__ import annotations

import numpy as np

from veto.training.config import TrainConfig


def embed(texts: list[str], encoder, batch_size: int = 32) -> np.ndarray:
    return np.asarray(
        encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False)
    )


def load_encoder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def fit_baseline(X: np.ndarray, y: np.ndarray, cfg: TrainConfig):
    """Class-balanced logistic regression on embeddings."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=cfg.baseline_C, class_weight="balanced", max_iter=1000, random_state=cfg.seed
    )
    return clf.fit(X, y)
