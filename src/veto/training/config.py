"""Configuration and reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TrainConfig:
    """All hyperparameters in one place. Conservative defaults for tiny datasets."""

    base_model: str = BASE_MODEL
    seed: int = 42
    max_length: int = 256
    epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    max_pos_weight: float = 10.0
    baseline_C: float = 1.0
    target_precision: float = 0.95
    cv_folds: int = 3
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def detect_device() -> str:
    """'mps' when available, else 'cpu'."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
