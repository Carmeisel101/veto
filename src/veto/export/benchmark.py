"""Local latency benchmark (development machine; not a substitute for on-device numbers)."""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from veto.export import tokenizer as tokmod
from veto.export.coreml import INPUT_NAMES
from veto.export.parity import load_fixtures, load_mlmodel


def _stats(ms: list[float]) -> dict:
    return {"median_ms": round(statistics.median(ms), 3),
            "p95_ms": round(float(np.percentile(ms, 95)), 3), "n": len(ms)}


def machine_description() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        chip = platform.processor() or "unknown"
    return f"{chip} ({platform.platform()})"


def benchmark(package: Path, tokenizer_dir: Path, seq_len: int, *, runs: int = 200,
              warmup: int = 20, compute_units: tuple[str, ...] = ("ALL", "CPU_ONLY")) -> dict:
    """Median/p95 latency for tokenization, Core ML inference and both together.

    Uses the committed fixture strings (cycled), after warm-up runs.
    """
    tok = tokmod.load_tokenizer(tokenizer_dir)
    texts = [f["text"] for f in load_fixtures()]
    cycle = [texts[i % len(texts)] for i in range(runs + warmup)]

    for t in cycle[:warmup]:
        tokmod.encode_fixed(tok, [t], seq_len)
    tok_ms = []
    for t in cycle[warmup:]:
        a = time.perf_counter()
        tokmod.encode_fixed(tok, [t], seq_len)
        tok_ms.append((time.perf_counter() - a) * 1000)

    result: dict = {"machine": machine_description(), "note": "development-machine measurement",
                    "runs": runs, "warmup": warmup, "tokenization": _stats(tok_ms)}
    for units in compute_units:
        ml = load_mlmodel(package, units)
        encoded = [tokmod.encode_fixed(tok, [t], seq_len) for t in cycle]
        for ids, mask in encoded[:warmup]:
            ml.predict({INPUT_NAMES[0]: ids, INPUT_NAMES[1]: mask})
        infer_ms, total_ms = [], []
        for t, (ids, mask) in zip(cycle[warmup:], encoded[warmup:], strict=True):
            a = time.perf_counter()
            tokmod.encode_fixed(tok, [t], seq_len)
            b = time.perf_counter()
            ml.predict({INPUT_NAMES[0]: ids, INPUT_NAMES[1]: mask})
            c = time.perf_counter()
            infer_ms.append((c - b) * 1000)
            total_ms.append((c - a) * 1000)
        result[f"coreml_{units.lower()}"] = {"inference": _stats(infer_ms),
                                             "total_with_tokenization": _stats(total_ms)}
    return result
