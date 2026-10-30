"""Python <-> Core ML parity: the exported model must behave like Veto's Python inference.

This is a deployment-equivalence test, not an evaluation of a new model. Reports contain only
aggregate numbers and indices; message text is never written to them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veto.export import tokenizer as tokmod
from veto.export.coreml import INPUT_NAMES, OUTPUT_NAME
from veto.export.fixtures import FIXTURES, load_fixtures  # noqa: F401  (re-exported)

#: Maximum allowed |python - coreml| probability difference on the committed fixture suite.
#: FP16: observed maxima were 0.0071 (fixtures), 0.0061 (3,614 local messages) and 0.0036
#: (near-threshold band), so 0.01 leaves about 1.4x headroom while still being ~100x tighter
#: than the error a real conversion bug (wrong pooling, mask or weights) would produce.
#: FP32: the observed maximum was 1e-6, so 1e-4 is a loose-but-meaningful bound.
PROB_TOLERANCE_FP16 = 0.01
PROB_TOLERANCE_FP32 = 1e-4


def load_mlmodel(package: Path, compute_units: str = "ALL"):
    import coremltools as ct

    return ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, compute_units))


_CLASSIFIERS: dict = {}


def python_probabilities(model_dir: Path, texts: list[str]) -> np.ndarray:
    """Probabilities from Veto's existing Python inference API (the reference)."""
    from veto.inference import VetoClassifier

    key = str(Path(model_dir).resolve())
    if key not in _CLASSIFIERS:
        _CLASSIFIERS[key] = VetoClassifier.load(Path(model_dir), device="cpu")
    clf = _CLASSIFIERS[key]
    out: list[float] = []
    for i in range(0, len(texts), 64):
        out.extend(clf.predict_proba(texts[i:i + 64]))
    return np.asarray(out, dtype=np.float64)


def coreml_probabilities(mlmodel, tokenizer, texts: list[str], seq_len: int) -> np.ndarray:
    ids, mask = tokmod.encode_fixed(tokenizer, texts, seq_len)
    out = np.empty(len(texts), dtype=np.float64)
    for i in range(len(texts)):
        pred = mlmodel.predict({INPUT_NAMES[0]: ids[i:i + 1], INPUT_NAMES[1]: mask[i:i + 1]})
        out[i] = float(np.asarray(pred[OUTPUT_NAME]).reshape(-1)[0])
    return out


def compare(py: np.ndarray, ml: np.ndarray, threshold: float) -> dict:
    """Probability and decision agreement between two score arrays."""
    diff = np.abs(py - ml)
    py_flag, ml_flag = py >= threshold, ml >= threshold
    flipped = np.flatnonzero(py_flag != ml_flag)
    max_diff = float(diff.max()) if len(py) else 0.0
    margins = [float(abs(py[i] - threshold)) for i in flipped]
    return {
        "n": int(len(py)),
        "max_abs_diff": float(diff.max()) if len(py) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(py) else 0.0,
        "median_abs_diff": float(np.median(diff)) if len(py) else 0.0,
        "p95_abs_diff": float(np.percentile(diff, 95)) if len(py) else 0.0,
        "classification_disagreements": int(len(flipped)),
        "disagreement_indices": [int(i) for i in flipped],
        "disagreement_margins": margins,  # |python probability - threshold| for each flip
        # A flip is "explained" when the message sits closer to the threshold than the largest
        # probability deviation observed: no reduced-precision model can resolve such cases.
        "disagreements_within_precision_noise": int(sum(m <= max_diff for m in margins)),
        "python_positive": int(py_flag.sum()),
        "coreml_positive": int(ml_flag.sum()),
    }


def threshold_band_texts(model_dir: Path, threshold: float, per_band: int = 25,
                         pool_size: int = 1500, seed: int = 20261029) -> dict[str, list[str]]:
    """Pick synthetic messages whose *Python* probability lies below, near and above threshold.

    Candidates are generated on the fly from the synthetic generator (invented text), scored by
    the Python model, then bucketed:

    * ``below``: probability <= threshold - 0.25
    * ``near``: |probability - threshold| <= 0.15 (closest first)
    * ``above``: probability >= threshold + 0.05
    """
    from veto.synthetic import generate_synthetic

    raw, _ = generate_synthetic(pool_size, 0.30, seed=seed)
    texts = [t for t in raw["text"].tolist() if t.strip("￼� \n")]
    p = python_probabilities(model_dir, texts)
    order_near = np.argsort(np.abs(p - threshold))
    bands = {
        "below": [i for i in np.argsort(-p) if p[i] <= threshold - 0.25][:per_band],
        "near": [i for i in order_near if abs(p[i] - threshold) <= 0.15][:per_band],
        "above": [i for i in np.argsort(-p) if p[i] >= threshold + 0.05][:per_band],
    }
    return {k: [texts[i] for i in v] for k, v in bands.items()}


def local_corpus_sample(labeled_path: Path, n: int | None, seed: int = 42):
    """Validation+test messages from the private labeled dataset (local only, never committed)."""
    import pandas as pd

    df = pd.read_parquet(labeled_path)
    df = df[df["split"].isin(["validation", "test"]) & df["label"].notna()]
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df["normalized_text"].tolist(), df["label"].astype(int).to_numpy()


def error_counts(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    flag = probs >= threshold
    return {"false_positives": int((flag & (labels == 0)).sum()),
            "false_negatives": int((~flag & (labels == 1)).sum())}


def run_parity(model_dir: Path, package: Path, tokenizer_dir: Path, seq_len: int,
               threshold: float, tolerance: float, *, local_labeled: Path | None = None,
               local_n: int | None = None, compute_units: str = "ALL") -> dict:
    """Run the fixture, threshold-band and (optionally) local-corpus parity checks."""
    tok = tokmod.load_tokenizer(tokenizer_dir)
    ml = load_mlmodel(package, compute_units)

    fixtures = load_fixtures()
    texts = [f["text"] for f in fixtures]
    py = python_probabilities(model_dir, texts)
    cm = coreml_probabilities(ml, tok, texts, seq_len)
    report: dict = {"compute_units": compute_units, "tolerance": tolerance, "threshold": threshold}
    fx = compare(py, cm, threshold)
    cats = sorted({f["category"] for f in fixtures})
    fx["categories"] = cats
    fx["passes"] = fx["classification_disagreements"] == 0 and fx["max_abs_diff"] <= tolerance
    report["fixtures"] = fx

    bands = threshold_band_texts(model_dir, threshold)
    report["threshold_bands"] = {}
    for name, band_texts in bands.items():
        if not band_texts:
            report["threshold_bands"][name] = {"n": 0}
            continue
        bp = python_probabilities(model_dir, band_texts)
        bc = coreml_probabilities(ml, tok, band_texts, seq_len)
        s = compare(bp, bc, threshold)
        s["python_prob_range"] = [float(bp.min()), float(bp.max())]
        report["threshold_bands"][name] = s

    if local_labeled is not None and Path(local_labeled).exists():
        ltexts, labels = local_corpus_sample(local_labeled, local_n)
        lp = python_probabilities(model_dir, ltexts)
        lc = coreml_probabilities(ml, tok, ltexts, seq_len)
        s = compare(lp, lc, threshold)
        s["python_errors"] = error_counts(labels, lp, threshold)
        s["coreml_errors"] = error_counts(labels, lc, threshold)
        report["local_corpus"] = s
    return report


def verdict(report: dict) -> str:
    """PASS, PASS_WITH_NOTES or FAIL for one ``run_parity`` report."""
    fx = report["fixtures"]
    if not fx["passes"]:
        return "FAIL"
    notes = False
    sections = list(report.get("threshold_bands", {}).values())
    if "local_corpus" in report:
        sections.append(report["local_corpus"])
    for s in sections:
        flips = s.get("classification_disagreements", 0)
        if flips:
            if flips > s.get("disagreements_within_precision_noise", 0):
                return "FAIL"
            notes = True
    return "PASS_WITH_NOTES" if notes else "PASS"
