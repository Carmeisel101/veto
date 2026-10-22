"""End-to-end orchestration: prepare dataset, baseline, fine-tune, compare, save."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from veto.corpus.builder import DEFAULT_CORPUS_PATH, build_corpus_from_files
from veto.dataset import grouping, labels, report, splits
from veto.evaluation import errors, metrics, threshold
from veto.inference import persistence
from veto.training import baseline, finetune
from veto.training.config import TrainConfig, detect_device, set_seed

logger = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
MODEL_DIR = Path("models/veto")
LABELED_PATH = Path("data/processed/labeled.parquet")


def prepare_dataset(cfg: TrainConfig, corpus_path: Path = DEFAULT_CORPUS_PATH,
                    annotations_path: Path = labels.DEFAULT_ANNOTATIONS_PATH,
                    out_path: Path = LABELED_PATH, artifacts: Path = ARTIFACTS) -> pd.DataFrame:
    """Label, group, split; write private labeled parquet and the aggregate report."""
    if not Path(corpus_path).exists():
        build_corpus_from_files(out_path=corpus_path)
    corpus = pd.read_parquet(corpus_path)
    ann = labels.load_solicitation_annotations(annotations_path)
    df = labels.build_labeled_dataset(corpus, ann)
    df["group_id"] = grouping.assign_groups(df["normalized_text"])
    ok = labels.trainable(df)
    df["split"] = pd.NA
    df.loc[df["message_id"].isin(ok["message_id"]), "split"] = df["message_id"].map(
        ok.assign(split=splits.train_val_test_split(ok, cfg.seed)).set_index("message_id")["split"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    rep = report.build_dataset_report(df)
    report.write_report(rep, artifacts)
    logger.info("Dataset: %d trainable (%d positive), %d need review.", rep["labeled"],
                rep["positive_count"], rep["review_required"])
    return df


def load_labeled(path: Path = LABELED_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def _split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["split"] == name].reset_index(drop=True)


def _cv(df_tv: pd.DataFrame, fit_predict, cfg: TrainConfig) -> dict | None:
    """Grouped stratified CV on train+validation only (never the test set)."""
    k = min(cfg.cv_folds, int(df_tv["label"].sum()))
    if k < 2:
        return None
    folds = splits.grouped_stratified_folds(df_tv, k, cfg.seed)
    oof = np.zeros(len(df_tv))
    for f in range(k):
        tr, te = (folds != f).to_numpy(), (folds == f).to_numpy()
        oof[te] = fit_predict(df_tv[tr].reset_index(drop=True), df_tv[te].reset_index(drop=True))
    m = metrics.compute_metrics(df_tv["label"], oof, 0.5)
    m["folds"] = k
    return m


def _evaluate(name, df, fit_predict_val, cfg, cv_fn) -> tuple[dict, dict]:
    """Shared evaluation: returns (metrics dict, raw bits incl. probabilities)."""
    tr, va, te = _split(df, "train"), _split(df, "validation"), _split(df, "test")
    p_va, p_te, extras = fit_predict_val(tr, va, te)
    sel = threshold.select_threshold(va["label"], p_va, cfg.target_precision)
    res = {
        "approach": name,
        "threshold": sel["threshold"],
        "threshold_reason": sel["reason"],
        "validation_threshold_table": sel["table"],
        "validation": metrics.compute_metrics(va["label"], p_va, sel["threshold"]),
        "test": metrics.compute_metrics(te["label"], p_te, sel["threshold"]),
        "cv_train_plus_validation": cv_fn(pd.concat([tr, va], ignore_index=True)),
        **{k: v for k, v in extras.items() if k.startswith("history")},
    }
    errs = errors.analyze_errors(va, p_va, sel["threshold"], "validation") + \
        errors.analyze_errors(te, p_te, sel["threshold"], "test")
    return res, {"errors": errs, **extras}


def run_all(cfg: TrainConfig, artifacts: Path = ARTIFACTS, model_dir: Path = MODEL_DIR,
            labeled_path: Path = LABELED_PATH) -> dict:
    """Run baseline + fine-tune, compare, save the selected model and artifacts."""
    set_seed(cfg.seed)
    device = detect_device()
    logger.info("Device: %s", device)
    df = load_labeled(labeled_path)
    df = labels.trainable(df)
    encoder = baseline.load_encoder(cfg.base_model, device)
    emb = dict(zip(df["message_id"], baseline.embed(df["normalized_text"].tolist(), encoder),
                   strict=True))

    def X(d):
        return np.stack([emb[m] for m in d["message_id"]])

    def base_fit(tr, va, te):
        clf = baseline.fit_baseline(X(tr), tr["label"].to_numpy(), cfg)
        return (clf.predict_proba(X(va))[:, 1], clf.predict_proba(X(te))[:, 1], {"clf": clf})

    def base_cv(tv):
        def fp(a, b):
            c = baseline.fit_baseline(X(a), a["label"].to_numpy(), cfg)
            return c.predict_proba(X(b))[:, 1]
        return _cv(tv, fp, cfg)

    def ft_fit(tr, va, te):
        m, tok, hist = finetune.fit_finetune(tr["normalized_text"].tolist(), tr["label"].tolist(),
                                             cfg, device, va["normalized_text"].tolist(),
                                             va["label"].tolist())
        pv = finetune.predict_proba(m, tok, va["normalized_text"].tolist(), device, cfg)
        pt = finetune.predict_proba(m, tok, te["normalized_text"].tolist(), device, cfg)
        return pv, pt, {"model": m, "tokenizer": tok, "history": hist}

    def ft_cv(tv):
        def fp(a, b):
            m, tok, _ = finetune.fit_finetune(a["normalized_text"].tolist(), a["label"].tolist(),
                                              cfg, device)
            return finetune.predict_proba(m, tok, b["normalized_text"].tolist(), device, cfg)
        return _cv(tv, fp, cfg)

    base_res, base_raw = _evaluate("embedding_logreg_baseline", df, base_fit, cfg, base_cv)
    ft_res, ft_raw = _evaluate("minilm_finetuned", df, ft_fit, cfg, ft_cv)

    def score(r):  # selection: pooled CV PR-AUC, then validation PR-AUC; baseline wins ties
        cv = (r["cv_train_plus_validation"] or {}).get("pr_auc")
        return (cv if cv is not None else -1.0, r["validation"]["pr_auc"] or -1.0)

    chosen = ft_res if score(ft_res) > score(base_res) else base_res
    chosen_raw = ft_raw if chosen is ft_res else base_raw
    comparison = {"selected": chosen["approach"],
                  "selection_rule": "highest pooled CV PR-AUC on train+validation, then "
                                    "validation PR-AUC; ties go to the simpler baseline",
                  "baseline": base_res, "finetuned": ft_res}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "evaluation.json").write_text(json.dumps(comparison, indent=2))
    (artifacts / "error_analysis.json").write_text(json.dumps(
        {"note": "PRIVATE: contains message text; gitignored.",
         "baseline": base_raw["errors"], "finetuned": ft_raw["errors"]}, indent=2))

    n_pos = int(df["label"].sum())
    meta = {
        "base_model": cfg.base_model, "model_version": "v1",
        "training_timestamp_note": "see saved_at",
        "training_dataset_size": int((df["split"] == "train").sum()),
        "total_labeled_size": len(df), "positive_count": n_pos,
        "negative_count": len(df) - n_pos, "random_seed": cfg.seed,
        "threshold": chosen["threshold"], "threshold_reason": chosen["threshold_reason"],
        "hyperparameters": cfg.to_dict(), "approach": chosen["approach"],
        "validation_metrics": chosen["validation"], "test_metrics": chosen["test"],
        "cv_metrics": chosen["cv_train_plus_validation"],
        "label_mapping": {"1": labels.POSITIVE_NAME, "0": labels.NEGATIVE_NAME},
        "reliability_warning": "Trained on very few positive examples; metrics are anecdotal.",
    }
    if chosen is ft_res:
        persistence.save_finetuned(chosen_raw["model"], chosen_raw["tokenizer"], model_dir, meta)
    else:
        persistence.save_baseline(encoder, chosen_raw["clf"], model_dir, meta)
    logger.info("Selected %s; saved to %s.", chosen["approach"], model_dir)
    return comparison
