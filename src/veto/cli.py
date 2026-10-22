"""Command-line interface: ``veto <command>`` (also installed as ``vito``)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from veto.training.config import TrainConfig


def _cfg(args) -> TrainConfig:
    cfg = TrainConfig()
    for k in ("seed", "epochs", "batch_size", "learning_rate"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="veto")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="corpus commands")
    csub = c.add_subparsers(dest="sub", required=True)
    csub.add_parser("build")
    csub.add_parser("inspect", help="build labeled dataset + report (no text printed)")

    t = sub.add_parser("train", help="run baseline + fine-tune, select and save the best")
    for name, typ in (("seed", int), ("epochs", int), ("batch_size", int),
                      ("learning_rate", float)):
        t.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)
    sub.add_parser("evaluate", help="print the saved evaluation summary")

    pr = sub.add_parser("predict", help="classify a message with the saved model")
    pr.add_argument("message")
    pr.add_argument("--model-dir", type=Path, default=Path("models/veto"))

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "corpus" and args.sub == "build":
        from veto.corpus.builder import build_corpus_from_files

        build_corpus_from_files()
    elif args.cmd == "corpus":
        from veto.training.pipeline import prepare_dataset

        prepare_dataset(TrainConfig())
        print(Path("artifacts/dataset_report.md").read_text())
    elif args.cmd == "train":
        from veto.training.pipeline import prepare_dataset, run_all

        cfg = _cfg(args)
        prepare_dataset(cfg)
        res = run_all(cfg)
        print(json.dumps({k: res[k] for k in ("selected", "selection_rule")}, indent=2))
    elif args.cmd == "evaluate":
        res = json.loads(Path("artifacts/evaluation.json").read_text())
        for k in ("baseline", "finetuned"):
            r = res[k]
            r = {a: b for a, b in r.items() if a != "validation_threshold_table"}
            print(k, json.dumps(r, indent=2))
        print("selected:", res["selected"])
    elif args.cmd == "predict":
        from veto.inference import VetoClassifier

        print(json.dumps(VetoClassifier.load(args.model_dir).predict(args.message), indent=2))


if __name__ == "__main__":
    main()
