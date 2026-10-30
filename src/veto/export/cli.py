"""``veto export coreml``: export, verify against Python inference, report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from veto.export import parity
from veto.export.benchmark import benchmark
from veto.export.coreml import export_coreml

REPORT_PATH = Path("artifacts/coreml_export_report.json")


def _mb(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def _summary(label: str, s: dict) -> None:
    print(f"    {label:<14} n={s['n']:<5} max={s['max_abs_diff']:.5f} "
          f"mean={s['mean_abs_diff']:.5f} p95={s['p95_abs_diff']:.5f} "
          f"disagreements={s['classification_disagreements']}"
          + (f" (within precision noise: {s['disagreements_within_precision_noise']})"
             if s["classification_disagreements"] else ""))


def run_export_coreml(args) -> int:
    print("Loading selected Veto model...")
    result = export_coreml(args.model_dir, args.output, args.precision, args.deployment_target)
    src, meta, sizes = result["source"], result["metadata"], result["sizes"]
    print(f"  Source model:  {meta['source_approach']} ({src.model_type}), {meta['model_version']}")
    print(f"  Base model:    {meta['base_model']}")
    print(f"  Format:        Core ML {meta['coreml_format']}, {args.precision.upper()}, "
          f"{args.deployment_target}+, input [1, {meta['max_sequence_length']}]")
    print(f"  Threshold:     {meta['threshold']:.4f} (stored in metadata; not in the graph)")

    status, report = "NOT_VERIFIED", {}
    tol = parity.PROB_TOLERANCE_FP16 if args.precision == "fp16" else parity.PROB_TOLERANCE_FP32
    if not args.no_verify:
        print("\nValidating export against Python inference...")
        local = args.labeled if args.labeled.exists() else None
        if local is None:
            print(f"  (no local corpus at {args.labeled}; skipping the large local comparison)")
        kwargs = dict(model_dir=args.model_dir, package=result["package"],
                      tokenizer_dir=args.output / "tokenizer", seq_len=meta["max_sequence_length"],
                      threshold=meta["threshold"], tolerance=tol, local_labeled=local,
                      local_n=args.local_sample)
        report[args.precision] = parity.run_parity(**kwargs, compute_units="ALL")
        report[f"{args.precision}_cpu_only"] = parity.run_parity(**kwargs, compute_units="CPU_ONLY")
        if args.precision == "fp16":  # FP32 Core ML reference, to separate precision from bugs
            with tempfile.TemporaryDirectory() as tmp:
                ref = export_coreml(args.model_dir, Path(tmp) / "fp32", "fp32",
                                    args.deployment_target)
                kwargs.update(package=ref["package"], tolerance=parity.PROB_TOLERANCE_FP32)
                report["fp32_reference"] = parity.run_parity(**kwargs, compute_units="ALL")
        verdicts = {k: parity.verdict(v) for k, v in report.items()}
        order = ["FAIL", "PASS_WITH_NOTES", "PASS"]
        status = min(verdicts.values(), key=order.index)
        for name, rep in report.items():
            print(f"  [{name}] compute units: {rep['compute_units']}  verdict: {verdicts[name]}")
            _summary("fixtures", rep["fixtures"])
            for band, s in rep["threshold_bands"].items():
                if s.get("n"):
                    _summary(f"band:{band}", s)
            if "local_corpus" in rep:
                _summary("local corpus", rep["local_corpus"])
        print(f"\nPython/Core ML parity: {status}")
        if status == "PASS_WITH_NOTES":
            print("  Flips occur only for messages whose Python probability is within the "
                  "FP16 deviation of the threshold.")

    bench = None
    if args.benchmark:
        print("\nBenchmarking (development Mac; not an iPhone measurement)...")
        bench = benchmark(result["package"], args.output / "tokenizer", meta["max_sequence_length"])
        print(f"  Machine: {bench['machine']}")
        print(f"  Tokenization: median {bench['tokenization']['median_ms']} ms, "
              f"p95 {bench['tokenization']['p95_ms']} ms")
        for k, v in bench.items():
            if k.startswith("coreml_"):
                print(f"  {k}: inference median {v['inference']['median_ms']} ms "
                      f"(p95 {v['inference']['p95_ms']}); with tokenization "
                      f"{v['total_with_tokenization']['median_ms']} ms")

    # Record the validation in metadata.json so the artifact is self-describing.
    meta_path = args.output / "metadata.json"
    fx = report.get(args.precision, {}).get("fixtures", {})
    meta["validation"] = {
        "status": status, "tolerance": tol,
        "fixture_examples": fx.get("n"),
        "fixtures_max_abs_diff": fx.get("max_abs_diff"),
        "fixtures_classification_disagreements": fx.get("classification_disagreements"),
        "reference": "Veto Python inference (veto.inference.VetoClassifier)",
    }
    if bench:
        meta["benchmark"] = bench
    for _ in range(3):  # the recorded size includes the file itself; iterate until stable
        meta["artifact_sizes_bytes"] = sizes
        meta_path.write_text(json.dumps(meta, indent=2))
        new = meta_path.stat().st_size
        if new == sizes["metadata"]:
            break
        sizes["metadata"] = new
        sizes["total"] = sizes["mlpackage"] + sizes["tokenizer"] + new

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"parity": report, "benchmark": bench, "sizes": sizes,
                                       "metadata": meta}, indent=2))

    print("\nArtifact:")
    print(f"  {result['package']}")
    print("Size:")
    print(f"  veto.mlpackage   {_mb(sizes['mlpackage'])}")
    print(f"  tokenizer/       {_mb(sizes['tokenizer'])}")
    print(f"  metadata.json    {sizes['metadata']} bytes")
    print(f"  total            {_mb(sizes['total'])}")
    print(f"Metadata:\n  {meta_path}\nReport:\n  {REPORT_PATH} (aggregate numbers only)")
    return 1 if status == "FAIL" else 0
