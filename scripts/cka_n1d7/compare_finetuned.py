#!/usr/bin/env python
"""Create a fair fine-tuned baseline-versus-fine-tuned CKA report."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


@dataclass
class Args:
    baseline_dir: str
    cka_dir: str
    output_dir: str = "outputs/cka_n1d7/comparison"
    require_training_metadata: bool = True


def _load(directory: Path):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    actions = np.load(directory / "actions.npz")
    return summary, actions["prediction"], actions["ground_truth"]


def _percent_change(baseline: float, cka: float) -> float | None:
    return None if baseline == 0 else (cka - baseline) / baseline * 100.0


def main(args: Args) -> None:
    baseline, baseline_prediction, baseline_gt = _load(Path(args.baseline_dir))
    cka, cka_prediction, cka_gt = _load(Path(args.cka_dir))

    protocol_keys = (
        "dataset_path",
        "embodiment_tag",
        "trajectory_ids",
        "samples_per_trajectory",
        "sample_stride",
        "warmup_steps",
        "denoising_steps",
        "seed",
        "gpu",
    )
    mismatches = {
        key: (baseline.get(key), cka.get(key))
        for key in protocol_keys
        if baseline.get(key) != cka.get(key)
    }
    if mismatches:
        raise ValueError(f"Baseline/CKA evaluation protocol differs: {mismatches}")
    if baseline.get("cka_pruning_manifest") is not None:
        raise ValueError("Baseline checkpoint unexpectedly contains a CKA pruning manifest")
    if cka.get("cka_pruning_manifest") is None:
        raise ValueError("CKA checkpoint has no pruning manifest; refusing a misleading report")
    if args.require_training_metadata and (
        baseline.get("training_runtime_seconds") is None
        or cka.get("training_runtime_seconds") is None
    ):
        raise ValueError(
            "Missing trainer_state.json metadata. Final CLP claims must compare a fine-tuned "
            "baseline with a fine-tuned CKA checkpoint. Use --no-require-training-metadata "
            "only for debugging, never for the final report."
        )
    if baseline_gt.shape != cka_gt.shape or not np.allclose(baseline_gt, cka_gt):
        raise ValueError("Baseline and CKA did not evaluate identical ground-truth action chunks")

    metrics = (
        "parameter_count",
        "training_runtime_seconds",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "mse",
        "mae",
    )
    rows = []
    for metric in metrics:
        if baseline.get(metric) is None or cka.get(metric) is None:
            continue
        rows.append(
            {
                "metric": metric,
                "baseline": baseline[metric],
                "cka": cka[metric],
                "change_percent": _percent_change(baseline[metric], cka[metric]),
            }
        )
    report = {
        "protocol": {key: baseline.get(key) for key in protocol_keys},
        "comparison": rows,
        "inference_speedup": baseline["latency_mean_ms"] / cka["latency_mean_ms"],
        "prediction_drift_mse": float(np.mean(np.square(cka_prediction - baseline_prediction))),
        "prediction_drift_mae": float(np.mean(np.abs(cka_prediction - baseline_prediction))),
        "quality_note": (
            "Offline MSE/MAE is secondary. Report simulator/robot rollout success rate when "
            "claiming parity with CLP N1.5."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)

    chart_metrics = ["parameter_count", "latency_mean_ms", "peak_allocated_gib", "mse", "mae"]
    baseline_values = np.asarray([baseline[key] for key in chart_metrics], dtype=np.float64)
    cka_ratio = np.asarray([cka[key] for key in chart_metrics]) / baseline_values
    x = np.arange(len(chart_metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(x - width / 2, np.ones_like(x), width, label="fine-tuned baseline")
    axis.bar(x + width / 2, cka_ratio, width, label="fine-tuned CKA / baseline")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, chart_metrics, rotation=15)
    axis.set_ylabel("Normalized value (baseline = 1.0)")
    axis.set_title("GR00T N1.7: fair recovery-finetuned CKA comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "normalized_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
