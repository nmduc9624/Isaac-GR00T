#!/usr/bin/env python
"""Compare internal CKA before and after pruning plus recovery fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gr00t.model.cka_pruning import linear_cka
from matplotlib import pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    baseline_calibration_dir: str
    optimized_calibration_dir: str
    output_dir: str = "outputs/cka_n1d7/recovered_heatmaps"


def _module_names(archive: np.lib.npyio.NpzFile) -> set[str]:
    return {key.split("__layer_", maxsplit=1)[0] for key in archive.files}


def _load_layers(archive: np.lib.npyio.NpzFile, module_name: str) -> list[np.ndarray]:
    prefix = f"{module_name}__layer_"
    keys = sorted(key for key in archive.files if key.startswith(prefix))
    return [archive[key] for key in keys]


def _matrix(layers: list[np.ndarray]) -> np.ndarray:
    depth = len(layers)
    result = np.eye(depth, dtype=np.float64)
    for row in range(depth):
        for column in range(row + 1, depth):
            score = linear_cka(layers[row], layers[column])
            result[row, column] = score
            result[column, row] = score
    return result


def _indices(metadata: dict, module_name: str, depth: int) -> list[int]:
    indices = metadata.get("module_layer_indices", {}).get(module_name)
    if indices is None:
        return list(range(depth))
    indices = [int(index) for index in indices]
    if len(indices) != depth:
        raise ValueError(
            f"{module_name}: metadata has {len(indices)} indices for activation depth {depth}"
        )
    return indices


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 1.0
    mask = ~np.eye(len(matrix), dtype=bool)
    return float(matrix[mask].mean())


def _adjacent_mean(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 1.0
    return float(np.diag(matrix, k=1).mean())


def _draw(
    module_name: str,
    baseline: np.ndarray,
    optimized: np.ndarray,
    baseline_indices: list[int],
    optimized_indices: list[int],
    output_path: Path,
) -> np.ndarray:
    baseline_lookup = {original: position for position, original in enumerate(baseline_indices)}
    missing = [index for index in optimized_indices if index not in baseline_lookup]
    if missing:
        raise ValueError(f"{module_name}: optimized original indices missing in baseline: {missing}")
    retained_positions = [baseline_lookup[index] for index in optimized_indices]
    baseline_retained = baseline[np.ix_(retained_positions, retained_positions)]
    delta = optimized - baseline_retained

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    baseline_image = axes[0].imshow(baseline, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_title(f"Baseline fine-tuned ({len(baseline_indices)} layers)")
    axes[0].set_xticks(range(len(baseline_indices)), baseline_indices, rotation=90)
    axes[0].set_yticks(range(len(baseline_indices)), baseline_indices)

    optimized_image = axes[1].imshow(optimized, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].set_title(f"CKA-pruned + recovery ({len(optimized_indices)} layers)")
    axes[1].set_xticks(range(len(optimized_indices)), optimized_indices, rotation=90)
    axes[1].set_yticks(range(len(optimized_indices)), optimized_indices)

    limit = max(float(np.max(np.abs(delta))), 1e-6)
    delta_image = axes[2].imshow(delta, vmin=-limit, vmax=limit, cmap="coolwarm")
    axes[2].set_title("Delta: recovered - baseline retained")
    axes[2].set_xticks(range(len(optimized_indices)), optimized_indices, rotation=90)
    axes[2].set_yticks(range(len(optimized_indices)), optimized_indices)

    for axis in axes:
        axis.set_xlabel("Original layer index")
        axis.set_ylabel("Original layer index")
    figure.colorbar(baseline_image, ax=axes[0], fraction=0.046, label="linear CKA")
    figure.colorbar(optimized_image, ax=axes[1], fraction=0.046, label="linear CKA")
    figure.colorbar(delta_image, ax=axes[2], fraction=0.046, label="CKA change")
    figure.suptitle(f"GR00T N1.7 representation comparison: {module_name}", fontsize=14)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return baseline_retained


def main(args: Args) -> None:
    baseline_dir = Path(args.baseline_calibration_dir)
    optimized_dir = Path(args.optimized_calibration_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metadata = json.loads(
        (baseline_dir / "metadata.json").read_text(encoding="utf-8")
    )
    optimized_metadata = json.loads(
        (optimized_dir / "metadata.json").read_text(encoding="utf-8")
    )
    baseline_archive = np.load(baseline_dir / "activations.npz")
    optimized_archive = np.load(optimized_dir / "activations.npz")

    if optimized_metadata.get("cka_pruning_manifest") is None:
        raise ValueError("Optimized checkpoint has no CKA pruning manifest")
    if baseline_metadata.get("sampled_steps") != optimized_metadata.get("sampled_steps"):
        raise ValueError("Heatmaps require identical calibration observations and seeds")

    common_modules = sorted(
        _module_names(baseline_archive) & _module_names(optimized_archive)
    )
    if not common_modules:
        raise ValueError("Baseline and optimized archives have no common CKA modules")

    report = {"args": asdict(args), "modules": {}}
    for module_name in common_modules:
        baseline_layers = _load_layers(baseline_archive, module_name)
        optimized_layers = _load_layers(optimized_archive, module_name)
        if baseline_layers[0].shape[0] != optimized_layers[0].shape[0]:
            raise ValueError(f"{module_name}: baseline/optimized sample counts differ")
        baseline_matrix = _matrix(baseline_layers)
        optimized_matrix = _matrix(optimized_layers)
        baseline_indices = _indices(
            baseline_metadata, module_name, len(baseline_layers)
        )
        optimized_indices = _indices(
            optimized_metadata, module_name, len(optimized_layers)
        )
        baseline_retained = _draw(
            module_name,
            baseline_matrix,
            optimized_matrix,
            baseline_indices,
            optimized_indices,
            output_dir / f"cka_before_after_{module_name}.png",
        )
        delta = optimized_matrix - baseline_retained
        report["modules"][module_name] = {
            "baseline_original_indices": baseline_indices,
            "optimized_original_indices": optimized_indices,
            "baseline_off_diagonal_mean": _off_diagonal_mean(baseline_matrix),
            "baseline_retained_off_diagonal_mean": _off_diagonal_mean(
                baseline_retained
            ),
            "optimized_off_diagonal_mean": _off_diagonal_mean(optimized_matrix),
            "baseline_retained_adjacent_mean": _adjacent_mean(baseline_retained),
            "optimized_adjacent_mean": _adjacent_mean(optimized_matrix),
            "delta_mean_absolute": float(np.mean(np.abs(delta))),
            "delta_frobenius": float(np.linalg.norm(delta)),
        }

    (output_dir / "cka_before_after_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
