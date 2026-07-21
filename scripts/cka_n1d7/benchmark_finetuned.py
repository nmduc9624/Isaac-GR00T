#!/usr/bin/env python
"""Offline benchmark for a fine-tuned baseline or fine-tuned CKA N1.7 checkpoint."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.model.cka_pruning import get_prunable_module_lists
from gr00t.policy.gr00t_policy import Gr00tPolicy
from matplotlib import pyplot as plt
import numpy as np
import torch
import tyro


@dataclass
class Args:
    model_path: str
    dataset_path: str
    embodiment_tag: str
    output_dir: str
    run_name: str = "finetuned"
    trajectory_ids: list[int] = field(default_factory=lambda: [0])
    samples_per_trajectory: int = 50
    sample_stride: int = 8
    warmup_steps: int = 5
    denoising_steps: int = 4
    seed: int = 42
    device: str = "cuda:0"


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prepare(traj, step, configs, tag, loader):
    data = extract_step_data(traj, step, configs, tag)
    observation = {f"state.{key}": value for key, value in data.states.items()}
    observation.update({f"video.{key}": np.asarray(value) for key, value in data.images.items()})
    for key in loader.modality_configs["language"].modality_keys:
        observation[key] = data.text
    return parse_observation_gr00t(observation, loader.modality_configs), data


def _flatten_prediction(prediction, action_keys):
    return np.concatenate([np.asarray(prediction[key][0]) for key in action_keys], axis=-1)


def _flatten_ground_truth(data, action_keys):
    return np.concatenate([np.asarray(data.actions[key]) for key in action_keys], axis=-1)


def _training_runtime(model_path: str) -> float | None:
    root = Path(model_path)
    if not root.is_dir():
        return None
    states = sorted(root.rglob("trainer_state.json"), key=lambda path: path.stat().st_mtime)
    if not states:
        return None
    state = json.loads(states[-1].read_text(encoding="utf-8"))
    for entry in reversed(state.get("log_history", [])):
        if "train_runtime" in entry:
            return float(entry["train_runtime"])
    return None


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The N1.7 benchmark requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed(args.seed)

    load_start = time.perf_counter()
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    _sync()
    model_load_seconds = time.perf_counter() - load_start

    loader = LeRobotEpisodeLoader(args.dataset_path, policy.get_modality_config())
    tag = EmbodimentTag.resolve(args.embodiment_tag)
    action_keys = loader.modality_configs["action"].modality_keys

    sample_specs = []
    for trajectory_id in args.trajectory_ids:
        traj = loader[trajectory_id]
        for step in list(range(0, len(traj), args.sample_stride))[: args.samples_per_trajectory]:
            sample_specs.append((trajectory_id, step, traj))
    if not sample_specs:
        raise ValueError("No benchmark samples were selected")

    _first_trajectory_id, first_step, first_traj = sample_specs[0]
    first_observation, _ = _prepare(
        first_traj, first_step, loader.modality_configs, tag, loader
    )
    with torch.inference_mode():
        for warmup_index in range(args.warmup_steps):
            _seed(args.seed + warmup_index)
            policy.get_action(first_observation)
    _sync()

    rows = []
    predictions = []
    ground_truth = []
    with torch.inference_mode():
        for trajectory_id, step, traj in sample_specs:
            observation, data = _prepare(traj, step, loader.modality_configs, tag, loader)
            sample_seed = args.seed + trajectory_id * 1_000_000 + step
            _seed(sample_seed)
            torch.cuda.reset_peak_memory_stats()
            _sync()
            started = time.perf_counter()
            prediction, _info = policy.get_action(observation)
            _sync()
            latency_ms = (time.perf_counter() - started) * 1000.0

            pred = _flatten_prediction(prediction, action_keys)
            target = _flatten_ground_truth(data, action_keys)
            horizon = min(len(pred), len(target))
            pred = pred[:horizon]
            target = target[:horizon]
            error = pred - target
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "step": step,
                    "seed": sample_seed,
                    "latency_ms": latency_ms,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "mse": float(np.mean(np.square(error))),
                    "mae": float(np.mean(np.abs(error))),
                }
            )
            predictions.append(pred)
            ground_truth.append(target)

    prediction_array = np.concatenate(predictions)
    ground_truth_array = np.concatenate(ground_truth)
    latencies = np.asarray([row["latency_ms"] for row in rows])
    module_depths = {
        name: len(layers) for name, layers in get_prunable_module_lists(policy.model).items()
    }
    summary = {
        **asdict(args),
        "gpu": torch.cuda.get_device_name(0),
        "model_load_seconds": model_load_seconds,
        "training_runtime_seconds": _training_runtime(args.model_path),
        "parameter_count": sum(parameter.numel() for parameter in policy.model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in policy.model.parameters() if parameter.requires_grad
        ),
        "module_depths": module_depths,
        "cka_pruning_manifest": getattr(policy.model.config, "cka_pruning_manifest", None),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in rows),
        "mse": float(np.mean(np.square(prediction_array - ground_truth_array))),
        "mae": float(np.mean(np.abs(prediction_array - ground_truth_array))),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "per_step.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "actions.npz", prediction=prediction_array, ground_truth=ground_truth_array
    )

    action_dim = prediction_array.shape[-1]
    figure, axes = plt.subplots(
        action_dim, 1, figsize=(12, max(3, 2.4 * action_dim)), squeeze=False
    )
    for dimension, axis in enumerate(axes[:, 0]):
        axis.plot(ground_truth_array[:, dimension], label="ground truth", linewidth=1.5)
        axis.plot(prediction_array[:, dimension], label=args.run_name, linewidth=1.1)
        axis.set_title(f"Action {dimension}")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_vs_ground_truth.png", dpi=150)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
