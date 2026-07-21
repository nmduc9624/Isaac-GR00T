#!/usr/bin/env python
"""Capture pooled N1.7 hidden states for CKA calibration.

Authentication is read by huggingface_hub from ``HF_TOKEN``. The token is
never accepted as a CLI argument, so it cannot leak through process listings.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.model.cka_pruning import get_prunable_module_lists
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import tyro


@dataclass
class Args:
    model_path: str = "nvidia/GR00T-N1.7-3B"
    dataset_path: str = "demo_data/droid_sample"
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    output_dir: str = "outputs/cka_n1d7/calibration"
    trajectory_ids: list[int] = field(default_factory=lambda: [0])
    samples_per_trajectory: int = 16
    sample_stride: int = 4
    denoising_steps: int = 1
    seed: int = 42
    device: str = "cuda:0"
    modules: list[str] = field(
        default_factory=lambda: ["backbone_language", "action_dit", "vl_self_attention"]
    )
    require_hf_token: bool = False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"Forward hook output has no tensor: {type(value)!r}")


class ActivationRecorder:
    def __init__(self, model, requested_modules: list[str]):
        available = get_prunable_module_lists(model)
        unknown = sorted(set(requested_modules) - set(available))
        if unknown:
            raise ValueError(f"Unavailable CKA module(s) {unknown}; available={sorted(available)}")
        self.activations: dict[str, list[list[np.ndarray]]] = {}
        self.handles = []
        for module_name in requested_modules:
            layers = available[module_name]
            self.activations[module_name] = [[] for _ in layers]
            for layer_index, layer in enumerate(layers):
                self.handles.append(
                    layer.register_forward_hook(self._hook(module_name, layer_index))
                )

    def _hook(self, module_name: str, layer_index: int):
        def record(_module, _inputs, output):
            tensor = _first_tensor(output).detach().float()
            if tensor.ndim < 2:
                tensor = tensor.reshape(1, -1)
            elif tensor.ndim > 2:
                tensor = tensor.mean(dim=tuple(range(1, tensor.ndim - 1)))
            for row in tensor.cpu().numpy():
                self.activations[module_name][layer_index].append(row)

        return record

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def save(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for module_name, layer_values in self.activations.items():
            counts = {len(values) for values in layer_values}
            if len(counts) != 1 or not counts or next(iter(counts)) < 2:
                raise RuntimeError(
                    f"Inconsistent/insufficient hook calls for {module_name}: "
                    f"{[len(values) for values in layer_values]}"
                )
            for layer_index, values in enumerate(layer_values):
                arrays[f"{module_name}__layer_{layer_index:03d}"] = np.stack(values)
        np.savez_compressed(output_dir / "activations.npz", **arrays)
        metadata["activation_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


def _prepare_observation(traj, step, modality_configs, embodiment_tag, loader):
    data_point = extract_step_data(traj, step, modality_configs, embodiment_tag)
    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.asarray(value)
    for language_key in loader.modality_configs["language"].modality_keys:
        observation[language_key] = data_point.text
    return parse_observation_gr00t(observation, loader.modality_configs)


def main(args: Args) -> None:
    if args.require_hf_token:
        from huggingface_hub import get_token

        if get_token() is None:
            raise RuntimeError(
                "No Hugging Face token was found in HF_TOKEN or the huggingface_hub cache"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("N1.7 CKA capture requires a CUDA GPU")
    _seed_everything(args.seed)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    loader = LeRobotEpisodeLoader(args.dataset_path, policy.get_modality_config())
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    inference_modalities = deepcopy(loader.modality_configs)
    inference_modalities.pop("action")
    recorder = ActivationRecorder(policy.model, args.modules)

    sampled_steps = []
    try:
        with torch.inference_mode():
            for trajectory_id in args.trajectory_ids:
                traj = loader[trajectory_id]
                steps = list(range(0, len(traj), args.sample_stride))[
                    : args.samples_per_trajectory
                ]
                for step in steps:
                    sample_seed = args.seed + trajectory_id * 1_000_000 + step
                    _seed_everything(sample_seed)
                    observation = _prepare_observation(
                        traj, step, inference_modalities, embodiment_tag, loader
                    )
                    policy.get_action(observation)
                    sampled_steps.append(
                        {"trajectory_id": trajectory_id, "step": step, "seed": sample_seed}
                    )
    finally:
        recorder.close()

    recorder.save(
        Path(args.output_dir),
        {
            "args": asdict(args),
            "sampled_steps": sampled_steps,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    )
    print(f"Saved CKA calibration activations to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
