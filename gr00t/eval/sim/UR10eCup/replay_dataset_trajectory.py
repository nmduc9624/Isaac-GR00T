# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a recorded UR10e trajectory as a simulator calibration gate.

This check validates the simulator before policy success rate is measured.  A
failure means the proxy scene/controller is not aligned well enough with the
dataset; it is not evidence about model or pruning quality.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .ur10e_cup_env import UR10eCupEnv


@dataclass
class ReplayResult:
    """Serializable ground-truth replay result."""

    status: str
    episode_index: int
    episode_file: str
    steps_available: int
    steps_executed: int
    approached_cup: bool
    left_contact_seen: bool
    right_contact_seen: bool
    both_contacts_seen: bool
    lifted_seen: bool
    stable_success: bool
    side_camera_valid: bool
    wrist_camera_valid: bool
    hardware_safety_violation: bool
    minimum_tool_cup_distance_m: float
    maximum_cup_height_m: float
    final_info: dict[str, Any]


def _episode_file(dataset_path: Path, episode_index: int) -> Path:
    matches = sorted(dataset_path.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet file for episode {episode_index}, found {len(matches)} "
            f"under {dataset_path}"
        )
    return matches[0]


def load_episode(dataset_path: Path, episode_index: int) -> tuple[Path, np.ndarray, np.ndarray]:
    """Load state/action arrays and enforce the seven-dimensional absolute contract."""

    episode_file = _episode_file(dataset_path, episode_index)
    table = pq.read_table(episode_file, columns=["observation.state", "action"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    shape_is_valid = (
        states.ndim == 2
        and actions.ndim == 2
        and states.shape[1:] == (7,)
        and actions.shape[1:] == (7,)
    )
    if not shape_is_valid:
        raise ValueError(
            "Episode must contain (T, 7) state/action arrays; "
            f"got {states.shape} and {actions.shape}"
        )
    if states.shape[0] != actions.shape[0] or states.shape[0] < 2:
        raise ValueError(
            f"State/action row counts are invalid: {states.shape[0]} and {actions.shape[0]}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Episode contains non-finite state/action values")
    return episode_file, states, actions


def _open_video_writer(path: Path | None, frame: np.ndarray, fps: int):
    if path is None:
        return None
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Video output requires opencv-python-headless") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(frame.shape[1]), int(frame.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create replay video: {path}")
    return writer


def _video_frame(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate((observation["video.side"], observation["video.wrist"]), axis=1)


def _camera_frame_valid(frame: np.ndarray) -> bool:
    return bool(
        frame.shape[-1:] == (3,)
        and np.isfinite(frame).all()
        and float(frame.mean()) > 1.0
        and float(frame.std()) > 1.0
    )


def _diagnostic_row(
    step_index: int,
    input_state: np.ndarray,
    expected_next_state: np.ndarray,
    action_row: np.ndarray,
    info: dict[str, Any],
    control_hz: int,
    side_camera_valid: bool,
    wrist_camera_valid: bool,
) -> dict[str, Any]:
    """Flatten one timestamp-aligned replay step for CSV inspection."""

    row: dict[str, Any] = {
        "step_index": step_index,
        "time_s": step_index / control_hz,
        "left_finger_contact": info["left_finger_contact"],
        "right_finger_contact": info["right_finger_contact"],
        "table_support": info["table_support"],
        "approached_cup": info["approached_cup"],
        "lifted": info["lifted"],
        "success": info["success"],
        "dataset_support_violation": info["dataset_support_violation"],
        "hardware_safety_violation": info["hardware_safety_violation"],
        "side_camera_valid": side_camera_valid,
        "wrist_camera_valid": wrist_camera_valid,
        "tool_cup_distance_m": info["tool_cup_distance_m"],
        "cup_height_m": info["cup_height_m"],
        "max_cup_height_m": info["max_cup_height_m"],
        "dataset_gripper_state": float(input_state[6]),
        "dataset_gripper_expected_next": float(expected_next_state[6]),
        "dataset_gripper_action": float(action_row[6]),
        "sim_gripper_open_fraction": info["gripper_open_fraction"],
        "sim_gripper_target": info["gripper_target"],
    }
    simulator_qpos = np.asarray(info["arm_qpos_rad"], dtype=np.float64)
    simulator_target = np.asarray(info["arm_target_rad"], dtype=np.float64)
    for index, name in enumerate(
        ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    ):
        row[f"dataset_state_{name}"] = float(input_state[index])
        row[f"dataset_expected_next_{name}"] = float(expected_next_state[index])
        row[f"dataset_action_{name}"] = float(action_row[index])
        row[f"sim_qpos_{name}"] = float(simulator_qpos[index])
        row[f"sim_target_{name}"] = float(simulator_target[index])
        row[f"sim_error_to_expected_{name}"] = float(
            simulator_qpos[index] - expected_next_state[index]
        )
    for prefix, values in (
        ("tool", info["tool_position_m"]),
        ("cup", info["cup_position_m"]),
    ):
        for axis, value in zip(("x", "y", "z"), values, strict=True):
            row[f"{prefix}_{axis}_m"] = float(value)
    return row


def summarize_replays(results: list[ReplayResult]) -> dict[str, Any]:
    """Build the multi-episode calibration acceptance summary."""

    if not results:
        raise ValueError("At least one replay result is required")
    passed = sum(result.status == "passed" for result in results)
    return {
        "status": "passed" if passed == len(results) else "failed",
        "episodes_requested": len(results),
        "episodes_passed": passed,
        "episodes_failed": len(results) - passed,
        "acceptance_rule": (
            "every episode must approach the cup, establish simultaneous two-finger contact, "
            "lift it, reach stable success, provide non-blank side/wrist cameras, and avoid "
            "hardware safety violations"
        ),
        "results": [asdict(result) for result in results],
    }


def replay_episode(
    dataset_path: Path,
    episode_index: int,
    *,
    config_path: str | None = None,
    max_steps: int | None = None,
    video_path: Path | None = None,
    diagnostics_path: Path | None = None,
    scene_reset_options: dict[str, Any] | None = None,
) -> ReplayResult:
    """Replay one demonstration and return the simulator acceptance contract."""

    episode_file, states, actions = load_episode(dataset_path, episode_index)
    limit = len(actions) if max_steps is None else min(len(actions), max_steps)
    if limit < 1:
        raise ValueError("max_steps must allow at least one action")

    env = UR10eCupEnv(config_path=config_path)
    writer = None
    diagnostics_file = None
    diagnostics_writer = None
    try:
        reset_options = {
            "arm_qpos": states[0, :6],
            "gripper_open_fraction": states[0, 6],
        }
        allowed_scene_keys = {"cup_xy", "cup_z", "cup_quaternion_wxyz"}
        unknown_scene_keys = set(scene_reset_options or {}) - allowed_scene_keys
        if unknown_scene_keys:
            raise ValueError(f"Unsupported scene reset options: {sorted(unknown_scene_keys)}")
        reset_options.update(scene_reset_options or {})
        observation, info = env.reset(seed=episode_index, options=reset_options)
        frame = _video_frame(observation)
        writer = _open_video_writer(video_path, frame, env.metadata["render_fps"])
        if writer is not None:
            import cv2

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        approached = bool(info["approached_cup"])
        left_contact = bool(info["left_finger_contact"])
        right_contact = bool(info["right_finger_contact"])
        both_contacts = left_contact and right_contact
        lifted = bool(info["lifted"])
        success = bool(info["success"])
        side_camera_valid = _camera_frame_valid(observation["video.side"])
        wrist_camera_valid = _camera_frame_valid(observation["video.wrist"])
        hardware_safety = bool(info["hardware_safety_violation"])
        minimum_distance = float(info["tool_cup_distance_m"])
        maximum_height = float(info["max_cup_height_m"])
        steps_executed = 0

        for step_index, action_row in enumerate(actions[:limit]):
            action = {
                "action.arm_joints": action_row[:6].astype(np.float32),
                "action.gripper": np.array([action_row[6]], dtype=np.float32),
            }
            observation, _, terminated, _, info = env.step(action)
            steps_executed += 1
            approached |= bool(info["approached_cup"])
            left_contact |= bool(info["left_finger_contact"])
            right_contact |= bool(info["right_finger_contact"])
            both_contacts |= bool(info["left_finger_contact"] and info["right_finger_contact"])
            lifted |= bool(info["lifted"])
            success |= bool(info["success"])
            current_side_camera_valid = _camera_frame_valid(observation["video.side"])
            current_wrist_camera_valid = _camera_frame_valid(observation["video.wrist"])
            side_camera_valid &= current_side_camera_valid
            wrist_camera_valid &= current_wrist_camera_valid
            hardware_safety |= bool(info["hardware_safety_violation"])
            minimum_distance = min(minimum_distance, float(info["tool_cup_distance_m"]))
            maximum_height = max(maximum_height, float(info["max_cup_height_m"]))
            if diagnostics_path is not None:
                diagnostic = _diagnostic_row(
                    step_index,
                    states[step_index],
                    (
                        states[step_index + 1]
                        if step_index + 1 < len(states)
                        else np.full(7, np.nan, dtype=np.float64)
                    ),
                    action_row,
                    info,
                    env.metadata["render_fps"],
                    current_side_camera_valid,
                    current_wrist_camera_valid,
                )
                if diagnostics_writer is None:
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostics_file = diagnostics_path.open("w", encoding="utf-8", newline="")
                    diagnostics_writer = csv.DictWriter(
                        diagnostics_file, fieldnames=list(diagnostic)
                    )
                    diagnostics_writer.writeheader()
                diagnostics_writer.writerow(diagnostic)
            if writer is not None:
                import cv2

                frame = _video_frame(observation)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated:
                break

        passed = (
            approached
            and both_contacts
            and lifted
            and success
            and side_camera_valid
            and wrist_camera_valid
            and not hardware_safety
        )
        return ReplayResult(
            status="passed" if passed else "failed",
            episode_index=episode_index,
            episode_file=str(episode_file),
            steps_available=len(actions),
            steps_executed=steps_executed,
            approached_cup=approached,
            left_contact_seen=left_contact,
            right_contact_seen=right_contact,
            both_contacts_seen=both_contacts,
            lifted_seen=lifted,
            stable_success=success,
            side_camera_valid=side_camera_valid,
            wrist_camera_valid=wrist_camera_valid,
            hardware_safety_violation=hardware_safety,
            minimum_tool_cup_distance_m=minimum_distance,
            maximum_cup_height_m=maximum_height,
            final_info=info,
        )
    finally:
        if writer is not None:
            writer.release()
        if diagnostics_file is not None:
            diagnostics_file.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        help="One or more episodes. Defaults to episode 73 for backwards compatibility.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path")
    parser.add_argument(
        "--episode-reset-options",
        type=Path,
        help=(
            "Optional JSON mapping from episode number to cup_xy, cup_z, and/or "
            "cup_quaternion_wxyz. Use this only with measured scene poses."
        ),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write diagnostics but return exit code zero when the calibration gate fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_index is not None and args.episode_indices is not None:
        raise SystemExit("Use only one of --episode-index or --episode-indices")
    default_episode = 73 if args.episode_index is None else args.episode_index
    episode_indices = args.episode_indices or [default_episode]
    episode_reset_options: dict[str, Any] = {}
    if args.episode_reset_options is not None:
        episode_reset_options = json.loads(args.episode_reset_options.read_text(encoding="utf-8"))
        if not isinstance(episode_reset_options, dict):
            raise SystemExit("--episode-reset-options must contain a JSON object")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for episode_index in episode_indices:
        episode_dir = (
            args.output_dir
            if len(episode_indices) == 1
            else args.output_dir / f"episode_{episode_index:06d}"
        )
        episode_dir.mkdir(parents=True, exist_ok=True)
        video_path = episode_dir / "ground_truth_replay.mp4" if args.save_video else None
        result = replay_episode(
            args.dataset_path,
            episode_index,
            config_path=args.config_path,
            max_steps=args.max_steps,
            video_path=video_path,
            diagnostics_path=episode_dir / "replay_diagnostics.csv",
            scene_reset_options=episode_reset_options.get(str(episode_index)),
        )
        (episode_dir / "ground_truth_replay_report.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8"
        )
        results.append(result)
        print(json.dumps(asdict(result), indent=2))
    if len(results) == 1:
        # Preserve the original single-episode report path for existing notebooks.
        (args.output_dir / "ground_truth_replay_report.json").write_text(
            json.dumps(asdict(results[0]), indent=2), encoding="utf-8"
        )
    summary = summarize_replays(results)
    summary.update(
        {
            "dataset_path": str(args.dataset_path.resolve()),
            "config_path": str(Path(args.config_path).resolve()) if args.config_path else None,
            "episode_reset_options_path": (
                str(args.episode_reset_options.resolve())
                if args.episode_reset_options is not None
                else None
            ),
            "action_contract": "absolute joint target; action[t] is compared with state[t+1]",
            "gripper_contract": "1=open, 0=closed; no additional temporal shift",
        }
    )
    report_path = args.output_dir / "ground_truth_replay_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    if summary["status"] != "passed" and not args.allow_failure:
        raise SystemExit(
            "Ground-truth replay calibration gate failed. Do not report policy success rate until "
            "the simulator passes approach, two-finger contact, lift, and stable-success checks."
        )


if __name__ == "__main__":
    main()
