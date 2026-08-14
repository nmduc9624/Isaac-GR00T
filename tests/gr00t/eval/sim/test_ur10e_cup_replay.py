# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for UR10e ground-truth replay diagnostics."""

from gr00t.eval.sim.UR10eCup.replay_dataset_trajectory import (
    ReplayResult,
    _camera_frame_valid,
    _diagnostic_row,
    summarize_replays,
)
import numpy as np


def _result(status: str, episode_index: int) -> ReplayResult:
    return ReplayResult(
        status=status,
        episode_index=episode_index,
        episode_file=f"episode_{episode_index:06d}.parquet",
        steps_available=100,
        steps_executed=100,
        approached_cup=status == "passed",
        left_contact_seen=status == "passed",
        right_contact_seen=status == "passed",
        both_contacts_seen=status == "passed",
        lifted_seen=status == "passed",
        stable_success=status == "passed",
        side_camera_valid=True,
        wrist_camera_valid=True,
        hardware_safety_violation=False,
        minimum_tool_cup_distance_m=0.01,
        maximum_cup_height_m=0.2,
        final_info={},
    )


def test_multi_episode_gate_requires_every_episode_to_pass():
    summary = summarize_replays([_result("passed", 73), _result("failed", 74)])
    assert summary["status"] == "failed"
    assert summary["episodes_passed"] == 1
    assert summary["episodes_failed"] == 1


def test_camera_gate_rejects_blank_frames():
    assert _camera_frame_valid(np.zeros((32, 32, 3), dtype=np.uint8)) is False
    textured = np.indices((32, 32)).sum(axis=0) % 2 * 255
    assert _camera_frame_valid(np.repeat(textured[..., None], 3, axis=2)) is True


def test_diagnostic_row_uses_next_state_alignment():
    input_state = np.arange(7, dtype=np.float64)
    expected_next = input_state + 0.5
    action = expected_next.copy()
    info = {
        "left_finger_contact": False,
        "right_finger_contact": False,
        "table_support": True,
        "approached_cup": False,
        "lifted": False,
        "success": False,
        "dataset_support_violation": False,
        "hardware_safety_violation": False,
        "tool_cup_distance_m": 0.5,
        "cup_height_m": 0.1,
        "max_cup_height_m": 0.1,
        "gripper_open_fraction": 0.8,
        "gripper_target": 0.5,
        "arm_qpos_rad": expected_next[:6].tolist(),
        "arm_target_rad": action[:6].tolist(),
        "tool_position_m": [0.0, 0.0, 0.2],
        "cup_position_m": [0.4, 0.2, 0.1],
    }
    row = _diagnostic_row(
        4,
        input_state,
        expected_next,
        action,
        info,
        control_hz=20,
        side_camera_valid=True,
        wrist_camera_valid=True,
    )
    assert row["time_s"] == 0.2
    assert row["dataset_expected_next_shoulder_pan"] == 0.5
    assert row["dataset_action_shoulder_pan"] == 0.5
    assert row["sim_error_to_expected_shoulder_pan"] == 0.0
    assert row["dataset_gripper_expected_next"] == 6.5
    assert row["side_camera_valid"] is True
