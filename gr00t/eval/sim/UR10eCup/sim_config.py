# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and fidelity contract for the UR10e Cup simulator.

The public dataset identifies the gripper only as ``Robotiq 2F`` and does not
publish the camera or robot-to-table calibration.  Consequently, the bundled
scene is deliberately labelled a proxy.  A calibrated result must opt in with
an external MJCF and explicit verification flags; this prevents proxy success
rate from being reported as real-robot success rate by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np


DATASET_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
MUJOCO_JOINT_NAMES = tuple(f"{name}_joint" for name in DATASET_JOINT_NAMES)
DEFAULT_ARM_ACTUATOR_NAMES = DATASET_JOINT_NAMES
DEFAULT_GRIPPER_JOINT_NAMES = ("left_finger_joint", "right_finger_joint")
DEFAULT_GRIPPER_ACTUATOR_NAMES = ("left_finger", "right_finger")

# Observed limits from khanhnd61/ur10e-cup.  These are dataset-support limits,
# not the wider UR10e hardware limits.  Staying inside them is intentional for
# an evaluation proxy trained from this single-scene dataset.
DATASET_ARM_LOW = np.array([-1.6301, -1.9923, -1.6848, -1.6242, 1.5702, -3.2003], dtype=np.float64)
DATASET_ARM_HIGH = np.array([-1.0360, -1.5708, -1.2006, -1.1484, 1.5716, -2.6050], dtype=np.float64)
DATASET_ARM_RESET = (DATASET_ARM_LOW + DATASET_ARM_HIGH) / 2.0
WRIST_2_INDEX = DATASET_JOINT_NAMES.index("wrist_2")


@dataclass(frozen=True)
class UR10eCupSimConfig:
    """Runtime configuration for :class:`UR10eCupEnv`."""

    simulation_fidelity: str = "proxy"
    model_xml_path: str | None = None
    calibration_source: str | None = None
    calibration_sha256: str | None = None
    base_to_world_matrix: tuple[float, ...] | None = None
    tool0_to_tcp_matrix: tuple[float, ...] | None = None
    side_camera_to_world_matrix: tuple[float, ...] | None = None
    wrist_camera_to_tool_matrix: tuple[float, ...] | None = None
    gripper_model: str = "unspecified_robotiq_2f_proxy"
    arm_joint_names: tuple[str, ...] = MUJOCO_JOINT_NAMES
    arm_actuator_names: tuple[str, ...] = DEFAULT_ARM_ACTUATOR_NAMES
    gripper_joint_names: tuple[str, ...] = DEFAULT_GRIPPER_JOINT_NAMES
    gripper_actuator_names: tuple[str, ...] = DEFAULT_GRIPPER_ACTUATOR_NAMES
    cup_freejoint_name: str = "cup_freejoint"
    cup_body_name: str = "cup"
    tool_body_name: str = "tool"
    tool_site_name: str | None = None
    left_finger_geom_name: str = "left_finger_pad"
    right_finger_geom_name: str = "right_finger_pad"
    cup_geom_name: str = "cup_geom"
    support_geom_names: tuple[str, ...] = ("table", "floor")
    side_camera_name: str = "side"
    wrist_camera_name: str = "wrist"
    gripper_closed_ctrl: tuple[float, ...] = (0.0, 0.0)
    gripper_open_ctrl: tuple[float, ...] = (0.035, 0.035)
    gripper_closed_qpos: tuple[float, ...] = (0.0, 0.0)
    gripper_open_qpos: tuple[float, ...] = (0.035, 0.035)
    control_hz: int = 20
    physics_timestep: float = 0.002
    render_height: int = 480
    render_width: int = 640
    wrist_2_policy: str = "clamp_current"
    approach_distance_m: float = 0.12
    lift_height_m: float = 0.05
    stable_success_steps: int = 5
    reset_settle_control_steps: int = 20
    dataset_support_tolerance_rad: float = 0.02
    hardware_joint_limit_tolerance_rad: float = 0.01
    cup_reset_xy: tuple[float, float] = (-0.42, -0.30)
    cup_reset_xy_noise: float = 0.015
    cup_reset_z: float = 0.105
    cup_reset_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    task_description: str = "pick up the cup"
    tool_transform_verified: bool = False
    camera_calibration_verified: bool = False
    gripper_calibration_verified: bool = False
    scene_calibration_verified: bool = False

    @property
    def physics_substeps(self) -> int:
        value = 1.0 / (self.control_hz * self.physics_timestep)
        rounded = round(value)
        if not np.isclose(value, rounded, atol=1e-9):
            raise ValueError(
                "control_hz and physics_timestep must yield an integer number of substeps; "
                f"got {value}"
            )
        return int(rounded)

    @property
    def calibration_verified(self) -> bool:
        return all(
            (
                self.tool_transform_verified,
                self.camera_calibration_verified,
                self.gripper_calibration_verified,
                self.scene_calibration_verified,
            )
        )

    @property
    def model_sha256(self) -> str | None:
        if not self.model_xml_path:
            return None
        model_path = Path(self.model_xml_path).expanduser()
        if not model_path.is_file():
            return None
        digest = hashlib.sha256()
        with model_path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_rigid_transform(name: str, values: tuple[float, ...] | None) -> None:
        if values is None:
            raise ValueError(f"calibrated fidelity requires {name}")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (16,) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain 16 finite row-major values")
        matrix = matrix.reshape(4, 4)
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError(f"{name} rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"{name} rotation determinant must be +1")

    def validate(self) -> None:
        if self.simulation_fidelity not in {"proxy", "calibrated"}:
            raise ValueError("simulation_fidelity must be 'proxy' or 'calibrated'")
        if self.control_hz != 20:
            raise ValueError("UR10e Cup data was recorded at 20 Hz; control_hz must remain 20")
        if self.wrist_2_policy != "clamp_current":
            raise ValueError("wrist_2_policy must be 'clamp_current' for this dataset")
        if self.render_height <= 0 or self.render_width <= 0:
            raise ValueError("render dimensions must be positive")
        if self.stable_success_steps < 1:
            raise ValueError("stable_success_steps must be >= 1")
        if self.approach_distance_m <= 0:
            raise ValueError("approach_distance_m must be positive")
        if self.reset_settle_control_steps < 1:
            raise ValueError("reset_settle_control_steps must be >= 1")
        if self.dataset_support_tolerance_rad < 0:
            raise ValueError("dataset_support_tolerance_rad must be >= 0")
        if self.hardware_joint_limit_tolerance_rad < 0:
            raise ValueError("hardware_joint_limit_tolerance_rad must be >= 0")
        if self.physics_timestep <= 0:
            raise ValueError("physics_timestep must be positive")
        if len(self.arm_joint_names) != 6 or len(set(self.arm_joint_names)) != 6:
            raise ValueError("arm_joint_names must contain six unique names in dataset order")
        if len(self.arm_actuator_names) != 6 or len(set(self.arm_actuator_names)) != 6:
            raise ValueError("arm_actuator_names must contain six unique names in dataset order")
        if not self.gripper_joint_names or len(set(self.gripper_joint_names)) != len(
            self.gripper_joint_names
        ):
            raise ValueError("gripper_joint_names must contain unique measurement joints")
        if not self.gripper_actuator_names or len(set(self.gripper_actuator_names)) != len(
            self.gripper_actuator_names
        ):
            raise ValueError("gripper_actuator_names must contain unique actuator names")
        if not self.support_geom_names:
            raise ValueError("support_geom_names must not be empty")
        closed = np.asarray(self.gripper_closed_ctrl, dtype=np.float64)
        opened = np.asarray(self.gripper_open_ctrl, dtype=np.float64)
        expected_ctrl_shape = (len(self.gripper_actuator_names),)
        if closed.shape != expected_ctrl_shape or opened.shape != expected_ctrl_shape:
            raise ValueError("gripper control endpoints must match gripper_actuator_names")
        if not np.isfinite(closed).all() or not np.isfinite(opened).all():
            raise ValueError("gripper control endpoints must be finite")
        if np.any(np.isclose(closed, opened)):
            raise ValueError("each gripper open and closed control endpoint must differ")
        closed_qpos = np.asarray(self.gripper_closed_qpos, dtype=np.float64)
        opened_qpos = np.asarray(self.gripper_open_qpos, dtype=np.float64)
        expected_qpos_shape = (len(self.gripper_joint_names),)
        if closed_qpos.shape != expected_qpos_shape or opened_qpos.shape != expected_qpos_shape:
            raise ValueError("gripper qpos endpoints must match gripper_joint_names")
        if not np.isfinite(closed_qpos).all() or not np.isfinite(opened_qpos).all():
            raise ValueError("gripper qpos endpoints must be finite")
        if np.any(np.isclose(closed_qpos, opened_qpos)):
            raise ValueError("each gripper open and closed qpos endpoint must differ")
        cup_xy = np.asarray(self.cup_reset_xy, dtype=np.float64)
        cup_quaternion = np.asarray(self.cup_reset_quaternion_wxyz, dtype=np.float64)
        if cup_xy.shape != (2,) or not np.isfinite(cup_xy).all():
            raise ValueError("cup_reset_xy must contain two finite values")
        if cup_quaternion.shape != (4,) or not np.isfinite(cup_quaternion).all():
            raise ValueError("cup_reset_quaternion_wxyz must contain four finite values")
        if not np.isclose(np.linalg.norm(cup_quaternion), 1.0, atol=1e-5):
            raise ValueError("cup_reset_quaternion_wxyz must be unit length")
        _ = self.physics_substeps
        if self.simulation_fidelity == "calibrated":
            if not self.model_xml_path:
                raise ValueError("calibrated fidelity requires model_xml_path")
            if not self.calibration_source:
                raise ValueError("calibrated fidelity requires calibration_source provenance")
            if not self.calibration_sha256 or len(self.calibration_sha256) != 64:
                raise ValueError("calibrated fidelity requires a 64-character calibration_sha256")
            try:
                int(self.calibration_sha256, 16)
            except ValueError as exc:
                raise ValueError("calibration_sha256 must be hexadecimal") from exc
            if not self.tool_site_name:
                raise ValueError("calibrated fidelity requires a measured TCP tool_site_name")
            for name in (
                "base_to_world_matrix",
                "tool0_to_tcp_matrix",
                "side_camera_to_world_matrix",
                "wrist_camera_to_tool_matrix",
            ):
                self._validate_rigid_transform(name, getattr(self, name))
            if not self.calibration_verified:
                raise ValueError(
                    "calibrated fidelity requires verified tool, camera, gripper, "
                    "and scene calibration"
                )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["physics_substeps"] = self.physics_substeps
        result["calibration_verified"] = self.calibration_verified
        result["model_sha256"] = self.model_sha256
        return result


def load_sim_config(path: str | Path | None = None) -> UR10eCupSimConfig:
    """Load configuration from JSON or return safe proxy defaults."""

    path = path or os.environ.get("GR00T_UR10E_SIM_CONFIG")
    if path is None:
        config = UR10eCupSimConfig()
    else:
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_xml_path = payload.get("model_xml_path")
        if model_xml_path:
            model_path = Path(model_xml_path).expanduser()
            if not model_path.is_absolute():
                payload["model_xml_path"] = str((config_path.parent / model_path).resolve())
        tuple_fields = (
            "arm_joint_names",
            "arm_actuator_names",
            "base_to_world_matrix",
            "tool0_to_tcp_matrix",
            "side_camera_to_world_matrix",
            "wrist_camera_to_tool_matrix",
            "gripper_joint_names",
            "gripper_actuator_names",
            "support_geom_names",
            "gripper_closed_ctrl",
            "gripper_open_ctrl",
            "gripper_closed_qpos",
            "gripper_open_qpos",
            "cup_reset_xy",
            "cup_reset_quaternion_wxyz",
        )
        for field in tuple_fields:
            if field in payload and payload[field] is not None:
                payload[field] = tuple(payload[field])
        config = UR10eCupSimConfig(**payload)
    config.validate()
    return config
