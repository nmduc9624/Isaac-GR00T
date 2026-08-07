# Fine-tuning GR00T N1.7 on UR10e + Robotiq Cup Picking

This recipe prepares [`khanhnd61/ur10e-cup`](https://huggingface.co/datasets/khanhnd61/ur10e-cup), a real-robot LeRobot v3 dataset containing 81 VR-teleoperated demonstrations of **pick up the cup**. The robot is a 6-DoF UR10e with a Robotiq 2F gripper, recorded at 20 FPS with side and wrist cameras.

The source action contract is important: each 7-D action is an **absolute** target containing six joint angles followed by the gripper command, and `action[t]` is approximately `observation.state[t+1]`. The preparation script rejects a snapshot that does not satisfy this contract.

## 1. Convert and validate the dataset

GR00T currently consumes its LeRobot v2-compatible format. Run the pinned conversion in the isolated conversion environment:

```bash
uv run --project scripts/lerobot_conversion \
  python examples/UR10eCup/prepare_dataset.py \
  --root examples/UR10eCup/ur10e_cup_lerobot
```

The converted dataset is written to:

```text
examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup
```

The converter preserves the original v3 snapshot beside it as `ur10e-cup_v30`. Keep that backup until `meta/gr00t_conversion.json` reports `"status": "ok"`. The report checks all 81 episode parquet files, both sets of episode videos, timestamps, finite values, binary gripper values, and the absolute one-step action alignment.

To re-run only validation:

```bash
uv run --project scripts/lerobot_conversion \
  python examples/UR10eCup/prepare_dataset.py \
  --root examples/UR10eCup/ur10e_cup_lerobot \
  --validate-only
```

## 2. Generate GR00T statistics

```bash
uv run python gr00t/data/stats.py \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UR10eCup/ur10e_cup_config.py
```

Do not reuse statistics from SO100, LIBERO, or another UR robot. State/action ordering and ranges are part of the model contract.

## 3. Fine-tune

The UR10e uses the checkpoint's reserved `NEW_EMBODIMENT` slot. `ur10e_cup_config.py` registers the two cameras, 6-D arm state, gripper state, and absolute 16-step action horizon.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --modality-config-path examples/UR10eCup/ur10e_cup_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/ur10e_cup_finetune
```

This source contains one task and one scene. A low training loss demonstrates fitting, not generalization. The preparation report records a deterministic recommended split of episodes 0-64 for training, 65-72 for validation, and 73-80 for testing, but the converter does **not** silently apply it. Build separate dataset subsets before claiming held-out accuracy.

## 4. Open-loop evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/ur10e_cup_finetune/checkpoint-10000 \
  --traj-ids 0 1 2 \
  --execution-horizon 8 \
  --steps 800 \
  --modality-keys arm_joints gripper \
  --save-plot-path /tmp/ur10e_cup_open_loop
```

Report arm and gripper errors separately. The `wrist_2` channel is effectively constant in this dataset and should not dominate aggregate MSE/MAE.

## 5. Real-robot deployment contract

There is no simulator for this cell, so success rate requires real UR10e rollouts. Before sending predictions to the robot controller:

1. Interpret model output as six absolute joint targets plus one gripper target.
2. Clamp `wrist_2` to the observed/current joint value; the dataset contains almost no motion for that channel.
3. Enforce UR controller joint position, velocity, acceleration, workspace, timeout, and watchdog limits independently of the model.
4. Start with shadow-mode logging, then low-speed trials with a physical emergency stop.
5. Record the number of trials, successes, completion time, safety stops, and videos. Offline MSE/MAE is not a substitute for real-robot success rate.

This repository recipe validates data and the model I/O contract; it does not replace the certified UR/Robotiq safety controller or provide a hardware transport implementation.
