# UR10e Real-Robot Shaking: Diagnosis and Improvement Plan

## Scope

This report summarizes the investigation of heavy UR10e shaking during real-time inference with the standalone GR00T N1.7 P50 checkpoint. The evidence examined includes:

- `prediction_vs_ground_truth.png`
- `aligned_prediction_vs_ground_truth_p50.png`
- The P50 training and evaluation notebook
- The pinned `cka-n1d7-recovery` branch
- GR00T's action processor and deployment guidance
- The UR10e dataset metadata and control frequency

## Main conclusion

The visible shaking is most likely caused by this chain:

```text
independent diffusion inference
    -> biased/noisy absolute joint-position chunk
    -> inference latency makes early actions stale
    -> abrupt replacement of the active chunk
    -> stiff UR position controller follows discontinuous targets
    -> physical joint/end-effector shaking
```

The current P50 checkpoint should not be considered safe for unrestricted real-robot operation. Robot-side smoothing alone may reduce symptoms, but the model and training workflow also need correction.

## What the plots actually show

### `prediction_vs_ground_truth.png`

The large saw-shaped pattern is mostly a visualization artifact. The benchmark concatenates many overlapping 16-step action windows into one long array. Each newly sampled window restarts at its own first future action, producing repeated ramps and resets that look like saw teeth.

This image therefore does **not** represent one continuous command trajectory sent to the robot.

### `aligned_prediction_vs_ground_truth_p50.png`

This is the more useful plot because overlapping action windows are aligned by episode and absolute timestamp. The large gaps/resets are episode boundaries rather than physical discontinuities inside one rollout.

However, it averages predictions that refer to the same timestamp. Unless the real-time controller performs the same temporal ensemble, this plot hides part of the disagreement between consecutive raw chunks.

The aligned plot still shows important model defects:

- `shoulder_pan` has a persistent offset.
- `wrist_1` does not reproduce the demonstrated trajectory shape.
- `wrist_3` has a persistent offset.
- The gripper produces random dips instead of remaining stable.
- Shoulder lift and elbow track substantially better than the other joints.

These are genuine model errors, not just plotting artifacts.

## Primary causes

### 1. Absolute joint-position prediction

The branch's UR10e modality configuration trains both the arm and gripper as absolute actions. Every diffusion call independently generates six large absolute joint targets. Small model or sampling differences therefore become direct target-position jumps when a new chunk replaces the previous one.

GR00T recommends state-relative arm prediction when adjacent chunks are inconsistent. Relative actions give the network a smaller, more uniform motion distribution to learn. The processor can convert relative predictions back to absolute physical joint targets before returning them to the robot.

### 2. P50 model quality is insufficient

The checkpoint retains only:

- 16/32 Action-DiT layers
- 8/16 language layers
- 4/4 VLSA layers

It shows poor tracking on several critical joints. Averaging eight diffusion seeds only reduced first-action MAE from approximately `0.011437` to `0.011219`, an improvement of about 1.9%. The reported mean sampling standard deviation was approximately `0.005319`.

This means most error is systematic model bias rather than removable sampling noise.

### 3. Pruning was performed before task adaptation

The standalone notebook computes CKA importance from the generic original GR00T checkpoint with a processor overlay, then prunes and recovers directly for UR10e.

For a task-specific robot policy, a better workflow is:

1. Fine-tune a full-depth UR10e baseline.
2. Compute CKA importance from that task-adapted baseline.
3. Prune the task-adapted model.
4. Recover the pruned model from the task-adapted baseline.

Layer importance measured before the model learns the UR10e task may select a poor subnetwork for precise control.

### 4. Inference latency creates stale actions

The UR10e dataset runs at 20 Hz, or one action every 50 ms. The notebook measured approximately 292 ms P95 model-only latency using 16 denoising steps.

```text
ceil(0.292 seconds * 20 actions/second) = 6 stale action steps
```

If an observation is captured at time `t`, inference completes around six dataset frames later. Executing the new chunk from `chunk[0]` commands a target intended for the past. This can produce lag, backward corrections, stop-and-go behavior, and oscillation.

The latency measurement also excludes camera capture, encoding, network transfer, preprocessing, robot communication, and controller delay. Real end-to-end staleness may be larger.

### 5. Abrupt chunk replacement

If the control process discards the current action buffer and immediately installs the first action of a newly sampled chunk, the boundary can contain a large position and velocity jump.

The aligned offline plot averages overlapping predictions, but a simple real-time client normally does not. Raw chunk-boundary inconsistency can therefore be worse than the plot suggests.

### 6. Low-frequency or aggressive UR control

Sending a new `servoj` target only when VLA inference finishes is not a stable servo loop. Universal Robots warns that noisy or low-frequency targets combined with high gain or short lookahead can cause vibration.

Policy targets should be generated at the demonstrated 20 Hz rate, then smoothly interpolated and delivered to the UR controller at its servo period.

### 7. Possible deployment contract mismatch

The real-time client must exactly reproduce the training contract:

- Six arm joints in this order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3
- Joint values in radians
- `side` and `wrist` RGB cameras in the expected order and orientation
- `uint8` RGB images in `[0, 255]`
- `float32` state arrays
- Correct gripper scale and semantics
- No accidental addition of already-absolute model outputs as deltas
- The same modality configuration and normalization statistics as training

Any mismatch can amplify instability.

## Required training and branch changes

### Change the arm to relative actions

In `examples/UR10eCup/ur10e_cup_config.py`, configure the arm as relative and leave the gripper absolute:

```python
ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
    state_key="arm_joints",
),
ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
),
```

After making this change:

- Create a new train-view/cache directory.
- Regenerate normal and relative statistics.
- Use a new experiment/variant name and output directory.
- Record the modality-config hash in the recovery identity.
- Train from scratch.
- Never resume the old absolute-action P50 checkpoint.

### Use baseline-first pruning

Recommended training sequence:

1. Train a full-depth relative-action UR10e baseline: 32/32 DiT, 16/16 language, 4/4 VLSA.
2. Confirm that the full model produces accurate, smooth raw chunks.
3. Capture CKA activations from the trained UR10e baseline.
4. Start with P25: 24/32 DiT, 12/16 language, 4/4 VLSA.
5. Recover P25 from the trained baseline.
6. Compare full baseline and P25 offline and on the robot.
7. Attempt more aggressive pruning only after P25 passes every control-quality gate.

P25 is the preferred initial efficiency target. Existing analytical results estimate about 1.28x compute speedup for P25, while P50 has demonstrated unacceptable action quality.

Training the current P50 model for 10,000 steps does not correct its absolute-action representation, pre-adaptation layer selection, or real-time chunk handling.

## Required notebook changes

The replacement notebook should:

- Build a separate relative-action train view.
- Force statistics regeneration when the action representation changes.
- Train and validate the full baseline first.
- Generate the pruning manifest from the task-adapted baseline.
- Use P25 as the first candidate.
- Sweep 4, 8, and 16 denoising steps instead of assuming 16 is optimal.
- Measure complete end-to-end latency when possible.
- Save complete raw action chunks with observation timestamps and horizon offsets.
- Keep episodes and action chunks visually separated.
- Produce per-joint metrics instead of relying on aggregate MSE/MAE.
- Package model weights, processor files, modality configuration, statistics, and their hashes together.

### Add control-quality diagnostics

For each model and denoising setting, calculate:

- Per-joint bias, MAE, RMSE, P95 error, and maximum error
- Per-joint diffusion variance across seeds
- Mean and P95 intra-chunk velocity
- Mean and P95 intra-chunk acceleration
- Position jump at each executed chunk boundary
- Velocity-direction similarity across boundaries
- Metrics after compensating for measured inference latency
- Full-model versus pruned-model degradation

Data-driven acceptance criteria are preferable to arbitrary thresholds. For example:

- Candidate boundary jumps should not exceed the ground-truth P95 per-frame motion.
- Candidate acceleration should remain close to the full baseline and demonstrations.
- Near-constant joints must not acquire persistent bias.
- A pruned model should remain within an agreed margin of the full baseline on every joint, not only aggregate MAE.

## Required real-time inference changes

### Use asynchronous inference

The robot-side control loop must continue executing a prepared buffer while inference runs in the background. It must not pause the robot while waiting for the GPU.

### Compensate for stale steps

Associate every observation and action chunk with timestamps. When a new chunk arrives:

```python
stale_steps = ceil(end_to_end_latency_seconds * 20)
first_usable_index = stale_steps
```

Discard actions whose intended execution timestamps have already passed. Use measured end-to-end latency, not only GPU latency.

### Align and ensemble overlapping predictions

Maintain predictions by intended execution timestamp. If several chunks predict the same future timestamp, blend them with greater weight on the newest observation. Do not average unrelated horizon offsets.

This temporal ensemble is easier to deploy with the existing 16-step horizon than GR00T RTC, which is experimental, not wired into the standard server/client path, and documented as requiring a horizon of at least 32.

### Smooth only after timestamp alignment

Smoothing should not be used to hide incorrectly aligned actions. After alignment:

- Blend incoming arm targets over approximately 2-4 policy samples.
- Enforce per-joint position-step, velocity, and acceleration limits derived from demonstrations and UR safety limits.
- Preserve a continuously advancing target trajectory.
- Process the gripper separately with thresholding, hysteresis, or debounce logic.
- Reject NaN, infinite, stale, out-of-range, or incorrectly shaped outputs.
- Stop safely if the action queue underflows rather than repeating discontinuous commands indefinitely.

### Denoising-step selection

Use the lowest denoising count that passes quality gates. Four steps is the standard deployment default and will reduce latency, but it must be compared against 8 and 16 because lower latency can trade against increased diffusion noise.

The selection should minimize end-to-end stale steps while preserving joint smoothness, not merely minimize model latency.

## UR10e controller guidance

The dataset and high-level policy operate at 20 Hz, but those targets should be interpolated in the robot-side process and issued at the UR e-Series servo period.

A conservative documented starting point is:

```text
servoj(q, t=0.002, lookahead_time=0.1, gain=300)
```

Do not increase gain or shorten lookahead while the target trajectory is noisy. Higher gain reacts more strongly to each target error, while a short lookahead reduces smoothing and can amplify vibration.

Controller parameters must be tuned cautiously on the actual robot with reduced speed, an unloaded workspace, appropriate safety limits, and an accessible physical emergency stop.

## Validation order

1. **Stop unrestricted testing with the current P50 checkpoint.**
2. **Ground-truth replay:** send recorded demonstration targets through the same interpolation and UR control stack at reduced speed. If this shakes, fix the controller before evaluating any model.
3. **Schema verification:** assert joint order, radians, image encoding, state dtype, gripper scale, and model/processor/statistics identity.
4. **Shadow mode:** run the full relative-action baseline without commanding the motors; log raw and post-processed chunks.
5. **Low-speed baseline test:** execute the full model with strict velocity/acceleration limits and an E-stop available.
6. **P25 offline comparison:** require per-joint and chunk-boundary quality close to the full baseline.
7. **P25 low-speed test:** deploy only after it passes the same gates.
8. **More aggressive pruning:** consider P37.5 or P50 only after independently passing all checks.

## Files that need modification

- Branch: `examples/UR10eCup/ur10e_cup_config.py`
- Branch: offline benchmark code to preserve raw chunk structure and calculate jitter metrics
- Notebook: baseline-first training, task-adapted CKA, P25 recovery, new cache/output identity, denoising sweep, and deployment gates
- Real-time UR10e client: asynchronous scheduling, timestamp alignment, stale-step removal, temporal ensemble, interpolation, limits, and gripper hysteresis

The notebook alone cannot resolve the shaking. Stable operation requires a new checkpoint **and** a corrected real-time controller.

## References

- [GR00T real-world deployment: jittering and stop-and-go](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md#6-common-issues-jittering-and-stop-and-go)
- [GR00T data/action configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md#action-modality)
- [GR00T policy input/output format](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md#understanding-the-action-format)
- [GR00T deployment arguments and denoising defaults](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md#standalone_inference_scriptpy)
- [UR10e-cup dataset metadata: 20 FPS](https://huggingface.co/datasets/khanhnd61/ur10e-cup/blob/12bbeaad3995e84cf43b99adecb0a8bc216cedbf/meta/info.json)
- [Universal Robots `servoj` documentation](https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-scriptmanual/G5/servoj_qavt0-008lookahead_time.htm)
