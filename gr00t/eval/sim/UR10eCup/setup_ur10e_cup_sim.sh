#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep the core GR00T lockfile unchanged: MuJoCo is needed only by this proxy
# simulator, and uv pip installs it into the already-created project venv.
uv pip install "mujoco>=3.1,<4"

uv run --no-sync python -c \
  "import gymnasium, mujoco; print('UR10e Cup simulator dependencies are ready')"
