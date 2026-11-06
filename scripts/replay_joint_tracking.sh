#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
exec conda run --no-capture-output -n sim python scripts/replay_joint_tracking.py "$@"
