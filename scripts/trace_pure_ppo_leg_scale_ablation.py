#!/usr/bin/env python3
"""Record held-out actuator traces for the fresh 0.35 rad scale ablation."""

from __future__ import annotations

from uz05 import pure_ppo_stand as task


EXPERIMENT_ID = "pure_ppo_stand_leg_scale_035_v1"

task.LEG_POSITION_SCALE_RAD = 0.35
task.TASK_ID = EXPERIMENT_ID

from uz05 import pure_ppo_contract as contract  # noqa: E402

contract.CONTRACT_VERSION = EXPERIMENT_ID

import trace_pure_ppo_stand as trace  # noqa: E402

trace.TRACE_SEED_BASE = {0: 23_000, 1: 23_100, 2: 23_200}


if __name__ == "__main__":
    raise SystemExit(trace.main())
