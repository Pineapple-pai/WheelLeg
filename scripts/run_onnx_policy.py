#!/usr/bin/env python3
"""Run or benchmark the deployment ONNX actor with CPUExecutionProvider."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from uz05.onnx_policy import OnnxPolicy
from uz05.spec import ACTOR_OBS_DIM, POLICY_RATE_HZ



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--benchmark-runs", type=int, default=1000)
    args = parser.parse_args()

    policy = OnnxPolicy(args.model)
    observation = (
        np.load(args.input_npy).astype(np.float32)
        if args.input_npy is not None
        else np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
    )
    action = policy.predict(observation)
    samples = []
    for _ in range(max(1, args.benchmark_runs)):
        start = time.perf_counter_ns()
        policy.predict(observation)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    values = np.asarray(samples)
    deadline_ms = 1000.0 / POLICY_RATE_HZ
    print("action: " + np.array2string(action, precision=6))
    print(f"latency_ms_mean: {values.mean():.4f}")
    print(f"latency_ms_p95: {np.quantile(values, 0.95):.4f}")
    print(f"latency_ms_p99: {np.quantile(values, 0.99):.4f}")
    print(f"latency_ms_max: {values.max():.4f}")
    print(f"policy_deadline_ms: {deadline_ms:.4f}")
    print(f"deadline_pass: {bool(values.max() < deadline_ms)}")


if __name__ == "__main__":
    main()
