#!/usr/bin/env python3
"""Export a passing 0.35 rad leg-scale checkpoint with full parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from uz05 import pure_ppo_stand as task


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pure_ppo_stand_leg_scale_035_v1"
VARIANT_TEST_SEEDS = {0: 23_000, 1: 23_100, 2: 23_200}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    task.LEG_POSITION_SCALE_RAD = 0.35
    task.TASK_ID = EXPERIMENT_ID
    from uz05 import pure_ppo_contract as contract

    contract.CONTRACT_VERSION = EXPERIMENT_ID
    import eval_pure_ppo_stand as evaluator

    evaluator.TEST_SEED_BASE = VARIANT_TEST_SEEDS
    import export_pure_ppo_stand_onnx as exporter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    argv = [
        str(Path(exporter.__file__).resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--training-seed", str(args.training_seed),
        "--evaluation-summary", str(args.evaluation_summary.resolve()),
        "--output", str(args.output.resolve()),
    ]
    sys.argv = argv
    result = exporter.main()
    if result != 0:
        return result

    output = args.output.resolve()
    parity_report = output.parent / f"{output.stem}_episode_parity" / "onnx_export_report.json"
    contract_path = output.with_suffix(output.suffix + ".contract.json")
    entry_hash = _sha256(Path(__file__).resolve())
    report = json.loads(parity_report.read_text(encoding="utf-8"))
    report["export_entrypoint_sha256"] = entry_hash
    parity_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["export_entrypoint_sha256"] = entry_hash
    contract_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "onnx_path": str(output),
        "onnx_sha256": report["onnx_sha256"],
        "contract_path": str(contract_path),
        "export_entrypoint_sha256": entry_hash,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
