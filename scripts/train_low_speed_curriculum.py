"""Progressively train direct PPO wheel-speed tracking."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--tag", default="direct_v1")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--updates-per-block", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--vx-max", type=float, default=0.30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parent = Path(args.base_checkpoint)
    if not (parent.exists() or Path(f"{parent}.zip").exists()):
        raise SystemExit(f"checkpoint not found: {parent}(.zip)")
    for block in range(1, args.blocks + 1):
        output = ROOT / "checkpoints" / f"ppo_direct_low_speed_{args.tag}_b{block:02d}" / "checkpoint"
        command = [
            sys.executable, "scripts/train_uz05.py", "--stage", "low_speed",
            "--checkpoint", str(parent), "--reset-optimizer",
            "--vx-range", "0.05", f"{args.vx_max * block / args.blocks:.3f}",
            "--command-zero-prob", "0.25", "--command-reverse-prob", "0.50",
            "--updates", str(args.updates_per_block), "--num-envs", str(args.num_envs),
            "--rollout-steps", str(args.rollout_steps), "--batch-size", str(args.batch_size),
            "--version", output.parent.name,
        ]
        if args.dry_run:
            print("$ " + " ".join(command))
        else:
            run(command)
        parent = output
    print(f"final checkpoint: {parent}.zip")


if __name__ == "__main__":
    main()
