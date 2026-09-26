"""Fine-tune a direct-target PPO policy with measured deployment latency."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], capture: bool = False) -> str:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=capture, check=True)
    if capture:
        print(result.stdout, end="", flush=True)
        return result.stdout
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=("stand", "low_speed"), default="stand")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--updates-per-phase", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-observation-delay", type=int, nargs=2, default=(1, 3))
    parser.add_argument(
        "--final-actuator-delay", type=int, nargs=2, default=(0, 1),
        help="final actuator delay range in 500 Hz motor ticks",
    )
    parser.add_argument("--vx-range", type=float, nargs=2, default=(0.05, 0.30))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parent = Path(args.checkpoint)
    if not (parent.exists() or Path(f"{parent}.zip").exists()):
        raise SystemExit(f"checkpoint not found: {parent}(.zip)")
    phases = (
        ((0, 0), (0, 0), 0, (0.0, 0.35)),
        ((0, max(args.final_observation_delay)),
         (0, max(args.final_actuator_delay)), 1, (0.35, 0.70)),
        (tuple(args.final_observation_delay),
         tuple(args.final_actuator_delay), 2, (0.70, 1.0)),
    )
    final_version = ""
    for index, (obs_delay, act_delay, stand_level, init_scale) in enumerate(
        phases, start=1
    ):
        version = f"direct_sim2real_{args.stage}_{args.tag}_{index}"
        final_version = version
        output = ROOT / "checkpoints" / version / "checkpoint"
        train = [
            sys.executable, "scripts/train_uz05.py", "--stage", args.stage,
            "--checkpoint", str(parent), "--reset-optimizer",
            "--updates", str(args.updates_per_phase), "--num-envs", str(args.num_envs),
            "--rollout-steps", str(args.rollout_steps), "--batch-size", str(args.batch_size),
            "--learning-rate", "1e-5", "--ent-coef", "0",
            "--stand-level", str(stand_level),
            "--init-scale-start", str(init_scale[0]),
            "--init-scale-end", str(init_scale[1]),
            "--deployment-mode", "--observation-delay-steps", str(obs_delay[0]), str(obs_delay[1]),
            "--actuator-delay-steps", str(act_delay[0]), str(act_delay[1]),
            "--version", version,
        ]
        if args.stage == "low_speed":
            train += ["--vx-range", str(args.vx_range[0]), str(args.vx_range[1]),
                      "--command-zero-prob", "0.25", "--command-reverse-prob", "0.50"]
        evaluate = [
            sys.executable, "scripts/eval_uz05.py", "--checkpoint", str(output),
            "--stage", args.stage, "--episodes", str(args.eval_episodes),
            "--stand-level", str(stand_level), "--init-scale", str(init_scale[1]),
            "--deployment-mode", "--observation-delay-steps", str(obs_delay[0]), str(obs_delay[1]),
            "--actuator-delay-steps", str(act_delay[0]), str(act_delay[1]),
        ]
        if args.dry_run:
            print("$ " + " ".join(train))
            print("$ " + " ".join(evaluate))
        else:
            run(train)
            run(evaluate, capture=True)
        parent = output
    print(f"final checkpoint: {parent}.zip")
    onnx_output = ROOT / "deployment" / "models" / f"{final_version}.onnx"
    export = [
        sys.executable, "scripts/export_onnx.py", "--checkpoint", str(parent),
        "--output", str(onnx_output),
    ]
    evaluate_onnx = [
        sys.executable, "scripts/eval_uz05.py", "--onnx", str(onnx_output),
        "--stage", args.stage, "--episodes", str(args.eval_episodes),
        "--stand-level", "2", "--init-scale", "1.0",
        "--deployment-mode",
        "--observation-delay-steps", str(args.final_observation_delay[0]),
        str(args.final_observation_delay[1]),
        "--actuator-delay-steps", str(args.final_actuator_delay[0]),
        str(args.final_actuator_delay[1]),
    ]
    if args.dry_run:
        print("$ " + " ".join(export))
        print("$ " + " ".join(evaluate_onnx))
    else:
        run(export)
        run(evaluate_onnx, capture=True)
    print(f"final ONNX: {onnx_output}")


if __name__ == "__main__":
    main()
