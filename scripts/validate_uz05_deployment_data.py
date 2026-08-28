#!/usr/bin/env python3
"""Validate measured UZ-05 model data before generating a deployment USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "cfg/model/uz05_deployment.yaml"


def find_required(value, path=""):
    missing = []
    if isinstance(value, dict):
        for key, child in value.items():
            missing.extend(find_required(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            missing.extend(find_required(child, f"{path}[{index}]"))
    elif value is None or value == "REQUIRED":
        missing.append(path)
    return missing


def validate_ranges(data):
    errors = []
    for side in ("left", "right"):
        kinematics = data["kinematics"][side]
        for joint in ("hip", "knee"):
            if kinematics[f"{joint}_continuous"] is not True:
                errors.append(f"kinematics.{side}.{joint}_continuous must remain true")
            limits = kinematics[f"{joint}_safe_operating_range_rad"]
            if all(isinstance(value, (int, float)) for value in limits) and limits[0] >= limits[1]:
                errors.append(
                    f"kinematics.{side}.{joint}_safe_operating_range_rad must be [lower, upper]"
                )

    positive_paths = [
        "contact.wheel_radius_m",
        "contact.wheel_width_m",
        "control.policy_rate_hz",
        "control.motor_control_rate_hz",
        "mass_properties.total_mass_kg",
        "safety.max_jump_height_m",
        "safety.max_landing_force_n",
    ]
    for path in positive_paths:
        value = OmegaConf.select(data, path)
        if isinstance(value, (int, float)) and value <= 0:
            errors.append(f"{path} must be positive")

    for name, actuator in data["actuators"].items():
        prefix = f"actuators.{name}"
        if actuator["direction"] not in (-1, 1):
            errors.append(f"{prefix}.direction must be -1 or 1")
        for field in (
            "reduction",
            "velocity_limit_rad_s",
            "continuous_torque_nm",
            "peak_torque_nm",
            "peak_duration_s",
            "current_limit_a",
            "temperature_limit_c",
        ):
            if not isinstance(actuator[field], (int, float)) or actuator[field] <= 0:
                errors.append(f"{prefix}.{field} must be positive")
        if actuator["peak_torque_nm"] < actuator["continuous_torque_nm"]:
            errors.append(f"{prefix}.peak_torque_nm must be >= continuous_torque_nm")

    for field in (
        "longitudinal_static_friction",
        "longitudinal_dynamic_friction",
        "lateral_static_friction",
        "lateral_dynamic_friction",
        "restitution",
    ):
        value = data["contact"][field]
        if not isinstance(value, (int, float)) or value < 0:
            errors.append(f"contact.{field} must be non-negative")
    if data["contact"]["restitution"] > 1:
        errors.append("contact.restitution must be <= 1")
    if data["control"]["motor_control_rate_hz"] < data["control"]["policy_rate_hz"]:
        errors.append("control.motor_control_rate_hz must be >= policy_rate_hz")
    if data["safety"]["hardware_test_requires_support_rig"] is not True:
        errors.append("safety.hardware_test_requires_support_rig must remain true")
    if data["safety"]["wheel_regenerative_absorber_required"] is not True:
        errors.append("safety.wheel_regenerative_absorber_required must remain true")
    if data["safety"]["wheel_bus_clamp_max_v"] > 35.0:
        errors.append("safety.wheel_bus_clamp_max_v must be <= 35 V")

    for field in ("components_csv", "mates_csv", "geometry_step", "mass_report"):
        path = Path(data["source"][field]).expanduser()
        if not path.is_file():
            errors.append(f"source.{field} does not exist: {path}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.config.exists():
        print(f"BLOCKED: deployment data does not exist: {args.config}")
        print("Start from cfg/model/uz05_deployment.template.yaml and fill only measured values.")
        return 2

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    missing = find_required(config)
    errors = validate_ranges(config) if not missing else []
    if missing:
        print(f"BLOCKED: {len(missing)} required deployment values are missing")
        for path in missing:
            print(f"  {path}")
        return 2
    if errors:
        print("BLOCKED: deployment values failed validation")
        for error in errors:
            print(f"  {error}")
        return 2

    print(f"PASS: deployment data is structurally complete: {args.config.resolve()}")
    print("NOTE: PASS confirms completeness, not physical correctness; measurements still require review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
