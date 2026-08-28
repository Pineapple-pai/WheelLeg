#!/usr/bin/env python3
"""Audit the UZ-05 URDF and Isaac USD without modifying either asset."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = Path("/home/p/下载/UZ-05-open总装11.57/urdf/UZ-05-open总装11.57.urdf")
DEFAULT_USD = REPO_ROOT / "omni_drones/robots/assets/twowheel_uz05/usd/twowheel_uz05.all_joints.usd"


def _float(element: ET.Element | None, attribute: str, default: float = 0.0) -> float:
    return float(element.get(attribute, default)) if element is not None else default


def audit_urdf(path: Path) -> int:
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    moving = [joint for joint in joints if joint.get("type") != "fixed"]
    parents: dict[str, tuple[str, str]] = {}
    children: dict[str, list[tuple[str, str]]] = defaultdict(list)
    total_mass = 0.0
    masses = {}

    for name, link in links.items():
        mass = _float(link.find("./inertial/mass"), "value")
        masses[name] = mass
        total_mass += mass
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        parents[child] = (parent, joint.get("name"))
        children[parent].append((child, joint.get("name")))

    print(f"URDF: {path}")
    print(f"  links={len(links)}, joints={len(joints)}, moving_joints={len(moving)}, mass={total_mass:.6f} kg")
    print(f"  transmissions={len(root.findall('transmission'))}, gazebo_blocks={len(root.findall('gazebo'))}")

    missing = defaultdict(list)
    for joint in moving:
        name = joint.get("name")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        if limit is None or limit.get("effort") is None:
            missing["effort"].append(name)
        if limit is None or limit.get("velocity") is None:
            missing["velocity"].append(name)
        if joint.get("type") != "continuous" and (
            limit is None or limit.get("lower") is None or limit.get("upper") is None
        ):
            missing["position limits"].append(name)
        if dynamics is None or dynamics.get("damping") is None:
            missing["damping"].append(name)
        if dynamics is None or dynamics.get("friction") is None:
            missing["friction"].append(name)
    for field, names in missing.items():
        print(f"  FAIL missing {field}: {len(names)}/{len(moving)} ({', '.join(names)})")
    non_wheel_continuous = [
        joint.get("name")
        for joint in moving
        if joint.get("type") == "continuous" and joint.get("name") not in ("Lwhl", "Rwhl")
    ]
    if non_wheel_continuous:
        print(
            "  FAIL non-wheel joints exported as continuous: "
            + ", ".join(non_wheel_continuous)
        )

    def chain(link_name: str) -> list[str]:
        result = [link_name]
        while link_name in parents:
            link_name = parents[link_name][0]
            result.append(link_name)
        return list(reversed(result))

    for wheel in ("Lwhl", "Rwhl"):
        if wheel in links:
            print(f"  wheel support chain {wheel}: {' -> '.join(chain(wheel))}")

    asymmetries = []
    for left_name in sorted(name for name in links if name.startswith("L")):
        right_name = "R" + left_name[1:]
        if right_name not in links:
            continue
        if not math.isclose(masses[left_name], masses[right_name], rel_tol=0.02, abs_tol=1e-4):
            asymmetries.append(
                f"mass {left_name}={masses[left_name]:.6f}, {right_name}={masses[right_name]:.6f} kg"
            )
        left_joint = next((j for j in joints if j.get("name") == left_name), None)
        right_joint = next((j for j in joints if j.get("name") == right_name), None)
        if left_joint is not None and right_joint is not None and left_joint.get("type") != right_joint.get("type"):
            asymmetries.append(
                f"joint type {left_name}={left_joint.get('type')}, {right_name}={right_joint.get('type')}"
            )
    for issue in asymmetries:
        print(f"  WARN left/right asymmetry: {issue}")

    wheel_chain_links = set(chain("Lwhl") + chain("Rwhl"))
    leg_branch_links = sorted(
        name for name in links if name[0:1] in ("L", "R") and name not in wheel_chain_links
    )
    print(f"  FAIL leg links outside wheel support chains: {', '.join(leg_branch_links)}")
    print("  FAIL no closed-loop constraints or independent actuator mapping are represented")
    return len(missing) + len(asymmetries) + 2 + bool(non_wheel_continuous)


def audit_usd(path: Path, reset_height: float) -> tuple[int, bool]:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError:
        print("USD: SKIP pxr is unavailable; run this script in the Isaac Sim conda environment")
        return 1, False

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        print(f"USD: FAIL cannot open {path}")
        return 1, False
    xforms = UsdGeom.XformCache()
    collisions = []
    material_bindings = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        points = prim.GetAttribute("points").Get()
        if not points:
            continue
        matrix = xforms.GetLocalToWorldTransform(prim)
        world_points = [matrix.Transform(point) for point in points]
        low_z = min(float(point[2]) for point in world_points)
        high_z = max(float(point[2]) for point in world_points)
        approximation = prim.GetAttribute("physics:approximation").Get()
        collisions.append((low_z, high_z, str(prim.GetPath()), approximation, len(points)))
        cursor = prim
        while cursor:
            if any("material:binding" in rel.GetName() and rel.GetTargets() for rel in cursor.GetRelationships()):
                material_bindings += 1
                break
            cursor = cursor.GetParent()

    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    loop_joints = [
        prim for prim in joints
        if prim.GetAttribute("physics:excludeFromArticulation").Get()
    ]
    loop_errors = []
    for prim in loop_joints:
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            loop_errors.append(f"{prim.GetPath()} loop closure has an angular drive")
        if prim.GetAttribute("physics:lowerLimit").HasAuthoredValueOpinion() or prim.GetAttribute(
            "physics:upperLimit"
        ).HasAuthoredValueOpinion():
            loop_errors.append(f"{prim.GetPath()} loop closure has arbitrary angle limits")
        points = []
        axes = []
        for index in (0, 1):
            targets = prim.GetRelationship(f"physics:body{index}").GetTargets()
            if len(targets) != 1:
                loop_errors.append(f"{prim.GetPath()} body{index} target count={len(targets)}")
                continue
            body = stage.GetPrimAtPath(targets[0])
            body_world = xforms.GetLocalToWorldTransform(body)
            local_position = prim.GetAttribute(f"physics:localPos{index}").Get()
            local_rotation = prim.GetAttribute(f"physics:localRot{index}").Get()
            if local_position is None or local_rotation is None:
                loop_errors.append(f"{prim.GetPath()} missing local frame {index}")
                continue
            points.append(body_world.Transform(local_position))
            joint_rotation = Gf.Rotation(Gf.Quatd(local_rotation))
            axis_name = str(prim.GetAttribute("physics:axis").Get() or "Z")
            axis = {
                "X": Gf.Vec3d(1, 0, 0),
                "Y": Gf.Vec3d(0, 1, 0),
                "Z": Gf.Vec3d(0, 0, 1),
            }[axis_name]
            axes.append((joint_rotation * body_world.ExtractRotation()).TransformDir(axis))
        if len(points) == 2 and len(axes) == 2:
            residual = (points[0] - points[1]).GetLength()
            alignment = abs(axes[0].GetNormalized() * axes[1].GetNormalized())
            print(
                f"  loop {prim.GetName()}: anchor_residual={residual:.9g} m "
                f"axis_alignment={alignment:.9g}"
            )
            if residual >= 0.0005 or alignment < 0.999999:
                loop_errors.append(
                    f"{prim.GetPath()} residual={residual:.9g}, alignment={alignment:.9g}"
                )
    bad_drives = []
    for prim in joints:
        if not prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            continue
        stiffness = prim.GetAttribute("drive:angular:physics:stiffness").Get()
        damping = prim.GetAttribute("drive:angular:physics:damping").Get()
        max_force = prim.GetAttribute("drive:angular:physics:maxForce").Get()
        if stiffness is not None and (
            float(stiffness) >= 1e6
            or float(damping or 0.0) == 0.0
            or float(max_force or 0.0) >= 1e20
        ):
            bad_drives.append(str(prim.GetPath()).rsplit("/", 1)[-1])

    print(f"USD: {path}")
    print(
        f"  joints={len(joints)}, loop_joints={len(loop_joints)}, "
        f"collision_meshes={len(collisions)}, reset_height={reset_height:.4f} m"
    )
    for low_z, high_z, prim_path, approximation, point_count in sorted(collisions):
        body = prim_path.split("/")[-2]
        print(
            f"  collision {body:10} z=[{low_z:+.5f}, {high_z:+.5f}] "
            f"world_at_reset_low={low_z + reset_height:+.5f} m "
            f"approx={approximation} points={point_count}"
        )
    failures = 0
    closure_status = stage.GetDefaultPrim().GetCustomDataByKey("uz05:cadClosureStatus")
    if closure_status and str(closure_status).startswith("repaired_") and len(loop_joints) != 4:
        print(f"  FAIL repaired CAD expects 4 loop joints, found {len(loop_joints)}")
        failures += 1
    if loop_errors:
        print("  FAIL invalid loop joints: " + "; ".join(loop_errors))
        failures += 1
    if bad_drives:
        print(f"  FAIL nonphysical imported angular drives: {', '.join(bad_drives)}")
        failures += 1
    if material_bindings == 0:
        print("  FAIL no physics material is bound to any collision mesh")
        failures += 1
    if any(path_.endswith("/base_link/collisions") for _, _, path_, _, _ in collisions):
        print("  WARN base_link uses one convex hull; cavities are filled by the approximation")
    valid_closed_loops = len(loop_joints) == 4 and not loop_errors
    return failures, valid_closed_loops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--reset-height", type=float, default=0.084)
    args = parser.parse_args()
    failures = 0
    failures += audit_urdf(args.urdf.resolve())
    print()
    usd_failures, valid_closed_loops = audit_usd(args.usd.resolve(), args.reset_height)
    failures += usd_failures
    print()
    print("READINESS")
    print("  fixed-leg standing: CONDITIONAL (simulation regression only)")
    print("  wheel translation: BLOCKED until wheel contact/slip is validated")
    if valid_closed_loops:
        print("  actuated-leg standing: CONDITIONAL (static loop valid; run sweep_uz05_closed_loop.py)")
        print("  jumping: BLOCKED by unmeasured actuator limits and landing/contact validation")
    else:
        print("  actuated-leg standing and jumping: BLOCKED by missing closed-loop and actuator model")
    print("  hardware deployment: BLOCKED by missing calibration and actuator limits")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
