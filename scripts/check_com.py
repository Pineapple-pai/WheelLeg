#!/usr/bin/env python3
"""Print aggregate COM relative to wheel axle and base_link for the two-wheel USD."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = ROOT / "omni_drones/robots/assets/twowheel_uz05/usd/twowheel_uz05.all_joints.usd"


def _fmt(vec) -> str:
    return f"({float(vec[0]):+.6f}, {float(vec[1]):+.6f}, {float(vec[2]):+.6f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--root", default="/twowheel_uz05")
    args = parser.parse_args()

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise FileNotFoundError(args.usd)

    cache = UsdGeom.XformCache()
    total_mass = 0.0
    weighted_com = Gf.Vec3d(0.0)
    bodies = []

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_attr = prim.GetAttribute("physics:mass")
        if mass_attr is None or not mass_attr.HasAuthoredValueOpinion():
            continue
        mass = float(mass_attr.Get())
        local_com = prim.GetAttribute("physics:centerOfMass").Get() or Gf.Vec3f(0.0)
        world_com = cache.GetLocalToWorldTransform(prim).Transform(local_com)
        total_mass += mass
        weighted_com += Gf.Vec3d(world_com) * mass
        bodies.append((str(prim.GetPath()), mass, Gf.Vec3d(world_com)))

    if total_mass <= 0.0:
        raise RuntimeError("No authored rigid-body masses found in USD.")

    aggregate_com = weighted_com / total_mass

    def prim_pos(name: str):
        prim = stage.GetPrimAtPath(f"{args.root}/{name}")
        if not prim or not prim.IsValid():
            return None
        return Gf.Vec3d(cache.GetLocalToWorldTransform(prim).ExtractTranslation())

    left_wheel = prim_pos("Lwhl")
    right_wheel = prim_pos("Rwhl")
    base_link = prim_pos("base_link")

    print(f"usd: {args.usd}")
    print(f"total_mass_kg: {total_mass:.6f}")
    print(f"aggregate_com_world: {_fmt(aggregate_com)}")
    print(f"base_link_world: {_fmt(base_link) if base_link is not None else 'MISSING'}")
    print(f"left_wheel_world: {_fmt(left_wheel) if left_wheel is not None else 'MISSING'}")
    print(f"right_wheel_world: {_fmt(right_wheel) if right_wheel is not None else 'MISSING'}")

    if left_wheel is not None and right_wheel is not None:
        wheel_mid = (left_wheel + right_wheel) * 0.5
        axle = right_wheel - left_wheel
        offset = aggregate_com - wheel_mid
        print(f"wheel_mid_world: {_fmt(wheel_mid)}")
        print(f"wheel_track_m: {axle.GetLength():.6f}")
        print(f"com_minus_wheel_mid_m: {_fmt(offset)}")

    if base_link is not None:
        print(f"com_minus_base_link_m: {_fmt(aggregate_com - base_link)}")

    print("top_masses:")
    for path, mass, world_com in sorted(bodies, key=lambda item: item[1], reverse=True)[:16]:
        print(f"  {path.rsplit('/', 1)[-1]:14} mass={mass:9.5f} world_com={_fmt(world_com)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
