#!/usr/bin/env python3
"""Build the audited UZ-05 static URDF and a non-destructive USD repair layer."""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "omni_drones/robots/assets/twowheel_uz05"
DEFAULT_BASE = ASSET / "usd/twowheel_uz05.all_joints.usd"
DEFAULT_RCRANK = ASSET / "usd/twowheel_uz05.rcrank.usd"
DEFAULT_USD = ASSET / "usd/twowheel_uz05.cad_repaired.usda"
DEFAULT_URDF = ASSET / "urdf/twowheel_uz05.cad_repaired.urdf"

WHEELS = {"Lwhl", "Rwhl"}
ACTIVE_LEG_INPUTS = {"LL", "Lcrank", "RL", "Rcrank"}
PASSIVE_LINKAGE_JOINTS = {"L12", "L234", "Lcal", "L45", "R12", "R234", "Rcal", "R45"}
CONTINUOUS_JOINTS = WHEELS | ACTIVE_LEG_INPUTS | PASSIVE_LINKAGE_JOINTS
PAIR_NAMES = (
    ("LL", "RL"),
    ("L234", "R234"),
    ("Lcal", "Rcal"),
    ("L45", "R45"),
    ("Lwhl", "Rwhl"),
    ("Lcrank", "Rcrank"),
    ("L12", "R12"),
)

CLOSURE_JOINTS = {
    "left_leg_closure": {
        "body0": "/twowheel_uz05/L12",
        "body1": "/twowheel_uz05/L234",
        "world_point": (-0.18775, -0.0000005, 0.0586415),
    },
    "right_leg_closure": {
        "body0": "/twowheel_uz05/R12",
        "body1": "/twowheel_uz05/R234",
        "world_point": (0.19275, -0.0000005, 0.0586415),
    },
    "left_lower_leg_closure": {
        "body0": "/twowheel_uz05/L45",
        "body1": "/twowheel_uz05/L234",
        # Independently fitted from the coaxial 5 mm and 4 mm holes in both
        # link meshes. Any X position on this transverse axis is equivalent.
        "world_point": (-0.1855, -0.13716242, 0.16906952),
    },
    "right_lower_leg_closure": {
        "body0": "/twowheel_uz05/R45",
        "body1": "/twowheel_uz05/R234",
        "world_point": (0.1855, -0.13716242, 0.16906952),
    },
}

CAD_FULL_MASS_KG = 18.68716973046
CAD_FULL_COM_M = Gf.Vec3d(-0.011479619716885435, -0.0028646201924304292, 0.11737279769417439)
CAD_EQUIVALENT_BASE = {
    "mass": 12.925939783250641,
    "center_of_mass": Gf.Vec3f(-0.017491670314694868, -0.0031104556469596697, 0.070852038883771326),
    "diagonal_inertia": Gf.Vec3f(0.59538277842676901, 0.62085669992993409, 1.1190841655043284),
    "principal_axes": Gf.Quatf(
        0.77950942333761164,
        Gf.Vec3f(-0.055098289679342917, -0.012367238501998714, -0.62383995448681395),
    ),
}


def _copy_attribute(source: Usd.Prim, target: Usd.Prim, name: str) -> None:
    attribute = source.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return
    target.CreateAttribute(name, attribute.GetTypeName(), custom=attribute.IsCustom()).Set(attribute.Get())


def _repair_usd(base: Path, rcrank_source: Path, output: Path) -> Usd.Stage:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = [os.path.relpath(base, output.parent).replace("\\", "/")]
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/twowheel_uz05"))

    source_stage = Usd.Stage.Open(str(rcrank_source))
    source_joint = source_stage.GetPrimAtPath("/twowheel_uz05/base_link/Rcrank")
    target_joint = stage.OverridePrim("/twowheel_uz05/base_link/Rcrank")
    target_joint.SetTypeName("PhysicsRevoluteJoint")
    for name in (
        "physics:axis",
        "physics:localPos0",
        "physics:localPos1",
        "physics:localRot0",
        "physics:localRot1",
    ):
        _copy_attribute(source_joint, target_joint, name)
    # PhysX expects explicit infinities for an unlimited revolute joint. A USD
    # value block composes as None but is interpreted as a zero-width limit by
    # the runtime in Isaac Sim 4.0.
    target_joint.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(float("-inf"))
    target_joint.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(float("inf"))

    repair_notes = {
        "uz05:cadSource": r"D:\WheelLeg\UZ-05-open总装.SLDASM",
        "uz05:cadConfiguration": "默认",
        "uz05:cadFullAssemblyMassKg": 18.68716973046,
        "uz05:cadChassisMassKg": 15.6823443830009,
        "uz05:cadLegMassKg": 0.994478771084523,
        "uz05:cadClosureStatus": "repaired_four_loop_axes_from_cad_meshes",
        "uz05:urdfTopology": "open_tree_for_visualization_and_static_regression",
    }
    root = stage.OverridePrim("/twowheel_uz05")
    for name, value in repair_notes.items():
        root.SetCustomDataByKey(name, value)

    base = stage.OverridePrim("/twowheel_uz05/base_link")
    base.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(CAD_EQUIVALENT_BASE["mass"])
    base.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Point3f).Set(
        CAD_EQUIVALENT_BASE["center_of_mass"]
    )
    base.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(
        CAD_EQUIVALENT_BASE["diagonal_inertia"]
    )
    base.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(
        CAD_EQUIVALENT_BASE["principal_axes"]
    )

    xforms = UsdGeom.XformCache()
    desired_world_rotation = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), 90.0)
    for name, closure in CLOSURE_JOINTS.items():
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/twowheel_uz05/loop_joints/{name}")
        body0_path = Sdf.Path(closure["body0"])
        body1_path = Sdf.Path(closure["body1"])
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])
        joint.CreateAxisAttr("Z")
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        joint.GetPrim().SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=["PhysxJointAPI"]),
        )
        joint.GetPrim().CreateAttribute(
            "physxJoint:enableProjection", Sdf.ValueTypeNames.Bool, custom=False
        ).Set(True)
        world_point = Gf.Vec3d(*closure["world_point"])
        for index, body_path in enumerate((body0_path, body1_path)):
            body = stage.GetPrimAtPath(body_path)
            body_world = xforms.GetLocalToWorldTransform(body)
            local_point = body_world.GetInverse().Transform(world_point)
            body_world_rotation = body_world.ExtractRotation()
            # USD joint frames compose as local * body under Gf's row-vector
            # convention. Solve local * body = desired world frame.
            local_rotation = desired_world_rotation * body_world_rotation.GetInverse()
            getattr(joint, f"CreateLocalPos{index}Attr")(Gf.Vec3f(local_point))
            getattr(joint, f"CreateLocalRot{index}Attr")(Gf.Quatf(local_rotation.GetQuat()))

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if prim.GetParent().GetPath() == Sdf.Path("/twowheel_uz05/loop_joints"):
            # Loop closures are passive constraints. A drive or arbitrary
            # angle limits here over-constrain the linkage and inject energy.
            continue
        name = prim.GetName()
        if name in CONTINUOUS_JOINTS:
            prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(float("-inf"))
            prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(float("inf"))
        else:
            # USD angular limits are authored in degrees. The previous +/-1.57
            # values constrained these joints to +/-1.57 degrees, not radians.
            prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(-90.0)
            prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(90.0)

        if name in PASSIVE_LINKAGE_JOINTS:
            # These coordinates are solved by the two closed loops on each
            # side. Any drive here would add an actuator that does not exist.
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            continue

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        if name in WHEELS:
            stiffness, damping, max_force = 0.0, 4.0, 2.46
        elif name in ACTIVE_LEG_INPUTS:
            stiffness, damping, max_force = 120.0, 12.0, 20.0
        else:
            stiffness, damping, max_force = 500.0, 20.0, 80.0
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(max_force)

    material = UsdShade.Material.Define(stage, "/twowheel_uz05/PhysicsMaterials/WheelRubber")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.2)
    physics_material.CreateDynamicFrictionAttr(1.0)
    physics_material.CreateRestitutionAttr(0.0)
    for wheel in sorted(WHEELS):
        collision = stage.GetPrimAtPath(f"/twowheel_uz05/{wheel}/collisions")
        binding = UsdShade.MaterialBindingAPI.Apply(collision)
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    layer.Save()
    return stage


def _format_vector(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _quat_to_rpy(quaternion: Gf.Quatd | Gf.Quatf) -> tuple[float, float, float]:
    matrix = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(quaternion)))
    # Gf composes row vectors. These indices apply the transpose required by
    # URDF's column-vector Rz(yaw)*Ry(pitch)*Rx(roll) convention.
    pitch_term = max(-1.0, min(1.0, -float(matrix[0][2])))
    pitch = math.asin(pitch_term)
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(matrix[1][2]), float(matrix[2][2]))
        yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
    else:
        roll = math.atan2(-float(matrix[2][1]), float(matrix[1][1]))
        yaw = 0.0
    return roll, pitch, yaw


def _origin(parent: ET.Element, matrix: Gf.Matrix4d) -> None:
    ET.SubElement(
        parent,
        "origin",
        xyz=_format_vector(matrix.ExtractTranslation()),
        rpy=_format_vector(_quat_to_rpy(matrix.ExtractRotationQuat())),
    )


def _body_prims(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    return {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }


def _joint_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]


def _write_urdf(stage: Usd.Stage, output: Path) -> None:
    bodies = _body_prims(stage)
    joints = _joint_prims(stage)
    xforms = UsdGeom.XformCache()
    robot = ET.Element("robot", name="twowheel_uz05_cad_repaired")
    robot.append(ET.Comment(
        " Static CAD repair: closed-loop constraints remain intentionally absent until "
        "SolidWorks chassis mates 259/260 and actuator limits are repaired and validated. "
    ))

    for name, prim in bodies.items():
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        com = prim.GetAttribute("physics:centerOfMass").Get() or Gf.Vec3f(0.0)
        principal_axes = prim.GetAttribute("physics:principalAxes").Get() or Gf.Quatf(1.0)
        diagonal = prim.GetAttribute("physics:diagonalInertia").Get() or Gf.Vec3f(1e-6)
        ET.SubElement(
            inertial,
            "origin",
            xyz=_format_vector(com),
            rpy=_format_vector(_quat_to_rpy(principal_axes)),
        )
        ET.SubElement(inertial, "mass", value=f"{float(prim.GetAttribute('physics:mass').Get()):.12g}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{float(diagonal[0]):.12g}",
            ixy="0",
            ixz="0",
            iyy=f"{float(diagonal[1]):.12g}",
            iyz="0",
            izz=f"{float(diagonal[2]):.12g}",
        )

        mesh_uri = f"package://twowheel_uz05/meshes/{name}.STL"
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=mesh_uri)
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=mesh_uri)

    for prim in joints:
        if prim.GetAttribute("physics:excludeFromArticulation").Get():
            continue
        body0 = prim.GetRelationship("physics:body0").GetTargets()[0]
        body1 = prim.GetRelationship("physics:body1").GetTargets()[0]
        parent_prim = stage.GetPrimAtPath(body0)
        child_prim = stage.GetPrimAtPath(body1)
        name = prim.GetName()
        if prim.IsA(UsdPhysics.FixedJoint):
            joint_type = "fixed"
        elif name in CONTINUOUS_JOINTS:
            joint_type = "continuous"
        else:
            joint_type = "revolute"
        joint = ET.SubElement(robot, "joint", name=name, type=joint_type)
        ET.SubElement(joint, "parent", link=parent_prim.GetName())
        ET.SubElement(joint, "child", link=child_prim.GetName())
        child_world = xforms.GetLocalToWorldTransform(child_prim)
        parent_world = xforms.GetLocalToWorldTransform(parent_prim)
        relative = child_world * parent_world.GetInverse()
        _origin(joint, relative)

        if joint_type != "fixed":
            axis_name = str(prim.GetAttribute("physics:axis").Get() or "Z")
            axis = {
                "X": Gf.Vec3d(1, 0, 0),
                "Y": Gf.Vec3d(0, 1, 0),
                "Z": Gf.Vec3d(0, 0, 1),
            }[axis_name]
            local_rot1 = prim.GetAttribute("physics:localRot1").Get() or Gf.Quatf(1.0)
            axis_in_child = Gf.Rotation(Gf.Quatd(local_rot1)).GetInverse().TransformDir(axis)
            ET.SubElement(joint, "axis", xyz=_format_vector(axis_in_child))
            if joint_type == "continuous":
                if name in WHEELS:
                    effort, velocity, damping = 2.46, 20.0, 4.0
                elif name in ACTIVE_LEG_INPUTS:
                    effort, velocity, damping = 20.0, 4.0, 12.0
                else:
                    # Passive closed-loop coordinates have no actuator. The
                    # URDF remains an open-tree static/interchange artifact.
                    effort, velocity, damping = 0.0, 20.0, 0.0
                ET.SubElement(
                    joint,
                    "limit",
                    effort=f"{effort:.12g}",
                    velocity=f"{velocity:.12g}",
                )
                ET.SubElement(joint, "dynamics", damping=f"{damping:.12g}", friction="0")
            else:
                lower_degrees = float(prim.GetAttribute("physics:lowerLimit").Get())
                upper_degrees = float(prim.GetAttribute("physics:upperLimit").Get())
                if name in PASSIVE_LINKAGE_JOINTS:
                    max_force, damping = 0.0, 0.0
                elif name in ACTIVE_LEG_INPUTS:
                    max_force, damping = 20.0, 12.0
                else:
                    max_force, damping = 80.0, 20.0
                ET.SubElement(
                    joint,
                    "limit",
                    lower=f"{math.radians(lower_degrees):.12g}",
                    upper=f"{math.radians(upper_degrees):.12g}",
                    effort=f"{max_force:.12g}",
                    velocity="4",
                )
                ET.SubElement(joint, "dynamics", damping=f"{damping:.12g}", friction="0")

    for wheel in sorted(WHEELS):
        gazebo = ET.SubElement(robot, "gazebo", reference=wheel)
        ET.SubElement(gazebo, "mu1").text = "1.2"
        ET.SubElement(gazebo, "mu2").text = "1.0"
        ET.SubElement(gazebo, "restitution_coefficient").text = "0.0"

    ET.indent(robot, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)


def _validate(stage: Usd.Stage, urdf: Path) -> None:
    joints = _joint_prims(stage)
    loop_joints = [
        prim
        for prim in joints
        if prim.GetAttribute("physics:excludeFromArticulation").Get()
    ]
    if len(loop_joints) != len(CLOSURE_JOINTS):
        raise RuntimeError(
            f"Expected {len(CLOSURE_JOINTS)} loop joints, found {len(loop_joints)}"
        )
    rcrank = stage.GetPrimAtPath("/twowheel_uz05/base_link/Rcrank")
    if not rcrank.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError("Rcrank was not repaired as a revolute joint")
    for prim in joints:
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if prim.GetAttribute("physics:excludeFromArticulation").Get():
            if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                raise RuntimeError(f"Loop closure must not have a drive: {prim.GetPath()}")
            continue
        name = prim.GetName()
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        if name in CONTINUOUS_JOINTS:
            if lower != float("-inf") or upper != float("inf"):
                raise RuntimeError(f"Continuous joint has a finite effective limit: {prim.GetPath()}")
        elif lower is None or upper is None:
            raise RuntimeError(f"Revolute joint is missing a finite limit: {prim.GetPath()}")
        elif not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
            raise RuntimeError(f"Revolute joint has a non-finite limit: {prim.GetPath()}")
        if name in PASSIVE_LINKAGE_JOINTS:
            if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                raise RuntimeError(f"Passive linkage joint has a drive: {prim.GetPath()}")
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        max_force = drive.GetMaxForceAttr().Get()
        if max_force is None or not math.isfinite(float(max_force)):
            raise RuntimeError(f"Non-finite drive max force: {prim.GetPath()}")
    xforms = UsdGeom.XformCache()
    for name in CLOSURE_JOINTS:
        joint = stage.GetPrimAtPath(f"/twowheel_uz05/loop_joints/{name}")
        if not joint.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"Missing closure joint: {name}")
        if not joint.GetAttribute("physics:excludeFromArticulation").Get():
            raise RuntimeError(f"Closure joint is not excluded from the articulation tree: {name}")
        points = []
        axes = []
        for index in (0, 1):
            body_path = joint.GetRelationship(f"physics:body{index}").GetTargets()[0]
            body = stage.GetPrimAtPath(body_path)
            body_world = xforms.GetLocalToWorldTransform(body)
            local_point = joint.GetAttribute(f"physics:localPos{index}").Get()
            local_rotation = Gf.Rotation(Gf.Quatd(joint.GetAttribute(f"physics:localRot{index}").Get()))
            points.append(body_world.Transform(local_point))
            axes.append((local_rotation * body_world.ExtractRotation()).TransformDir(Gf.Vec3d(0, 0, 1)))
        if (points[0] - points[1]).GetLength() >= 0.0005:
            raise RuntimeError(f"Closure position residual exceeds 0.5 mm: {name}")
        if abs(axes[0].GetNormalized() * axes[1].GetNormalized()) < 0.999999:
            raise RuntimeError(f"Closure axes are not parallel: {name}")

    total_mass = 0.0
    weighted_com = Gf.Vec3d(0.0)
    for prim in _body_prims(stage).values():
        mass = float(prim.GetAttribute("physics:mass").Get())
        local_com = prim.GetAttribute("physics:centerOfMass").Get()
        world_com = xforms.GetLocalToWorldTransform(prim).Transform(local_com)
        total_mass += mass
        weighted_com += Gf.Vec3d(world_com) * mass
    aggregate_com = weighted_com / total_mass
    if not math.isclose(total_mass, CAD_FULL_MASS_KG, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"USD mass {total_mass:.12g} does not match CAD {CAD_FULL_MASS_KG:.12g}")
    if (aggregate_com - CAD_FULL_COM_M).GetLength() >= 1e-6:
        raise RuntimeError(
            f"USD COM {_format_vector(aggregate_com)} does not match CAD {_format_vector(CAD_FULL_COM_M)}"
        )
    tree = ET.parse(urdf)
    links = tree.findall("link")
    urdf_joints = tree.findall("joint")
    if len(links) != 19 or len(urdf_joints) != 18:
        raise RuntimeError(f"Unexpected URDF topology: links={len(links)}, joints={len(urdf_joints)}")
    for mesh in tree.findall(".//mesh"):
        mesh_path = ASSET / mesh.get("filename").split("/meshes/", 1)[1]
        mesh_path = ASSET / "meshes" / mesh_path.name
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)

    def multiply(left, right):
        return [
            [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
            for row in range(4)
        ]

    def urdf_transform(origin: ET.Element):
        x, y, z = (float(value) for value in origin.get("xyz").split())
        roll, pitch, yaw = (float(value) for value in origin.get("rpy").split())
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ]

    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    poses = {"base_link": identity}
    pending = list(tree.findall("joint"))
    while pending:
        previous_count = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent in poses:
                poses[child] = multiply(poses[parent], urdf_transform(joint.find("origin")))
                pending.remove(joint)
        if len(pending) == previous_count:
            raise RuntimeError("URDF joint graph is disconnected")

    xforms = UsdGeom.XformCache()
    worst_position_error = 0.0
    for name, prim in _body_prims(stage).items():
        expected = xforms.GetLocalToWorldTransform(prim).ExtractTranslation()
        actual = [poses[name][index][3] for index in range(3)]
        error = math.sqrt(sum((actual[index] - float(expected[index])) ** 2 for index in range(3)))
        worst_position_error = max(worst_position_error, error)
    if worst_position_error >= 1e-6:
        raise RuntimeError(f"URDF zero-pose position error is {worst_position_error:.9g} m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-usd", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--rcrank-usd", type=Path, default=DEFAULT_RCRANK)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--output-urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    stage = _repair_usd(args.base_usd.resolve(), args.rcrank_usd.resolve(), args.output_usd.resolve())
    _write_urdf(stage, args.output_urdf.resolve())
    _validate(stage, args.output_urdf.resolve())
    print(f"USD: {args.output_usd.resolve()}")
    print(f"URDF: {args.output_urdf.resolve()}")
    print(
        f"Validated: 19 links, 18 tree joints, {len(CLOSURE_JOINTS)} loop joints, "
        "Rcrank revolute, finite drives, wheel material, zero-pose FK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
