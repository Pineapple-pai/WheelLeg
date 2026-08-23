import argparse
import os

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--merge-fixed-joints", action="store_true")
    parser.add_argument("--fix-base", action="store_true")
    args = parser.parse_args()

    app = SimulationApp({"headless": True})

    import omni.kit.commands
    import omni.usd
    from omni.isaac.core.utils.extensions import enable_extension
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    enable_extension("omni.importer.urdf")
    app.update()

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config.")

    import_config.merge_fixed_joints = args.merge_fixed_joints
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = args.fix_base
    import_config.distance_scale = 1.0

    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=os.path.abspath(args.urdf),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError(f"URDF import failed: {args.urdf}")

    stage = omni.usd.get_context().get_stage()
    root_path = Sdf.Path(stage_path).GetPrefixes()[0]
    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim.IsValid():
        stage.SetDefaultPrim(root_prim)
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))

    os.makedirs(os.path.dirname(os.path.abspath(args.usd)), exist_ok=True)
    stage.GetRootLayer().Export(os.path.abspath(args.usd))
    print(f"Imported articulation root: {stage_path}")
    print(f"Saved USD: {os.path.abspath(args.usd)}")

    app.close()


if __name__ == "__main__":
    main()
