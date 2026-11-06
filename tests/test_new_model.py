"""New asset integration: terrain, actuator caps and checkpoint separation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.model import UZ05Model, TerrainSpec
from uz05.spec import DEFAULT_MODEL_XML, MODEL_XML, REPO_ROOT, ORIGINAL_MODEL_XML, MESH_DIR
from train_uz05 import require_compatible_contract, write_contract


class NewModelTests(unittest.TestCase):
    def test_flat_and_stairs_replace_standalone_floor(self):
        for terrain in (TerrainSpec(), TerrainSpec(kind='stairs')):
            sim = UZ05Model(terrain, visual=False)
            m = sim.model
            self.assertEqual(np.sum(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), 1)
            self.assertEqual(m.geom('ground').friction[0], terrain.friction[0])
            self.assertEqual((m.nq, m.nv, m.nu, m.neq), (21, 20, 6, 4))
            self.assertEqual(m.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
            np.testing.assert_allclose(m.dof_armature[sim.wheel_dof_adr], .001)
            np.testing.assert_allclose(m.actuator_ctrlrange[sim.wheel_actuator_ids,1]
                                       *m.actuator_gear[sim.wheel_actuator_ids,0], 3.69, atol=1e-8)

    def test_new_contract_cannot_silently_resume_legacy_model(self):
        self.assertEqual(MODEL_XML, DEFAULT_MODEL_XML.resolve())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'checkpoint'
            write_contract(checkpoint, 'joint_tracking_v16', 1)
            require_compatible_contract(checkpoint, 'joint_tracking_v16')
            path = Path(str(checkpoint)+'_contract.json')
            c = json.loads(path.read_text())
            self.assertEqual(c['model_xml_path'], str(MODEL_XML))
            import hashlib
            for previous in (REPO_ROOT/'uz05_修正后(3).xml', ORIGINAL_MODEL_XML):
                c['model_xml_sha256'] = hashlib.sha256(previous.read_bytes()).hexdigest()
                path.write_text(json.dumps(c))
                # Wheel-map adaptation must not bypass either historical MJCF.
                with self.assertRaisesRegex(SystemExit, 'model_xml_sha256'):
                    require_compatible_contract(checkpoint, 'joint_tracking_v16', adapt_wheel_torque_model=True)

    def test_only_wheel_armature_changes_and_assets_resolve(self):
        old=ET.parse(ORIGINAL_MODEL_XML).getroot()
        new=ET.parse(DEFAULT_MODEL_XML).getroot()
        new.find('compiler').set('meshdir',old.find('compiler').get('meshdir'))
        for side in ('left','right'):
            joint=old.find(f".//joint[@name='{side}_J_link6_wheel']")
            self.assertEqual(joint.get('armature'),'0.01')
            joint.set('armature','0.001')
        self.assertEqual(ET.tostring(old),ET.tostring(new))
        for mesh in new.findall('asset/mesh'):
            if mesh.get('file'):self.assertTrue((MESH_DIR/mesh.get('file')).is_file())


if __name__ == '__main__':
    unittest.main()
