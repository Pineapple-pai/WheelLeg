"""从编译后的模型里提取腿机构几何（body 相对位置 + hinge 轴 + connect site）。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uz05.env import UZ05Env
np.set_printoptions(precision=5, suppress=True, linewidth=200)
env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0)
m = env.sim.model
names = [mujoco_name for mujoco_name in (
    "left_leg_mount", "left_link_right_leg_2", "left_link_right_leg_5",
    "left_link_right_leg_3", "left_link_right_leg_1", "left_link_right_leg_6",
    "left_right_wheel", "right_leg_mount", "right_link_right_leg_2",
    "right_link_right_leg_5", "right_link_right_leg_3", "right_link_right_leg_1",
    "right_link_right_leg_6", "right_right_wheel")]
import mujoco
for n in names:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    if bid < 0:
        print(f"{n}: NOT FOUND"); continue
    print(f"{n:26s} id={bid:3d} parent={m.body_parentid[bid]:3d} pos={m.body_pos[bid]}")
print("\n=== hinge 关节 ===")
for jid in range(m.njnt):
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE: continue
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    bid = m.jnt_bodyid[jid]
    print(f"{n:26s} body={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid):24s} "
          f"axis={m.jnt_axis[jid]} pos={m.jnt_pos[jid]} range={m.jnt_range[jid]}")
print("\n=== connect 等式约束的 site ===")
for eid in range(m.neq):
    if m.eq_type[eid] != mujoco.mjtEqualityType.mjEQ_CONNECT: continue
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, eid)
    s1, s2 = m.eq_obj1id[eid], m.eq_obj2id[eid]
    n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s1)
    n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s2)
    print(f"{n:26s} {n1} (body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s1])}, pos {m.site_pos[s1]})")
    print(f"{'':26s} {n2} (body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s2])}, pos {m.site_pos[s2]})")
env.close()
