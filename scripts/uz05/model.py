"""UZ-05 的 MuJoCo 模型封装：加载、索引、接触、地形、几何量测。"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .spec import MESH_DIR, MODEL_XML

HIP_JOINTS = (
    "left_J_chassis_link2",
    "left_J_chassis_link4",
    "right_J_chassis_link2",
    "right_J_chassis_link4",
)
WHEEL_JOINTS = ("left_J_link6_wheel", "right_J_link6_wheel")
WHEEL_CONTACT_GEOMS = ("left_wheel_collision", "right_wheel_collision")
BODY_CONTACT_GEOMS = ("chassis_collision", "gimbal_collision")
HIP_SITES = ("left_hip_axis_limit_site", "right_hip_axis_limit_site")
WHEEL_SITES = ("left_wheel_axis_limit_site", "right_wheel_axis_limit_site")

CONTACT_FORCE_THRESHOLD = 1.0


@dataclass
class TerrainSpec:
    """地形描述。``flat`` = 平面；``stairs`` = 沿 +X 的阶梯。"""

    kind: str = "flat"
    step_height: float = 0.06
    step_depth: float = 0.30
    num_steps: int = 8
    start_x: float = 0.8
    width: float = 4.0
    friction: tuple[float, float, float] = (1.0, 0.01, 0.001)


def _strip_visual_geoms(root: ET.Element) -> tuple[int, int]:
    """删掉纯显示用的 mesh geom 及其 mesh 资产，返回 (geom 数, mesh 数)。

    **为什么必须删**：``body_000..011.stl`` 每个 10 MB（≈20 万三角面），
    ``gimal_00X.obj`` 各 8 MB。MuJoCo 编译这些高模约需 **2 GB 内存/进程**，
    而 8 个 subproc 环境就是 16 GB —— 14 GB 的机器直接卡死（已实测）。
    它们 ``contype=0 conaffinity=0``、质量也走独立的 ``<inertial>``（已逐 body
    对比：删前删后 mass/inertia/ipos 完全一致），对物理零影响。
    需要渲染（回放/截图）时用 ``UZ05_VISUAL=1`` 保留。
    """
    worldbody = root.find("worldbody")
    removed_geoms = 0
    for parent in worldbody.iter():
        for geom in list(parent):
            if (
                geom.tag == "geom"
                and geom.get("type") == "mesh"
                and geom.get("contype") == "0"
                and geom.get("conaffinity") == "0"
            ):
                parent.remove(geom)
                removed_geoms += 1
    used = {g.get("mesh") for g in worldbody.iter("geom") if g.get("mesh")}
    removed_meshes = 0
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset):
            # 只删有 file 的（贵）；内联 vertex/face 的网格留着，反正很小
            if (mesh.tag == "mesh" and mesh.get("file")
                    and mesh.get("name") not in used):
                asset.remove(mesh)
                removed_meshes += 1
    return removed_geoms, removed_meshes


def _build_xml(terrain: TerrainSpec, visual: bool = False) -> str:
    """把 meshdir 与地形注入基础 MJCF，返回临时文件路径。"""
    tree = ET.parse(MODEL_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(MESH_DIR))
    compiler.set("angle", "radian")

    if not visual:
        _strip_visual_geoms(root)

    worldbody = root.find("worldbody")
    friction = " ".join(str(v) for v in terrain.friction)
    ET.SubElement(
        worldbody, "geom", name="ground", type="plane", size="20 20 0.1",
        friction=friction, rgba="0.25 0.28 0.31 1",
    )
    if visual:
        # Large, low-contrast checkerboard for replay only. These boxes are
        # visual-only, so the physical plane and its friction remain unchanged.
        tile_half_extent = 6.0
        tile_size = 0.60
        tile_z = 0.002
        tile_count = int(round(2.0 * tile_half_extent / tile_size))
        tile_colors = ("0.17 0.20 0.23 1", "0.28 0.31 0.34 1")
        for ix in range(tile_count):
            x = -tile_half_extent + (ix + 0.5) * tile_size
            for iy in range(tile_count):
                y = -tile_half_extent + (iy + 0.5) * tile_size
                ET.SubElement(
                    worldbody, "geom", name=f"replay_tile_{ix}_{iy}", type="box",
                    pos=f"{x:.5f} {y:.5f} {tile_z:.5f}",
                    size=f"{tile_size / 2:.5f} {tile_size / 2:.5f} 0.001",
                    rgba=tile_colors[(ix + iy) % 2], contype="0", conaffinity="0",
                )
    if terrain.kind == "stairs":
        for index in range(terrain.num_steps):
            height = terrain.step_height * (index + 1)
            x0 = terrain.start_x + terrain.step_depth * index
            ET.SubElement(
                worldbody, "geom", name=f"step_{index}", type="box",
                pos=f"{x0 + terrain.step_depth / 2:.5f} 0 {height / 2:.5f}",
                size=f"{terrain.step_depth / 2:.5f} {terrain.width / 2:.5f} {height / 2:.5f}",
                friction=friction, rgba="0.42 0.45 0.48 1",
            )

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=str(MODEL_XML.parent), delete=False
    )
    handle.write(ET.tostring(root, encoding="unicode"))
    handle.close()
    return handle.name


class UZ05Model:
    """模型 + 数据 + 索引 + 传感量的薄封装。"""

    def __init__(self, terrain: TerrainSpec | None = None, visual: bool | None = None):
        self.terrain = terrain or TerrainSpec()
        # 默认删掉高模显示网格（省 ~2GB/进程，见 _strip_visual_geoms）；
        # 要渲染时设 UZ05_VISUAL=1。
        if visual is None:
            visual = os.environ.get("UZ05_VISUAL", "0") not in ("0", "", "false", "False")
        self.visual = bool(visual)
        path = _build_xml(self.terrain, visual=self.visual)
        try:
            self.model = mujoco.MjModel.from_xml_path(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)

        jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        sid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

        self.hip_qpos_adr = [int(self.model.jnt_qposadr[jid(n)]) for n in HIP_JOINTS]
        self.hip_dof_adr = [int(self.model.jnt_dofadr[jid(n)]) for n in HIP_JOINTS]
        self.wheel_qpos_adr = [int(self.model.jnt_qposadr[jid(n)]) for n in WHEEL_JOINTS]
        self.wheel_dof_adr = [int(self.model.jnt_dofadr[jid(n)]) for n in WHEEL_JOINTS]
        self.wheel_geoms = [gid(n) for n in WHEEL_CONTACT_GEOMS]
        self.body_geoms = [gid(n) for n in BODY_CONTACT_GEOMS]
        self.hip_sites = [sid(n) for n in HIP_SITES]
        self.wheel_sites = [sid(n) for n in WHEEL_SITES]
        self.chassis_body = bid("chassis")
        self.wheel_bodies = [
            int(self.model.jnt_bodyid[jid(n)]) for n in WHEEL_JOINTS
        ]
        self.wheel_radius = 0.055
        self._contact_force = np.zeros(6, dtype=np.float64)

    # ---------------------------------------------------------------- 状态
    def reset(self, height: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = float(height)
        mujoco.mj_forward(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    # ---------------------------------------------------------------- 几何
    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[:3]

    @property
    def base_quat(self) -> np.ndarray:
        """MuJoCo 顺序 (w, x, y, z)。"""
        return self.data.qpos[3:7]

    @property
    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[:3]

    @property
    def base_ang_vel_world(self) -> np.ndarray:
        return self.data.qvel[3:6]

    def rotation_matrix(self) -> np.ndarray:
        """机体系 → 世界系的旋转矩阵。"""
        return self.data.xmat[self.chassis_body].reshape(3, 3)

    def body_frame(self, world_vector: np.ndarray) -> np.ndarray:
        return self.rotation_matrix().T @ np.asarray(world_vector, dtype=np.float64)

    def leg_lengths(self) -> np.ndarray:
        """左/右腿长（hip 站点 → 轮站点，tendon 口径）。"""
        return np.array(
            [
                np.linalg.norm(self.data.site_xpos[h] - self.data.site_xpos[w])
                for h, w in zip(self.hip_sites, self.wheel_sites)
            ],
            dtype=np.float64,
        )

    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.hip_qpos_adr], dtype=np.float64)

    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self.hip_dof_adr], dtype=np.float64)

    def wheel_velocities(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self.wheel_dof_adr], dtype=np.float64)

    # ---------------------------------------------------------------- 接触
    def contact_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """返回 (轮法向力 L/R, 轮接地标志 L/R, 机身接地标志, 腾空标志)。"""
        wheel_force = np.zeros(2, dtype=np.float64)
        wheel_hit = np.zeros(2, dtype=np.float64)
        body_hit = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            mujoco.mj_contactForce(self.model, self.data, index, self._contact_force)
            normal = abs(float(self._contact_force[0]))
            for side, geom in enumerate(self.wheel_geoms):
                if geom in (geom1, geom2):
                    wheel_force[side] += normal
                    if normal > CONTACT_FORCE_THRESHOLD:
                        wheel_hit[side] = 1.0
            if body_hit == 0.0 and any(g in (geom1, geom2) for g in self.body_geoms):
                if normal > CONTACT_FORCE_THRESHOLD:
                    body_hit = 1.0
        airborne = 1.0 if (wheel_hit.sum() == 0.0 and body_hit == 0.0) else 0.0
        return wheel_force, wheel_hit, body_hit, airborne

    # ------------------------------------------------------- 域随机化
    def nominal(self) -> dict:
        """记录未随机化的基准值，供每 episode 恢复。"""
        return {
            "mass": self.model.body_mass.copy(),
            "inertia": self.model.body_inertia.copy(),
            "ipos": self.model.body_ipos.copy(),
            "friction": self.model.geom_friction.copy(),
            "damping": self.model.dof_damping.copy(),
        }

    def apply_randomization(self, base: dict, params: dict) -> None:
        """把采样到的域随机化参数写进 MuJoCo 模型。

        骨骼质量、惯量、质心、轮子摩擦、关节阻尼都在这里改；
        控制器增益（关节 PD、电调力矩常数）由环境侧按同一批参数缩放。
        """
        m = self.model
        m.body_mass[:] = base["mass"]
        m.body_inertia[:] = base["inertia"]
        m.body_ipos[:] = base["ipos"]
        m.geom_friction[:] = base["friction"]
        m.dof_damping[:] = base["damping"]

        mass_scale = params["base_mass_scale"][0]
        m.body_mass[self.chassis_body] *= mass_scale
        m.body_inertia[self.chassis_body] *= params["base_inertia_scale"][0]
        m.body_ipos[self.chassis_body] += np.array([
            params["base_com_x"][0], params["base_com_y"][0], params["base_com_z"][0]
        ])
        fric = params["wheel_friction_scale"][0]
        for geom in self.wheel_geoms:
            m.geom_friction[geom, 0] *= fric          # 滑动摩擦
        damp = params["joint_damping_scale"][0]
        m.dof_damping[self.hip_dof_adr] *= damp
        m.dof_damping[self.wheel_dof_adr] *= damp
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- 地形
    def support_height(self, x: float, y: float) -> float:
        """脚下支撑面高度（解析式，仅用于相对高度观测）。"""
        if self.terrain.kind != "stairs":
            return 0.0
        if x < self.terrain.start_x:
            return 0.0
        level = int((x - self.terrain.start_x) // self.terrain.step_depth) + 1
        level = min(level, self.terrain.num_steps)
        return self.terrain.step_height * level

    def terrain_scan(self, base_x: float, forward_x: np.ndarray) -> np.ndarray:
        """沿机体前方采样地形高度，返回相对当前支撑面的高度。

        平面地形下恒为 0 → 观测分布不随地形切换而突变。
        """
        support = self.support_height(base_x, 0.0)
        offsets = np.linspace(0.10, 0.90, forward_x.size)
        if self.terrain.kind != "stairs":
            return np.zeros(forward_x.size, dtype=np.float64)
        heights = np.array(
            [self.support_height(base_x + float(off), 0.0) for off in offsets],
            dtype=np.float64,
        )
        return heights - support
