"""腿 + 轮协同动态平衡控制器（UZ-05 零指令站定 / 姿态保持）。

分工（与最终目标一致）
----------------------
* **腿**：低频姿态、机体高度、支撑力、扰动恢复。
  髋关节共同位置偏置 → 轮心相对机体前后移动 → 前后移动 CoP → 直接给出
  与姿态角同量级（~30 Nm/rad）的恢复力矩。
* **轮**：高频 pitch 修正、前后移动、速度控制。
  共模电流 → 地面反作用力 → 机体前后加速度；带宽高，负责 pitch 速率阻尼。

实测符号（MuJoCo 标定，务必遵守）
--------------------------------
===========  ==================  ===================
输入          俯仰角加速度         机体前向加速度
===========  ==================  ===================
正轮电流 I     ω̇ > 0（后仰）        v̇x < 0（向前加速）
正腿偏置 u     ω̇ < 0（前倾）        v̇x > 0（向后加速）
===========  ==================  ===================

因此：姿态（θ, ω）反馈两路都取**负号**；位置/速度（x, vx）反馈两路都取
**正号**（腿做低频位置主力，轮做高频阻尼 + 少量位置修正）。

增益量纲
--------
* ``kl_*`` 输出**腿关节偏置（rad）**：kl_p rad/rad, kl_x rad/m …
* ``kw_*`` 输出**轮电流（A）**：kw_p A/rad, kw_x A/m …
* ``ky_p`` 输出**差模轮电流（A）**：A/(rad/s)

标定值见 ``scripts/coord_gains.json``（bench: pitch RMS 0.09°、峰值漂移 0.8 cm）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoordinatedBalanceParams:
    """协同平衡控制器增益（默认值为 MuJoCo 标定结果）。

    分工（与目标一致）
    ------------------
    * **腿**（``kl_*`` / 输出关节偏置 rad）：低频。姿态角速度阻尼、前后位置、
      位置积分、CoP 前后移动（支撑力方向），是位置环的主力。
    * **轮**（``kw_*`` / 输出电流 A）：高频。pitch 角 + 角速度阻尼、速度、
      少量位置修正，是高频 pitch 修正的主力。

    标定路径
    --------
    1. 符号辨识（``scripts/diag_osc.py identify``）：正轮电流 → 后仰；
       正腿偏置 → 前倾。两路姿态反馈都取负号。
    2. 结构化网格 + (1+1)-ES（``tune_coord3.py`` → ``optimize_coord_robust.py``
       → ``tune_impulse.py``）。
    3. 130 工况验证（``diag_full_compare.py``）：40 个课程内站定 + 14 个压力
       工况（初始冲量 / 大倾角），**54/54 全通过**。

    ⚠️ 两条实测教训
    --------------
    1. **腿位置环不能太硬**：``kl_x=8, kl_i=20`` 会在位置误差稍大时把腿偏置
       顶到 ±0.35 rad 行程上限（轮心前后行程只有 ±8.3 cm），随后进入极限环。
       现在的取值留足了行程余量 —— 这比加大增益更重要。
    2. **输出一阶低通必须有**：不过滤时 PD 平衡律在 125 Hz 上超过 Nyquist
       边界，共模电流逐帧 ±8 A 翻转（``dI ≈ 12 A/step``），姿态看着稳但实机
       会烧电调 / 打齿。``filter_alpha=0.9`` 兼顾响应与平滑。
    """

    # --- 腿：低频（姿态阻尼 + 位置/速度 + 积分）---
    kl_p: float = 0.6         # rad/rad      腿：姿态（保证腿参与姿态控制）
    kl_d: float = 0.0142      # rad/(rad/s)  腿：姿态速率阻尼
    kl_v: float = 1.13        # rad/(m/s)    腿：前后速度
    kl_x: float = 3.10        # rad/m        腿：前后位置
    kl_i: float = 5.61        # rad/(m·s)    腿：位置积分（消除稳态漂移）
    # 腿长快速变化时，差模动作会给机体一个短暂的前后冲量；在这段
    # 过渡期临时提高位置/速度保持增益，让腿在获得明显位移之前就把
    # 支撑点拉回目标位置。静止或腿长已稳定时缩放为 1，不改变原站定
    # 标定结果。leg_length_rate 来自腿长编码器/FK，因此实物可直接复用。
    height_transition_position_scale: float = 1.30
    height_transition_velocity_scale: float = 1.25
    height_transition_rate_threshold: float = 0.010  # m/s，满增益阈值
    # --- 轮：高频（pitch 阻尼 + 速度 + 位置）---
    kw_p: float = 21.17       # A/rad        轮：高频 pitch 修正
    kw_d: float = 15.82       # A/(rad/s)    轮：pitch 速率阻尼
    kw_v: float = 0.0         # A/(m/s)      轮：前后速度
    kw_x: float = 18.66       # A/m          轮：前后位置
    kw_i: float = 0.0         # A/(m·s)      轮：位置积分
    ky_p: float = 10.0        # A/(rad/s)    轮：偏航阻尼（差模）
    # 输出限幅（物理单位）
    leg_offset_limit: float = 0.35    # rad，= ACTION_SPEC hip_position_offset
    current_limit: float = 8.0        # A，= ACTION_SPEC wheel_common
    diff_current_limit: float = 6.0   # A，= ACTION_SPEC wheel_differential
    integral_limit: float = 0.30      # m·s，位置积分抗饱和
    # 积分项对腿偏置 / 轮电流的**独立限幅**。nominal 工况下
    # ``kl_i·integral`` 稳态只需 ≈ -0.074 rad，所以限到 0.08 与不限幅结果
    # 完全一致（实测逐帧相同）；保留这个旋钮是为了实机调试时保证任何时刻
    # 都留有回弹余量。默认放到 0.35（≈ 满行程）等于不限幅。
    integral_leg_limit: float = 0.35
    integral_wheel_limit: float = 0.0
    # ==================================================================
    # ★ 腿长控制环（腿负责机体高度）
    #
    # 机构事实（实测标定，`scripts/diag_leg_ws.py`）：
    #   腿是 2-DOF 闭环五连杆。**腿长只取决于 q2 与 q4 之差**：
    #       L ≈ 0.1841 + 0.1279·(q2 - q2_nom) - 0.1195·(q4 - q4_nom)
    #   即 dL/d(q2-q4) ≈ -0.1237 m/rad。
    #   旧的"4 个关节共同偏置"只改 q2 和 q4 的**和**，腿长完全不变
    #   （实测 trim ±0.35 rad 腿长恒定 0.1842 m）—— 这就是为什么以前
    #   根本没能改变腿长。
    #
    # 因此高度环输出的是**差模关节偏置** d：
    #       q2 += d/2, q4 -= d/2  （左右腿相同）
    #   与平衡器的共模偏置 u_leg 正交，互不干扰。
    # ==================================================================
    # ★ 腿长环用**速率受限积分**（rad diff 每秒每米误差），不用开环前馈：
    #   机构在奇异位形附近梯度剧变，前馈实测会把腿往反方向带。
    height_rate_gain: float = 20.0    # (动作单位/s) per m；快速跟踪课程
    height_retract_rate_gain: float = 3.0
    # 腿长速度反馈（action / (m/s)），抑制伸缩末端回弹；值很小即可，
    # 因为 leg_length_rate 已经是物理 m/s，而差模每步还有速率限幅。
    height_rate_damping: float = 0.05
    # 收腿方向单独的速度阻尼。收腿到低位时机构惯性更容易造成下冲，
    # 不能为了提高伸腿响应而把两方向绑成同一个阻尼值。
    height_retract_rate_damping: float = 0.05
    # 只在最低腿长目标附近提高收腿制动阻尼；中高腿长和伸腿响应不受影响。
    height_low_target_brake_damping_scale: float = 1.0
    height_rate_brake_threshold: float = 0.02  # m/s，进入惯性制动区
    # 目标本身在网页/训练中按斜坡变化时，不能把每一帧的参考更新当成
    # “新目标”清空积分器。参考斜坡的单帧变化通常是 2~6 mm；只有真正的
    # 大跳变才需要重置历史积分。
    height_reference_jump_reset_m: float = 0.020
    # 只有进入目标附近的制动带才撤掉同向位置驱动；远离目标时即使腿正在
    # 朝目标运动，也必须继续给驱动，否则动态目标会被执行器速度甩开。
    height_brake_error_m: float = 0.020
    # 目标腿长的速度前馈比例。0 表示只用位置环；部署端可逐步增大。
    height_target_rate_feedforward_scale: float = 0.0
    # 目标附近的保持带：远距离保持快速带宽，进入该误差后切换到更小的
    # 位置增益/差模速率，避免连杆在目标两侧来回打摆。
    height_hold_error_m: float = 0.010
    height_hold_rate_limit: float = 0.35
    height_hold_rate_gain: float = 15.0
    height_integral_sign_decay: float = 0.0  # 穿过目标时清除旧方向积分
    # ★ 伸展/收缩的**速率限幅**。实测这是最关键的安全参数：
    #   快速伸展会在过渡过程中把机体推倒（0.28 m 目标在 461 步时触发
    #   station_limit），而**稳态**在伸展位形下是稳的（固定 trim 到 0.282 m
    #   可以无限站住）。把速率压到 0.25 动作单位/s（≈4.6 mm/s 腿长变化）
    #   后过渡扰动降到与站定同一量级。代价是 1 cm 腿长变化需要 ~1.5 s。
    height_rate_limit: float = 0.90   # 动作单位/s，差模最大变化率
    # 收腿方向的机构惯性更容易把腿推过下限；单独限速并关闭收腿前馈，
    # 由误差积分慢慢收回，避免从 0.27 m 直接降到 0.15 m 时冲到 0.13 m。
    height_retract_rate_limit: float = 0.12  # 动作单位/s
    height_retract_slow_rate_limit: float = 0.04
    # Start the low-speed approach early enough for mechanical leg inertia to
    # dissipate before the ±3 mm deadband.  At 30 mm the old loop could still
    # cross the target with a large residual leg speed under DR.
    height_retract_slow_error_m: float = 0.060
    height_retract_feedforward_scale: float = 0.0
    height_deadband_m: float = 0.003  # m，腿长误差死区（±3 mm 内不动）
    height_kp: float = 0.0            # 保留字段（未使用，供实机调）
    height_kd: float = 0.0            # 保留字段（未使用，供实机调）
    height_ki: float = 0.0            # 保留字段（未使用，供实机调）
    # 差模偏置限幅（rad）。**默认 0 = 腿长环关闭**，保持已验证的站定行为
    # 完全不变；要用腿长变化时必须显式打开：
    #     env.coordinated.enable_leg_length(lo=0.15, hi=0.31)
    # ±1.35 rad 才能覆盖 0.15~0.35 m 全行程，会与关节软限位冲突，
    # 所以 `set_leg_range` 会按目标范围反算限幅。
    # 0 = 关闭；enable_leg_length 会按目标范围设定。
    # 当前训练范围 0.15~0.27 m 需要约 0.19 动作单位；上限留出余量，
    # 但不允许网页/脚本误把超出训练域的目标送进闭环。
    height_diff_limit: float = 0.0
    # 几何增益只用于估算差模的安全范围，负载、关节 PD 刚度和共模平衡
    # 会让实际静态增益变小。这个余量必须足够覆盖高腿长端的稳态补偿，
    # 否则差模一到限幅，积分器再大也只能留下数毫米到十几毫米误差。
    height_diff_limit_margin_scale: float = 2.20
    # ★ 积分上限必须够大：稳态误差 e_ss ≈ rate_limit/(2·gain)。
    #   实测 rate_limit=0.25、gain=4、integral_limit=0.25 时 e_ss ≈ 62 mm；
    #   把 gain 提到 8、integral_limit 提到 2.0，e_ss 降到 ~15 mm 以内。
    height_integral_limit: float = 2.0    # m·s
    # 前馈（逆运动学增益）
    # ★ 实测（diag_leg_sign.py，闭环平衡保持中）：差模偏置 d 与腿长的关系
    #       d = −0.15 → 0.1501 m ;  d = 0 → 0.1842 m ;  d = +0.15 → 0.2237 m
    #       d = +0.30 → 0.2671 m ;  d = +0.50 → 0.2819 m（饱和）
    #   即 **正 d 伸长**，静态增益 ≈ +0.26 m/动作单位（≈0.74 m/rad 关节）。
    #   更关键的是这一路**几乎不耦合姿态**：整段扫描 pitch ≈ 0.00°、
    #   漂移峰值 0.002 cm、左右腿长差 0.5 mm。所以腿长可以独立控制。
    # ★ 已按 `LEG_DIFF_ACTION_SCALE = 0.70` 重标：机构实测关节差模 0.2/0.4/0.6 rad
    #   → 腿长 0.2405/0.2991/0.3538 m，线性拟合 ≈ 0.28 m/0.2rad ⇒ 每 rad 1.4 m。
    #   新缩放下 1 个动作单位 = 0.70 rad ⇒ 0.98 m/动作；但负载/饱和会降低有效值，
    #   这里取保守的 0.55，让积分环去补剩余误差。
    leg_length_gain: float = 0.55     # m / 动作单位（差模）
    # ★ 前馈缩放。实测（`diag_leg_range_check.py` 的 q2/q4 反推）真实静态增益
    #   约为 0.30 m/动作单位，而 0.55 是按机构斜率估的**过激进**值：用它做
    #   全量前馈会把腿直接推到行程尽头（网页把目标从 0.30 拉到 0.150 时，
    #   前馈 -0.19 动作单位 → 关节差模 -0.133 rad，腿长过冲到 0.1327 m，
    #   触发 tilt_limit）。改成只承担 40%，剩余误差交给积分环平滑收敛。
    leg_feedforward_scale: float = 1.00
    # 关节差模每弧度的腿长增益 = 0.26 / 0.35（POS_ACTION_SCALE）
    leg_length_gain_per_rad: float = 1.40   # m/rad
    nominal_leg_length: float = 0.18413
    # ==================================================================
    # ★★ 共模 / 差模的**交叉耦合**（实测非零，必须解耦）
    #
    # 只给差模 d（q2 += d/2, q4 −= d/2）会**同时**改变腿长和腿的前后角度：
    #     dL/d(差模)    = −0.1237 m/rad
    #     d|x|/d(差模)  = +0.2375 m/rad   ← 很大
    # 只给共模 c（q2 += c, q4 += c）几乎不改腿长，但改前后角度：
    #     dL/d(共模)    ≈ 0
    #     d|x|/d(共模)  ≈ 0.218 m/rad（符号见下）
    #
    # 所以差模**不能**直接当"腿长控制"用 —— 第一版腿长环就是因此一上就把
    # 机器人推倒（它同时把支撑点挪了 0.2375/0.1237 ≈ 1.9 倍腿长位移）。
    # 必须解这个 2×2 线性系统，求出同时满足
    # 「腿长 = L_ref」与「支撑点前后 = u_leg」的 (共模, 差模)：
    #     ΔL      = kk·c + kd·d
    #     u_leg   = kc·c + kx·d
    # ==================================================================
    leg_diff_to_angle: float = 0.2375    # m/rad，差模 → 前后角度（|·|）
    leg_common_to_angle: float = -0.218  # m/rad，共模 → 前后角度（带符号）
    # 显式前馈 trim（归一化动作单位，调试/标定用；正常运行为 0）
    # c_trim 改腿的前后角度（不改腿长），d_trim 主要改腿长。
    common_trim: float = 0.0
    diff_trim: float = 0.0
    # 腿长目标可达范围（训练/回放统一为 0.15~0.27 m）
    leg_length_target_min: float = 0.150
    leg_length_target_max: float = 0.270
    # ★ 腿长环的默认目标 = **负载稳态腿长**，不是几何标称腿长。
    #   关节 PD 是有限刚度，负载下腿会再压缩 ~0.2 mm（实测
    #   `nominal_leg_length=0.18413` 而实际稳态 ≈0.1839）。若把目标设成
    #   nominal，积分环会一直追一个到不了的误差，缓慢把腿拉长直到
    #   `height_limit`（实测 374 步摔倒）。所以默认目标用实测负载值。
    leg_length_ref_default: float = 0.1839
    # ★ 输出一阶低通系数（1 = 不滤波）。PD 形式的平衡律在 125 Hz 上对
    #   pitch_rate 的增益会超过 Nyquist 稳定性边界，表现为共模电流逐帧
    #   ±8 A 翻转（实测 dI ≈ 12 A/step）——机体姿态看着稳，但电流在颤振，
    #   实机上就是电调/电机发热与齿轮冲击。滤波把它压回连续控制。
    filter_alpha: float = 0.9
    # 腿差模独立低通；轮子仍使用 filter_alpha 的高频姿态带宽。
    height_filter_alpha: float = 0.9


class CoordinatedBalance:
    """协同平衡控制器：输入机体状态，输出**归一化动作**（与策略同一接口）。

    返回 ``(action6, info)``：``action6[0]`` 共模轮电流、``[1]`` 差模轮电流、
    ``[2:6]`` 4 个腿关节的共同位置偏置。
    """

    def __init__(self, params: CoordinatedBalanceParams | None = None,
                 leg_scale: float = 0.35, wheel_scale: float = 8.0,
                 diff_scale: float = 6.0):
        self.p = params or CoordinatedBalanceParams()
        self.leg_scale = float(leg_scale)
        self.wheel_scale = float(wheel_scale)
        self.diff_scale = float(diff_scale)
        self.integral = 0.0
        # ★ 腿长环状态
        self.leg_length_ref = float(self.p.leg_length_ref_default)
        self.height_integral = 0.0
        self.height_last_error = 0.0
        # Keep the unsaturated encoder error as well as the deadbanded error.
        # If the raw error crosses zero while it is inside the ±deadband, the
        # old implementation stored only 0.0 and missed the sign change.  The
        # retract integral then kept pushing the legs below the low-height
        # target under domain randomization.
        self.height_last_raw_error = 0.0
        self.height_ref_direction = 0.0
        self.height_reference_delta_m = 0.0
        self.height_diff = 0.0
        self.height_feedforward = 0.0
        self.last = {"current": 0.0, "leg_offset": 0.0, "diff_current": 0.0,
                     "height_diff": 0.0, "leg_length_ref": self.leg_length_ref}

    def reset(self) -> None:
        self.integral = 0.0
        self.height_integral = 0.0
        self.height_last_error = 0.0
        self.height_last_raw_error = 0.0
        self.height_ref_direction = 0.0
        self.height_reference_delta_m = 0.0
        self.height_diff = 0.0
        self.height_feedforward = 0.0
        self.leg_length_ref = float(self.p.leg_length_ref_default)
        self.last = {"current": 0.0, "leg_offset": 0.0, "diff_current": 0.0,
                     "height_diff": 0.0, "leg_length_ref": self.leg_length_ref}
        self._filtered = None

    def set_leg_length_ref(self, value: float) -> None:
        """设置腿长目标（m）。会被限到 ``leg_length_target_min/max``。

        同时按**实测静态增益**把差模直接前馈到目标附近（一次到位），
        让积分环只需修小误差。为什么需要：纯积分收敛受机构动力学限制，
        实测从标称 0.184 走到 0.155 需要 ~424 步（3.4 s），在 1000 步的
        episode 里一半时间都在暂态，腿长奖励被大量吃掉。
        """
        p = self.p
        new_ref = float(np.clip(
            value, self.p.leg_length_target_min, self.p.leg_length_target_max))
        old_ref = float(self.leg_length_ref)
        if abs(new_ref - old_ref) < 1e-9:
            return   # 目标未变：不动积分器（否则每步清零 → 永远收敛不了）
        ref_delta = new_ref - old_ref
        self.height_reference_delta_m = ref_delta
        # 网页和训练都使用小步斜坡更新目标。仅按“单步变化 > 4 mm”判断
        # 会漏掉方向反转，使上一段伸腿积分继续作用在下一段收腿上，表现
        # 为先冲过目标 20~30 mm 再回弹。方向一旦反转，立即清掉旧积分。
        if self.height_ref_direction * ref_delta < 0.0:
            self.height_integral = 0.0
            self.height_last_error = 0.0
            self.height_last_raw_error = 0.0
        self.height_ref_direction = float(np.sign(ref_delta))
        self.leg_length_ref = new_ref
        # 只记录前馈目标；实际移动由 _step_leg_modes 按速率限幅执行。
        # ⚠️ 不能在这里直接跳到目标：实测瞬跳 +0.27 动作单位会在 61 步内把
        #    机体推倒（pitch 峰值 6.15°、漂移 13.9 cm）。
        self.height_feedforward = 0.0
        if self.p.height_diff_limit > 0.0:
            # ⚠️ 静态增益在接近机构饱和点时会**高估**所需差模：实测命令
            #    0.26 m 时前馈算出 0.292，而机构在 d≈0.40 就饱和、再往上
            #    腿长恒为 0.2816 m。过驱动会把腿顶在行程末端，失去恢复权限
            #    → 失稳（实测 0.26 目标在各种速率下都在 200~540 步摔倒）。
            #    这里按"机构实测饱和点"钳制前馈，只让积分环补剩余误差。
            ff_scale = self.p.leg_feedforward_scale
            # 伸长可以用几何前馈快速到位；收腿时前馈会叠加机构惯性，
            # 改为纯反馈并交给更低的 retract 速率限制。
            if new_ref < old_ref - 1e-9:
                ff_scale *= self.p.height_retract_feedforward_scale
            ff = (ff_scale
                  * (self.leg_length_ref - self.p.leg_length_ref_default)
                  / max(self.p.leg_length_gain, 1e-6))
            self.height_feedforward = float(np.clip(
                ff, -self.p.height_diff_limit, self.p.height_diff_limit))
            # 高度目标斜坡会每步更新 ref；小步更新不能清空积分器，
            # 否则训练中的 command ramp 永远没有稳态积分，表现为低收敛。
            if abs(new_ref - old_ref) >= p.height_reference_jump_reset_m:
                self.height_integral = 0.0
                self.height_last_error = 0.0
                self.height_last_raw_error = 0.0

    def snap_leg_length(self) -> None:
        """把差模**立即**置到当前目标（用于 episode 开始时预置）。

        为什么需要：`set_leg_length_ref` 只记录前馈目标、由 `_step_leg_modes`
        按速率限幅靠近（避免瞬跳把机体推倒）。但**episode 起点**是唯一可以
        瞬跳的时机 —— 此时还没有平衡状态可破坏，预置到位可以省掉整段伸展
        暂态（实测能消掉 ~2 cm 的起始漂移）。
        """
        self.height_diff = float(self.height_feedforward)

    def enable_leg_length(self, lo: float = 0.15, hi: float = 0.31,
                          ref: float | None = None) -> None:
        """打开腿长控制环（腿负责机体高度）。

        默认**关闭**（`height_diff_limit = 0`），以保证已验证的站定行为不被
        改变。打开后腿长会在 [lo, hi] 内跟踪 `set_leg_length_ref()`。
        """
        self.set_leg_range(lo, hi)
        if ref is not None:
            self.set_leg_length_ref(ref)

    def disable_leg_length(self) -> None:
        """关闭腿长环并回到标称腿长（差模归零）。"""
        self.p.height_diff_limit = 0.0
        self.height_diff = 0.0
        self.leg_length_ref = float(self.p.leg_length_ref_default)

    def set_leg_range(self, lo: float, hi: float) -> None:
        """设置腿长目标可达范围，并按实测静态增益反算差模限幅。

        单位说明（踩过坑）
        ----------------
        差模 `height_diff` 的量纲是**归一化动作单位**（±1 ↔ ±0.35 rad 关节），
        `leg_length_gain = 0.26 m/动作单位` 是实测静态增益。所以::

            limit[动作单位] = max(|hi − L0|, |L0 − lo|) / leg_length_gain

        一开始把 0.26 当成 m/rad 用，限幅差了 1/0.35 = 2.9 倍，表现为目标腿长
        永远追不上、积分一路爬到限幅（实测 0.28 m 目标在 461 步后失稳）。
        """
        self.p.leg_length_target_min = float(min(lo, hi))
        self.p.leg_length_target_max = float(max(lo, hi))
        L0 = self.p.leg_length_ref_default
        span = max(abs(self.p.leg_length_target_max - L0),
                   abs(L0 - self.p.leg_length_target_min))
        # ★ 限幅不能再用"保守 gain 反推"：leg_length_gain 是 0.55（保守估计），
        #   反推出来的 0.273 动作单位不够覆盖 0.15~0.32 m，导致高目标处被裁剪、
        #   跟踪误差 -62 mm。`leg_length_gain_per_rad`（1.40 m/rad）是从机构
        #   几何实测的，用它换算 → 0.152 m 行程只需 0.109 rad 关节差 →
        #   0.155 动作单位。负载闭环重扫表明 0.300 m 目标实际需要
        #   约 0.202 动作单位。带载闭环复测表明，0.15~0.27 m 的高端还需要
        #   一点稳态补偿余量，因此把系数从 1.75 提到 2.20；对当前范围得到
        #   约 0.193 动作单位，而不是把 0.270 m 命令静默卡在约 0.265 m。
        #   关节侧总差模 = limit × LEG_DIFF_ACTION_SCALE，与共模叠加后
        #   仍不超过 ±0.35 rad 的舒适区间。
        per_action = self.p.leg_length_gain_per_rad * 0.70  # m / 动作单位（几何）
        self.p.height_diff_limit = float(np.clip(
            self.p.height_diff_limit_margin_scale * span
            / max(per_action, 1e-6), 0.15, 0.45))

    def _step_leg_modes(self, leg_error: float, u_leg: float,
                        dt: float, leg_length_rate: float = 0.0) -> tuple[float, float]:
        """腿长环 → 差模；腿的前后角度 = 平衡器的 u_leg（共模）− 差模的副作用。

        分工（关键：不要动平衡器已经标定好的闭环）
        ------------------------------------------
        * **共模 c** 是平衡器一直在用的那个量（`kl_*` 那套增益就是按它标的）。
          所以 c 必须**直接**等于 u_leg（必要时减去差模带来的角度副作用），
          **绝不能**再乘/除任何系数 —— 第一版解耦把 c 缩放成 u_leg/kc
          （kc≈−0.218 ⇒ 放大 4.6 倍），直接把闭环相位裕度毁掉，
          实测 7.98 Hz 自激、1.5 秒内摔倒。
        * **差模 d** 是新增的腿长自由度，用**速率受限积分**驱动：
          ``ḋ = gain · e_L``（e_L = 实测腿长 − 目标腿长）。
          只用测量反馈，天然适应机构的强非线性（开环逆解在奇异位形附近
          会把腿往反方向带，实测）。
        * 差模会顺带改变腿的前后角度，因此共模要做**前馈补偿**：
          ``c = u_leg − (kx/kc)·d``。这只影响腿长环引入的那一项，
          平衡器原本的 u_leg 映射关系不变。
        """
        p = self.p
        kx = p.leg_diff_to_angle
        kc = p.leg_common_to_angle
        # ★ 符号：实测**正差模 ⇒ 伸长**（d=+0.15 → 腿长 +40 mm），
        #   所以 e_L > 0（比目标长）时要**减小** d：ḋ = −gain·e_L。
        #   （第一版写反了，结果"要伸长"却一路缩短到 0.14 m 撞 height_limit。）
        raw_leg_error = float(leg_error)
        if abs(raw_leg_error) < p.height_deadband_m:
            leg_error = 0.0
        retracting = leg_error > p.height_deadband_m
        hold_band = max(p.height_deadband_m, p.height_hold_error_m)
        in_hold_band = abs(raw_leg_error) <= hold_band
        if retracting:
            # 误差较大时保持响应速度；提前进入目标附近的低速区，
            # 把机械惯性留在目标前的 60 mm 内消化掉。
            rate_limit = (p.height_retract_slow_rate_limit
                          if leg_error <= p.height_retract_slow_error_m
                          else p.height_retract_rate_limit)
        else:
            rate_limit = p.height_rate_limit
        rate_gain = (p.height_retract_rate_gain if retracting
                     else p.height_rate_gain)
        if in_hold_band:
            rate_limit = min(rate_limit, max(0.0, p.height_hold_rate_limit))
            rate_gain = max(0.0, p.height_hold_rate_gain)
        rate_cap = rate_limit * dt
        # ★ 真 PI，不是纯比例：之前写成 d = −gain·e_L（只有 P），
        #   在 rate_limit 约束下必然留下稳态误差 e_ss ≈ rate_limit/(2·gain)
        #   —— 实测 50 mm，且抬高增益也压不下去（因为 d 本身被 rate 限住）。
        #   正确结构：积分器承担稳态误差（e_L 不上限幅），前馈/比例只负责
        #   快速逼近。
        #   e_L < 0（比目标短）⇒ 要伸长 ⇒ 积分项增大。
        # 误差换号表示已经穿过目标；衰减旧积分，避免惯性过冲后积分器
        # 仍持续把腿推向原方向，造成“收敛过低/上下摆动”。
        # Detect a real crossing from the encoder error, not the deadbanded
        # value.  A 3 mm deadband must not hide the transition from
        # ``+epsilon`` to ``-epsilon``; otherwise the retract integral survives
        # the crossing and drives a low-leg overshoot.
        if (abs(raw_leg_error) > 1e-9
                and abs(self.height_last_raw_error) > 1e-9
                and self.height_last_raw_error * raw_leg_error < 0.0):
            self.height_integral *= float(np.clip(
                p.height_integral_sign_decay, 0.0, 1.0
            ))
        self.height_integral = float(np.clip(
            self.height_integral - leg_error * dt,
            -p.height_integral_limit, p.height_integral_limit))
        self.height_last_error = float(leg_error)
        self.height_last_raw_error = raw_leg_error
        # 前馈项以**速率限幅**靠近目标（见 set_leg_length_ref 的说明）
        ff_step = float(np.clip(self.height_feedforward - self.height_diff,
                                -rate_cap, rate_cap))
        # 目标斜坡速度前馈。它只在调用方刚刚更新过参考时生效，并在本次
        # 控制周期结束后清零；这样网页端每帧更新目标时可直接传递期望腿速，
        # 而静态目标不会遗留一个持续推动的“隐藏速度命令”。
        target_rate = (self.height_reference_delta_m / max(dt, 1e-6))
        target_rate_step = float(np.clip(
            p.height_target_rate_feedforward_scale
            * target_rate / max(p.leg_length_gain, 1e-6) * dt,
            -rate_cap, rate_cap,
        ))
        position_step = (ff_step
                         + rate_gain * self.height_integral * dt
                         - rate_gain * leg_error * dt)
        rate_damping = (p.height_retract_rate_damping if retracting
                        else p.height_rate_damping)
        step = (position_step
                + target_rate_step
                - rate_damping * float(leg_length_rate))
        # 机械腿仍在向目标方向运动时，只在目标附近进入制动带。旧逻辑在
        # “只要朝目标运动”时把 step 整体清零，动态参考会因此持续落后；
        # 这里保留目标速度前馈和测得速度阻尼，让远距离跟踪继续加速，
        # 近距离才撤掉位置驱动、消化机构惯性。
        brake = max(0.0, float(p.height_rate_brake_threshold))
        moving_toward = (raw_leg_error * float(leg_length_rate) < 0.0
                         and abs(float(leg_length_rate)) > brake)
        low_target_braking = bool(
            self.leg_length_ref <= p.leg_length_target_min + 0.005
            and leg_length_rate < -brake
            and abs(raw_leg_error) <= max(
                p.height_deadband_m, p.height_brake_error_m)
        )
        if ((moving_toward or low_target_braking)
                and abs(raw_leg_error) <= max(
                    p.height_deadband_m, p.height_brake_error_m)):
            brake_band = max(p.height_deadband_m, p.height_brake_error_m)
            brake_span = max(brake_band - p.height_deadband_m, 1e-6)
            # Keep a reduced position correction in the brake band. Setting
            # the whole step to zero was safe but left a permanent 8~15 mm
            # bias whenever the mechanism was still moving toward the target.
            position_scale = float(np.clip(
                (abs(raw_leg_error) - p.height_deadband_m) / brake_span,
                0.0, 1.0,
            ))
            brake_damping = rate_damping
            if low_target_braking:
                brake_damping *= max(
                    1.0, float(p.height_low_target_brake_damping_scale)
                )
            step = (target_rate_step
                    + position_scale * position_step
                    - brake_damping * float(leg_length_rate))
        # 速率限幅：即使误差很大，差模也不超过 height_rate_limit 每秒
        step = float(np.clip(step, -rate_cap, rate_cap))
        # ★ 抗积分饱和：目标超出当前命令范围/闭环恢复权限时，
        #   继续积分只会让环在饱和点附近来回冲。规则：已经在限幅上、且这一步还要
        #   往同方向推 ⇒ 冻结。
        at_limit = ((self.height_diff >= p.height_diff_limit - 1e-9 and step > 0.0)
                    or (self.height_diff <= -p.height_diff_limit + 1e-9 and step < 0.0))
        if not at_limit:
            self.height_diff = float(np.clip(self.height_diff + step,
                                            -p.height_diff_limit, p.height_diff_limit))
        self.height_reference_delta_m = 0.0
        c = u_leg
        if abs(kc) > 1e-6:
            c = u_leg - (kx / kc) * self.height_diff
        return float(c), self.height_diff

    def __call__(self, pitch: float, pitch_rate: float, body_vx: float,
                 station_error: float, yaw_rate: float, dt: float,
                 leg_length: float | None = None,
                 leg_length_rate: float = 0.0) -> tuple[np.ndarray, dict]:
        p = self.p
        self.integral = float(np.clip(
            self.integral + station_error * dt, -p.integral_limit, p.integral_limit
        ))
        # ---- 腿：低频姿态 + 位置（重积分，积分单独限幅防回弹余量被吃掉）----
        # 腿长变化阶段提高低频位置保持带宽。只按实际腿长速度门控，
        # 且仅在腿长目标偏离标称值时生效，避免普通站立时的编码器噪声
        # 改变原有平衡器增益。
        height_transition = 0.0
        if (p.height_diff_limit > 0.0
                and abs(self.leg_length_ref - p.leg_length_ref_default)
                > p.height_deadband_m):
            height_transition = float(np.clip(
                abs(float(leg_length_rate))
                / max(p.height_transition_rate_threshold, 1e-6),
                0.0, 1.0))
        kl_v = p.kl_v * (
            1.0 + (p.height_transition_velocity_scale - 1.0)
            * height_transition
        )
        kl_x = p.kl_x * (
            1.0 + (p.height_transition_position_scale - 1.0)
            * height_transition
        )
        u_leg = (-p.kl_p * pitch - p.kl_d * pitch_rate
                 + kl_v * body_vx + kl_x * station_error
                 + float(np.clip(self.integral, -p.integral_leg_limit,
                                 p.integral_leg_limit)) * p.kl_i)
        # ---- 高度/腿长环（腿负责机体高度）----
        # 目标腿长变化 = 前馈(L_ref − L_nom) + PI 修正；再与 u_leg 一起解耦到
        # (共模, 差模)。**必须解耦**：差模对支撑点前后的影响是它对腿长影响的
        # 1.9 倍，直接用会把机器人推倒（第一版实测）。
        if p.height_diff_limit > 0.0:
            measured = float(leg_length) if leg_length is not None else p.nominal_leg_length
            leg_error = measured - self.leg_length_ref     # >0 表示比目标长
        else:
            leg_error = 0.0
        u_leg = float(np.clip(u_leg, -p.leg_offset_limit, p.leg_offset_limit))
        u_common, u_diff = self._step_leg_modes(
            leg_error, u_leg, dt, leg_length_rate=leg_length_rate)
        u_common += p.common_trim
        u_diff += p.diff_trim
        # ---- 轮：高频 pitch 修正 + 位置/速度 ----
        current = (-p.kw_p * pitch - p.kw_d * pitch_rate
                   + p.kw_v * body_vx + p.kw_x * station_error
                   + float(np.clip(self.integral, -p.integral_wheel_limit,
                                   p.integral_wheel_limit)) * p.kw_i)
        current = float(np.clip(current, -p.current_limit, p.current_limit))
        diff = float(np.clip(-p.ky_p * yaw_rate, -p.diff_current_limit,
                             p.diff_current_limit))
        # ---- 输出一阶低通：压掉 125 Hz 上的 bang-bang 翻转 ----
        alpha = float(np.clip(p.filter_alpha, 0.0, 1.0))
        raw = np.array([current, diff, u_common, u_diff], dtype=np.float64)
        height_alpha = float(np.clip(p.height_filter_alpha, 0.0, 1.0))
        if alpha >= 1.0 or self._filtered is None:
            self._filtered = raw
        else:
            self._filtered[:2] = self._filtered[:2] + alpha * (
                raw[:2] - self._filtered[:2]
            )
            self._filtered[2:] = self._filtered[2:] + height_alpha * (
                raw[2:] - self._filtered[2:]
            )
        current, diff = float(self._filtered[0]), float(self._filtered[1])
        u_common, u_diff = float(self._filtered[2]), float(self._filtered[3])
        self.last = {"current": current, "leg_offset": u_common, "diff_current": diff,
                     "height_diff": u_diff,
                     "leg_length_ref": self.leg_length_ref}
        return self.to_action(current, u_common, diff, u_diff), dict(self.last)

    def to_action(self, current: float, leg_offset: float, diff: float,
                  height_diff: float = 0.0) -> np.ndarray:
        """归一化动作 → 6 维动作。

        腿通道的**语义**（与 `ActuatorBank` 的差模解码配合）::

            a[2], a[3] = 共模偏置 / leg_scale,  差模偏置 / leg_scale
            a[4], a[5] = 共模偏置 / leg_scale,  差模偏置 / leg_scale

        共模改腿的俯仰角（不影响腿长），差模改腿长。
        """
        a = np.zeros(6, dtype=np.float64)
        a[0] = current / max(self.wheel_scale, 1e-9)
        a[1] = diff / max(self.diff_scale, 1e-9)
        a[2] = leg_offset / max(self.leg_scale, 1e-9)
        a[3] = height_diff / max(self.leg_scale, 1e-9)
        a[4] = leg_offset / max(self.leg_scale, 1e-9)
        a[5] = height_diff / max(self.leg_scale, 1e-9)
        return np.clip(a, -1.0, 1.0)
