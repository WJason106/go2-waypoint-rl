"""
parkour_reward.py
=================
专为 Parkour 四足机器人场景设计的奖励函数库。
基于 isaaclab ManagerBasedRLEnv API，移植并扩展自 legged_robot.py 的 parkour rewards 区域。

模块结构
--------
1. Navigation         —— 目标追踪、Waypoint 进度与偏航对齐
2. Body Stability     —— 机体稳定性惩罚（z 轴速度、横滚/俯仰角速度、姿态）
3. Joint & Actuator   —— 关节加速度、力矩、动作平滑性惩罚
4. Feet & Contact     —— 脚部离地高度、步态、摔绊、气时方差
5. Terrain-Aware      —— 地形边缘惩罚（需要 env 挂载 terrain 数据）
6. Collision & Safety —— 碰撞惩罚

用法示例（reward_cfg.py 片段）
------------------------------
    import parkour_reward as R

    RewardsCfg:
        tracking_goal_vel   = RewTerm(func=R.tracking_goal_vel,   weight=+2.0, params={"command_name": "waypoint"})
        tracking_yaw        = RewTerm(func=R.tracking_yaw,        weight=+0.5, params={"command_name": "waypoint"})
        waypoint_progress   = RewTerm(func=R.waypoint_progress,   weight=+1.0, params={"command_name": "waypoint"})
        waypoint_reached    = RewTerm(func=R.waypoint_reached,    weight=+5.0, params={"command_name": "waypoint", "bonus": 1.0})
        lin_vel_z           = RewTerm(func=R.lin_vel_z,           weight=-2.0)
        ang_vel_xy          = RewTerm(func=R.ang_vel_xy,          weight=-0.05)
        orientation         = RewTerm(func=R.orientation,         weight=-1.0)
        dof_acc             = RewTerm(func=R.dof_acc,             weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        torques             = RewTerm(func=R.torques,             weight=-1e-5,  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        delta_torques       = RewTerm(func=R.delta_torques,       weight=-1e-7,  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        action_rate         = RewTerm(func=R.action_rate,         weight=-0.01)
        hip_pos             = RewTerm(func=R.hip_pos,             weight=-1.0,   params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])})
        dof_error           = RewTerm(func=R.dof_error,           weight=-0.04,  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        feet_stumble        = RewTerm(func=R.feet_stumble,        weight=-2.0,   params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot"])})
        feet_edge           = RewTerm(func=R.feet_edge,           weight=-1.0,   params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot"]), "terrain_level_threshold": 3})
        foot_clearance      = RewTerm(func=R.foot_clearance,      weight=+0.2,   params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]), "target_height": 0.1, "std": 0.05, "tanh_mult": 2.0})
        feet_gait           = RewTerm(func=R.feet_gait,           weight=+0.1,   params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot"]), "period": 0.5, "offset": [0.0, 0.5, 0.5, 0.0], "command_name": "waypoint"})
        collision           = RewTerm(func=R.collision,           weight=-1.0,   params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["base", ".*_thigh"])})
        energy              = RewTerm(func=R.energy,              weight=-1e-4,  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# 1. Navigation — 目标追踪 / Waypoint 进度 / 偏航对齐
# =============================================================================

def tracking_goal_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励机器人沿 waypoint 方向的速度追踪，奖励值归一化到 [0, 1]。

    移植自 legged_robot._reward_tracking_goal_vel，改为从 command_term 读取相对位置。

    计算公式：

    .. math::

        r = \\frac{\\min(v_{\\text{proj}},\\, v_{\\text{cmd}})}{v_{\\text{cmd}} + \\epsilon}

    其中 :math:`v_{\\text{proj}}` 为世界系速度在 waypoint 方向上的投影（clamp ≥ 0），
    :math:`v_{\\text{cmd}}` 为指令速度幅值（command 第 0 维绝对值）。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    # waypoint 在机体系下的相对方向 (N, 2)
    d_b = cmd_term.waypoint_rel_xy
    d_norm = torch.norm(d_b, dim=-1, keepdim=True).clamp_min(1e-5)
    d_hat = d_b / d_norm  # 单位方向向量（机体系）

    # 机体系 xy 速度
    if hasattr(asset.data, "root_lin_vel_b"):
        v_xy = asset.data.root_lin_vel_b[:, :2]
    else:
        v_w = asset.data.root_lin_vel_w
        v_b = quat_apply_inverse(asset.data.root_quat_w, v_w)
        v_xy = v_b[:, :2]

    v_proj = torch.sum(d_hat * v_xy, dim=-1).clamp_min(0.0)

    # 指令速度
    cmd_vec = env.command_manager.get_command(command_name)
    v_cmd = cmd_vec[:, 0].abs().clamp_min(1e-5)

    return torch.minimum(v_proj, v_cmd) / v_cmd


def tracking_yaw(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
) -> torch.Tensor:
    """奖励机头偏航朝向与 waypoint 方向的对齐程度。

    移植自 legged_robot._reward_tracking_yaw，偏航误差由机体系 waypoint 方向的
    atan2 直接给出，无需额外坐标变换。

    计算公式：

    .. math::

        r = e^{-|\\Delta\\psi|}

    其中 :math:`\\Delta\\psi = \\text{atan2}(d_y^b,\\, d_x^b)` 为机体系下 waypoint 的偏航误差。
    """
    cmd_term = env.command_manager.get_term(command_name)
    d_b = cmd_term.waypoint_rel_xy  # (N, 2)
    yaw_error = torch.atan2(d_b[:, 1], d_b[:, 0])
    return torch.exp(-torch.abs(yaw_error))


def waypoint_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    min_clip: float = -0.5,
    max_clip: float = 0.5,
) -> torch.Tensor:
    """奖励沿路径的前进进度，除以 step_dt 抵消 Manager 的自动乘法"""
    cmd_term = env.command_manager.get_term(command_name)
    # 【核心修复】：除以 step_dt 转回速度量级
    progress_vel = cmd_term.waypoint_progress / env.step_dt
    return torch.clamp(progress_vel, min=min_clip, max=max_clip)

def waypoint_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    bonus: float = 1.0,
) -> torch.Tensor:
    """在机器人到达当前 waypoint 的那一步给予稀疏奖励加成"""
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "reached_this_step"):
        return torch.zeros(env.num_envs, device=env.device)
    # 【核心修复】：除以 step_dt，使其成为真正的大额单次奖励
    return cmd_term.reached_this_step.float() * bonus / env.step_dt

def velocity_heading_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    only_positive: bool = True,
    normalize_speed: bool = False,
    eps: float = 1e-6,
    max_vel: float = 1.0,  # 【核心修复】：加上最高速度限制参数
) -> torch.Tensor:
    """奖励机体速度方向与 waypoint 方向的一致性（内积）。"""
    asset: RigidObject = env.scene["robot"]
    cmd_term = env.command_manager.get_term(command_name)

    d_b = cmd_term.waypoint_rel_xy  
    d_hat = d_b / torch.norm(d_b, dim=-1, keepdim=True).clamp_min(eps)

    if hasattr(asset.data, "root_lin_vel_b"):
        v_xy = asset.data.root_lin_vel_b[:, :2]
    else:
        v_w = asset.data.root_lin_vel_w
        v_b = quat_apply_inverse(asset.data.root_quat_w, v_w)
        v_xy = v_b[:, :2]

    proj = torch.sum(v_xy * d_hat, dim=-1)

    # 【核心修复】：限制最大奖励速度，防止为了高分而疯狂飞扑
    proj = torch.clamp(proj, max=max_vel)

    if normalize_speed:
        proj = proj / torch.norm(v_xy, dim=-1).clamp_min(eps)
    if only_positive:
        proj = proj.clamp_min(0.0)

    return proj


# =============================================================================
# 2. Body Stability — 机体稳定性惩罚
# =============================================================================

def lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    jump_class_id: int | None = None,
    jump_scale: float = 2.0,
) -> torch.Tensor:
    """惩罚机体 z 轴线速度（L2 平方）。

    移植自 legged_robot._reward_lin_vel_z。
    原实现对非跳跃地形（env_class != 17）将权重缩减为 0.5；
    此处通过可选参数 ``jump_class_id`` 实现同等逻辑，默认不区分地形类别。

    若需区分地形类别，请在 env 上挂载 ``env_class`` 张量，并传入对应的 ``jump_class_id``。

    计算公式：

    .. math::

        r = {\\dot{z}}^2
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])

    if jump_class_id is not None and hasattr(env, "env_class"):
        # 非跳跃地形缩减惩罚，与原实现保持一致
        rew[env.env_class != jump_class_id] *= (1.0 / jump_scale)

    return rew


def ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机体绕 x/y 轴的角速度（俯仰 + 滚转），L2 平方之和。

    移植自 legged_robot._reward_ang_vel_xy。

    .. math::

        r = \\omega_x^2 + \\omega_y^2
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    jump_class_id: int | None = None,
) -> torch.Tensor:
    """惩罚机体非水平姿态，对投影重力向量的 xy 分量取 L2 平方之和。

    移植自 legged_robot._reward_orientation。
    原实现仅对跳跃地形（env_class == 17）施加该惩罚；
    传入 ``jump_class_id`` 可还原该行为，非跳跃地形惩罚清零。

    .. math::

        r = g_x^{b\\,2} + g_y^{b\\,2}
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    if jump_class_id is not None and hasattr(env, "env_class"):
        # 仅对跳跃地形施加惩罚（原始行为）
        rew[env.env_class != jump_class_id] = 0.0

    return rew


def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励机体保持在目标高度附近，使用 L2 平方误差（越小越好，作为惩罚使用时 weight < 0）。

    .. math::

        r = (z - z_{\\text{target}})^2
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


# =============================================================================
# 3. Joint & Actuator — 关节 / 执行器平滑性惩罚
# =============================================================================

def dof_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚关节加速度，L2 平方之和。

    移植自 legged_robot._reward_dof_acc（原实现手动差分 Δvel/dt）。
    isaaclab 的 Articulation 提供 ``joint_acc``，精度更高且无需额外缓存。

    .. math::

        r = \\sum_i \\ddot{q}_i^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚电机输出力矩，L2 平方之和。

    移植自 legged_robot._reward_torques。
    与 ``energy``（= |τ||q̇|）的区别：本函数仅关注力矩幅值，不考虑运动速度。

    .. math::

        r = \\sum_i \\tau_i^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def delta_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚相邻步之间的力矩变化量，L2 平方之和。

    移植自 legged_robot._reward_delta_torques。
    上一步力矩缓存在 ``env._pkr_prev_torque_{asset_name}`` 属性中，首次调用自动初始化为 0。

    .. math::

        r = \\sum_i (\\tau_i^t - \\tau_i^{t-1})^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cur = asset.data.applied_torque[:, asset_cfg.joint_ids]
    key = f"_pkr_prev_torque_{asset_cfg.name}"
    if not hasattr(env, key):
        setattr(env, key, torch.zeros_like(cur))
    prev: torch.Tensor = getattr(env, key)
    just_reset = env.episode_length_buf == 1          # (N,) bool
    prev[just_reset] = cur[just_reset].detach()
    rew = torch.sum(torch.square(cur - prev), dim=1)
    setattr(env, key, cur.detach().clone())
    return rew


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """惩罚相邻步之间的动作变化量，L2 范数。

    移植自 legged_robot._reward_action_rate。
    动作缓存在 ``env._pkr_prev_action`` 属性中，首次调用自动初始化为 0。

    .. math::

        r = \\|a^t - a^{t-1}\\|_2
    """
    cur = env.action_manager.action
    key = "_pkr_prev_action"
    if not hasattr(env, key):
        setattr(env, key, torch.zeros_like(cur))
    prev: torch.Tensor = getattr(env, key)
    just_reset = env.episode_length_buf == 1          # (N,) bool
    prev[just_reset] = cur[just_reset].detach()
    rew = torch.norm(cur - prev, dim=1)
    setattr(env, key, cur.detach().clone())
    return rew


def dof_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚所有关节与默认位置的偏差，L2 平方之和。

    移植自 legged_robot._reward_dof_error。
    与 ``hip_pos`` 的区别：本函数覆盖所有指定关节，而非仅髋关节。

    .. math::

        r = \\sum_i (q_i - q_i^{\\text{default}})^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=1)


def hip_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚髋关节与默认位置的偏差，L2 平方之和。

    移植自 legged_robot._reward_hip_pos。
    使用时请在 SceneEntityCfg 中通过 ``joint_names`` 指定髋关节，例如：

    .. code-block:: python

        asset_cfg=SceneEntityCfg("robot", joint_names=[".*_hip_joint"])

    .. math::

        r = \\sum_{i \\in \\text{hip}} (q_i - q_i^{\\text{default}})^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=1)


def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚关节消耗的机械功率（能量代价），= |τ| · |q̇|。

    与 ``torques`` 的区别：同时考虑力矩和速度，更接近真实功耗。

    .. math::

        r = \\sum_i |\\tau_i| \\cdot |\\dot{q}_i|
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


# =============================================================================
# 4. Feet & Contact — 脚部奖励 / 步态 / 接触时序
# =============================================================================

def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚脚部撞击竖直面（横向接触力 > 4 × 竖向接触力）。

    移植自 legged_robot._reward_feet_stumble。

    .. math::

        r = \\mathbf{1}\\left[\\exists\\, i:\\; F_{xy,i} > 4\\,|F_{z,i}|\\right]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, F, 3)
    forces_z = torch.abs(forces[:, :, 2])
    forces_xy = torch.linalg.norm(forces[:, :, :2], dim=2)
    return torch.any(forces_xy > 4.0 * forces_z, dim=1).float()


def feet_edge(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_level_threshold: int = 3,
) -> torch.Tensor:
    """惩罚脚部踩在地形边缘（Parkour 地形特有）。

    移植自 legged_robot._reward_feet_edge。
    需要 env 上挂载以下属性（由地形初始化时设置）：
        - ``env.terrain_edge_mask``  : BoolTensor (rows, cols)，边缘格点为 True
        - ``env.terrain_level``      : LongTensor (N,)，当前地形难度等级
        - ``env.terrain_hor_scale``  : float，水平分辨率（m/cell）
        - ``env.terrain_border_size``: float，地形边框尺寸（m）
    若上述属性不存在，则跳过该奖励并返回全零（不报错）。

    仅在地形难度高于 ``terrain_level_threshold`` 的环境中生效，
    低难度地形不惩罚（给予足够探索余地）。

    .. math::

        r = \\sum_{i \\in \\text{feet}} \\mathbf{1}[\\text{foot}_i \\text{ on edge} \\wedge \\text{contact}_i]
    """
    # 检查依赖属性
    required = ("terrain_edge_mask", "terrain_level", "terrain_hor_scale", "terrain_border_size")
    if not all(hasattr(env, attr) for attr in required):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 接触状态（使用历史滤波防止单帧抖动）
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0  # (N, F)

    # 脚部 xy 世界坐标 → 地图格点索引
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (N, F, 2)
    edge_mask: torch.Tensor = env.terrain_edge_mask
    hor_scale: float = env.terrain_hor_scale
    border: float = env.terrain_border_size

    grid_idx = ((feet_pos_w + border) / hor_scale).round().long()  # (N, F, 2)
    grid_idx[..., 0] = grid_idx[..., 0].clamp(0, edge_mask.shape[0] - 1)
    grid_idx[..., 1] = grid_idx[..., 1].clamp(0, edge_mask.shape[1] - 1)

    # 查表：每只脚是否在边缘格点上
    feet_on_edge = edge_mask[grid_idx[..., 0], grid_idx[..., 1]]  # (N, F)

    # 只有接触地面时才计入惩罚
    feet_at_edge = is_contact & feet_on_edge  # (N, F)

    # 高难度地形才施加惩罚
    high_level = env.terrain_level > terrain_level_threshold  # (N,)
    return high_level.float() * torch.sum(feet_at_edge.float(), dim=-1)


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """奖励摆动脚达到目标离地高度，使用 Gaussian 形状核。

    .. math::

        r = \\exp\\!\\left(-\\frac{\\sum_i (z_i - z_{\\text{target}})^2 \\cdot
            \\tanh(k \\|\\dot{p}_{xy,i}\\|)}{\\sigma}\\right)

    参数
    ----
    target_height : 目标离地高度（m）
    std           : Gaussian 宽度（m²，控制奖励衰减速度）
    tanh_mult     : 速度调制系数 k（脚快速运动时才给奖励）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    z_err = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    vel_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    vel_tanh = torch.tanh(tanh_mult * vel_xy)
    reward = torch.sum(z_err * vel_tanh, dim=1)
    return torch.exp(-reward / std)


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """奖励摆动脚在机体坐标系下达到目标离地高度。

    与 ``foot_clearance`` 的区别：高度基准为机体坐标系而非世界坐标系，
    适合坡面、台阶等地形，使奖励与机体姿态解耦。

    仅在有运动指令时生效（指令范数 > 0.1）；
    重力向量 z 分量越接近 -1（机体越水平），奖励增益越大（clamp 到 [0, 0.7]）。
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 脚部位置 / 速度：世界系 → 机体系
    foot_pos_rel = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    foot_vel_rel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w.unsqueeze(1)

    foot_pos_b = torch.zeros_like(foot_pos_rel)
    foot_vel_b = torch.zeros_like(foot_vel_rel)
    for i in range(len(asset_cfg.body_ids)):
        foot_pos_b[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, foot_pos_rel[:, i, :])
        foot_vel_b[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, foot_vel_rel[:, i, :])

    z_err = torch.square(foot_pos_b[:, :, 2] - target_height).view(env.num_envs, -1)
    vel_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_b[:, :, :2], dim=2))
    rew = torch.sum(z_err * vel_tanh, dim=1)

    # 运动指令门控
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    rew *= (cmd_norm > 0.1).float()

    # 姿态增益：机体越水平，奖励越大
    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    rew *= upright
    return rew


def feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """奖励腿部跟随预设相位步态（stance/swing 交替）。

    通过基于时钟的相位信号定义期望步态，当实际接触状态与期望一致时给奖励。

    参数
    ----
    period    : 步态周期（s）
    offset    : 每条腿的相位偏移列表，长度需与 sensor_cfg.body_ids 一致。
                四足 trot 步态示例：[0.0, 0.5, 0.5, 0.0]（FR/RR 同相，FL/RL 反相）。
    threshold : 相位 < threshold 为 stance，≥ threshold 为 swing。
    command_name : 若指定，则仅在有运动指令（范数 > 0.1）时生效。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0  # (N, F)

    # 全局相位 → 各腿相位
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)  # (N, 1)
    leg_phases = torch.cat(
        [((global_phase + off) % 1.0) for off in offset], dim=-1
    )  # (N, F)

    rew = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phases[:, i] < threshold
        # XOR 为 True 表示期望与实际不一致，取反后为 True 表示一致
        rew += (~(is_stance ^ is_contact[:, i])).float()

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        rew *= (cmd_norm > 0.1).float()

    return rew


def air_time_variance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_clip: float = 0.5,
) -> torch.Tensor:
    """惩罚各脚腾空/接触时间的方差，鼓励四足对称步态。

    需要 ContactSensor 配置 ``track_air_time=True``。

    .. math::

        r = \\text{Var}(\\text{clip}(t_{\\text{air}}, 0, T)) +
            \\text{Var}(\\text{clip}(t_{\\text{contact}}, 0, T))
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not sensor.cfg.track_air_time:
        raise RuntimeError(
            "air_time_variance 需要 ContactSensor 开启 track_air_time=True，"
            "请在传感器配置中设置该选项。"
        )
    air = torch.clip(sensor.data.last_air_time[:, sensor_cfg.body_ids], max=max_clip)
    contact = torch.clip(sensor.data.last_contact_time[:, sensor_cfg.body_ids], max=max_clip)
    return torch.var(air, dim=1) + torch.var(contact, dim=1)


def feet_contact_when_still(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "waypoint",
) -> torch.Tensor:
    """奖励在指令为零时所有脚保持接触地面（站立静止奖励）。

    .. math::

        r = \\sum_i \\mathbf{1}[\\text{contact}_i] \\cdot \\mathbf{1}[\\|c\\| < 0.1]
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return torch.sum(is_contact.float(), dim=-1) * (cmd_norm < 0.1).float()


def feet_too_near(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2,
) -> torch.Tensor:
    """惩罚两脚过近（仅支持双足，取前两个 body_ids 计算距离）。

    .. math::

        r = \\max(0,\\; d_{\\text{thresh}} - \\|p_0 - p_1\\|)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, F, 3)
    dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - dist).clamp(min=0.0)


# =============================================================================
# 5. Collision & Safety — 碰撞与安全惩罚
# =============================================================================

def collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """惩罚指定身体部位发生非期望碰撞（接触力范数超过阈值时计入）。

    移植自 legged_robot._reward_collision。
    通过 ``sensor_cfg.body_names`` 指定受惩罚部位，例如：

    .. code-block:: python

        sensor_cfg=SceneEntityCfg("contact_sensor", body_names=["base", ".*_thigh"])

    .. math::

        r = \\sum_i \\mathbf{1}[\\|F_i\\| > \\text{threshold}]
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, B, 3)
    return torch.sum((torch.norm(forces, dim=-1) > threshold).float(), dim=1)

def waypoint_progress_reward(env: ManagerBasedRLEnv, command_name: str = "waypoint") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return torch.clamp(cmd.waypoint_progress, min=-0.5, max=0.5)


def waypoint_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    bonus: float = 1.0,
) -> torch.Tensor:
    """
    无状态实现：直接读取 command 在当前 step 的事件标志。
    """
    cmd = env.command_manager.get_term(command_name)
    if not hasattr(cmd, "reached_this_step"):
        # 兼容：若旧 command 未实现该字段，则不给奖励，避免错误状态驻留
        return torch.zeros(env.num_envs, device=env.device)
    return cmd.reached_this_step.float() * bonus


def waypoint_velocity_inner_product(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    only_positive: bool = True,
    normalize_speed: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    坐标系一致实现：
    - 速度统一使用 body frame 的 v_b_xy
    - 目标方向使用 body frame 的 d_b（waypoint_rel_xy）
    - 再做点积
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term(command_name)

    d_b = cmd.waypoint_rel_xy  # (N,2)
    d_norm = torch.norm(d_b, dim=-1, keepdim=True).clamp_min(eps)
    d_hat = d_b / d_norm

    # 优先直接使用 body velocity
    if hasattr(robot.data, "root_lin_vel_b"):
        v_b_xy = robot.data.root_lin_vel_b[:, :2]
    else:
        # fallback: world -> body 再取xy，避免坐标系混���
        v_w = robot.data.root_lin_vel_w  # (N,3)
        v_b = quat_apply_inverse(robot.data.root_quat_w, v_w)
        v_b_xy = v_b[:, :2]

    proj_speed = torch.sum(v_b_xy * d_hat, dim=-1)

    if normalize_speed:
        speed = torch.norm(v_b_xy, dim=-1).clamp_min(eps)
        proj_speed = proj_speed / speed

    if only_positive:
        proj_speed = torch.clamp(proj_speed, min=0.0)

    return proj_speed

def waypoint_heading_inner_product(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
) -> torch.Tensor:
    """
    机头朝向追踪奖励：
    鼓励机器人的正前方（机体系 X 轴）对准 waypoint 的方向。
    """
    cmd = env.command_manager.get_term(command_name)
    
    # 获取机体系下的目标相对坐标 (N, 2)
    d_b = cmd.waypoint_rel_xy
    
    # 将方向向量归一化
    d_norm = torch.norm(d_b, dim=-1, keepdim=True).clamp_min(1e-6)
    d_hat = d_b / d_norm
    
    # 因为在机体坐标系下，机器人的正前方向量始终是 [1, 0]
    # 所以 [1, 0] 与 d_hat [dx, dy] 的点积，就是 d_hat 的 x 分量
    heading_reward = d_hat[:, 0]
    
    return heading_reward

