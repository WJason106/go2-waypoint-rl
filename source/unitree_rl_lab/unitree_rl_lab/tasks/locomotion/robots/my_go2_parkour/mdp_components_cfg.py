# mdp_components_cfg.py
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import curriculums_parkour as parkour_curr
from unitree_rl_lab.tasks.locomotion.mdp import parkour_rewards as R
from unitree_rl_lab.tasks.locomotion.mdp import parkour_terminations as parkour_done


# =============================================================================
# 命令配置
# =============================================================================
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 主任务：追踪一条由多个 waypoint 组成的路径。
    # 由于 WaypointCommand 现在已经修好了终点状态机，
    # 它会额外暴露：
    # - reached_this_step
    # - waypoint_reached_count
    # - path_completed
    #
    # 奖励、curriculum、termination 都会直接共享这些状态。
    waypoint = mdp.WaypointCommandCfg(
        asset_name="robot",
        resampling_time_range=(9999.0, 9999.0),  # 一个 episode 内基本不换路径
        debug_vis=False,
        num_waypoints=3,
        waypoint_radius=0.35,
        x_range=(1.5, 5.0),
        y_range=(-1.0, 1.0),
        forward_only=True,
    )


# =============================================================================
# 动作配置
# =============================================================================
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


# =============================================================================
# 奖励配置
# =============================================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP — Parkour optimized."""

    # --- 1. 导航主任务 (由于底层修了 dt 和 max_vel，现在的权重非常健康) ---
    waypoint_progress = RewTerm(
        func=R.waypoint_progress,
        weight=5.0,  # 鼓励稳步靠近
        params={"command_name": "waypoint"},
    )
    waypoint_reached = RewTerm(
        func=R.waypoint_reached,
        weight=5.0,  # 吃到航点的暴击奖励
        params={"command_name": "waypoint", "bonus": 1.0},
    )
    waypoint_heading_track = RewTerm(
        func=R.tracking_yaw,
        weight=0.5,
    )
    waypoint_vel_track = RewTerm(
        func=R.velocity_heading_alignment,
        weight=1.5,  # 重新启用，底层有限速器，不用怕它飞扑
        params={
            "command_name": "waypoint",
            "only_positive": True,
            "normalize_speed": False,
            "max_vel": 1.0, 
        },
    )

    # --- 2. 稳定性惩罚 ---
    base_linear_velocity = RewTerm(func=R.lin_vel_z, weight=-2.0)
    base_angular_velocity = RewTerm(func=R.ang_vel_xy, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=R.orientation, weight=-1.0)

    # --- 3. 关节平滑与能耗 ---
    joint_acc = RewTerm(func=R.dof_acc, weight=-2.5e-7)
    joint_torques = RewTerm(func=R.torques, weight=-1e-5)
    delta_torques = RewTerm(func=R.delta_torques, weight=-1e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    action_rate = RewTerm(func=R.action_rate, weight=-0.01)
    energy = RewTerm(func=R.energy, weight=-2e-5)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-4)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # --- 4. 关节位置约束 ---
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.1,  # 【核心调整】：从 -0.5 降到 -0.1，减轻思想包袱，让它敢迈开腿
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.3,
            "command_name": "waypoint",
        },
    )
    hip_pos = RewTerm(func=R.hip_pos, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")})

    # --- 5. 步态与碰撞 ---
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "command_name": "waypoint", "threshold": 0.5})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 【核心调整】：严抓肚皮擦地，名字严格使用合法的 "base"
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
    )
    # 【核心调整】：一旦倾角过大（约43度）立马判死，不给它靠惯性滑行的机会
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.75},
    )
    route_completed = DoneTerm(
        func=parkour_done.route_completed,
        params={"command_name": "waypoint"},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(
        func=parkour_curr.terrain_levels_waypoint,
        params={
            "command_name": "waypoint",
            "promote_ratio": 0.95,  # 【核心调整】：极其严苛，几乎必须走完所有航点才能升难度
            "demote_ratio": 0.40,   # 【核心调整】：走不到一半直接降级
            "level_step": 1,
        },
    )


# =============================================================================
# 事件配置
# =============================================================================
@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.01, 0.01), "yaw": (-0.01, 0.01)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.3, 0.3),
            "velocity_range": (-0.1, 0.1),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.1, 0.1)}},
    )


