# obs_cfg.py
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from unitree_rl_lab.tasks.locomotion import mdp

# ============================================================
# 观测配置
# ============================================================

@configclass
class ObservationsCfg:
    """
    观测配置

    PolicyCfg 维度说明：
      base_ang_vel       3
      projected_gravity  3
      joint_pos_rel     12
      joint_vel_rel     12
      last_action       12
      height_scanner   160
      waypoint_rel_xy    2
      waypoint_distance  1
      ──────────────────
      合计             205  （× history_length=5 → 实际输入 1025）

    CriticCfg 维度说明：
      base_lin_vel       3
      base_ang_vel       3
      projected_gravity  3
      joint_pos_rel     12
      joint_vel_rel     12
      joint_effort      12
      last_action       12
      height_scanner   160
      waypoint_rel_xy    2
      waypoint_distance  1
      ──────────────────
      合计             220  （× history_length=5 → 实际输入 1100）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # ===== Waypoint 导航观测 =====
        waypoint_rel_xy = ObsTerm(
            func=mdp.waypoint_rel_xy,
            params={"command_name": "waypoint"},
            clip=(-10.0, 10.0),
        )
        waypoint_distance = ObsTerm(
            func=mdp.waypoint_distance,
            params={"command_name": "waypoint"},
            clip=(0.0, 20.0),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (privileged)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        # ===== Waypoint 导航观测（Critic 也需要知道目标方向）=====
        waypoint_rel_xy = ObsTerm(
            func=mdp.waypoint_rel_xy,
            params={"command_name": "waypoint"},
            clip=(-10.0, 10.0),
        )
        waypoint_distance = ObsTerm(
            func=mdp.waypoint_distance,
            params={"command_name": "waypoint"},
            clip=(0.0, 20.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()

