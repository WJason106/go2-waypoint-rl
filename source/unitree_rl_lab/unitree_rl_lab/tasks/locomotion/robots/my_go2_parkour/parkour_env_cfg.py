"""
railway_env_cfg.py
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from .scene_cfg import RobotSceneCfg
from .obs_cfg import ObservationsCfg
from .mdp_components_cfg import (
    ActionsCfg, CommandsCfg, RewardsCfg, 
    TerminationsCfg, EventCfg, CurriculumCfg
)

@configclass
class ParkourEnvCfg(ManagerBasedRLEnvCfg):
    scene:        RobotSceneCfg   = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    commands:     CommandsCfg     = CommandsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()
    curriculum:   CurriculumCfg   = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class ParkourPlayCfg(ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 1. 减少环境数量，方便观察单机表现
        self.scene.num_envs = 16
        # 2. 锁定地形行列，方便对比
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 10

        # 3. 修正命令引用：由于你用 waypoint 替换了 base_velocity，这里要改名
        # 在演示模式下，通常我们会开启调试显示 (debug_vis)
        if hasattr(self.commands, "waypoint"):
            self.commands.waypoint.debug_vis = True  # 极其关键：演示时看到航点在哪里
            
            # 如果你想在演示时让航点距离更远或者路径更宽，可以在这里改
            # self.commands.waypoint.num_waypoints = 5 
            # self.commands.waypoint.x_range = (2.0, 6.0)

        # 4. (可选) 演示时关闭动作噪声，查看策略的最优表现
        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False
