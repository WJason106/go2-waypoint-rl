import gymnasium as gym

# 注册你自定义的平地+崎岖预训练环境
gym.register(
    id="Unitree-Go2-Parkour",  # 你可以在启动训练脚本时使用这个 ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 指向你拆分后的主环境配置类
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",
        # 指向你拆分后的测试/Play环境配置类
        "play_env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourPlayCfg",
        # 沿用宇树官方的 PPO 算法配置文件
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
