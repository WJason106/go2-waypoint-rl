# 🐕 Unitree Go2 跑酷强化学习框架 (Parkour RL Framework)

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-4.x-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个高度定制化的强化学习（RL）框架，专为 **Unitree Go2** 机器狗设计，用于实现跑酷和复杂地形下的运动控制。本项目基于 NVIDIA Isaac Lab 和 `unitree_rl_lab` 深度二次开发。

![演示]
![alt text](<picture.png>)

## ✨ 核心特性

本项目对标准的四足机器人 MDP（马尔可夫决策过程）进行了深度重构，旨在支持高动态、高速度的敏捷跑酷任务：
⚡ 极速的训练收敛与高样本效率 (Accelerated Convergence & High Sample Efficiency): 结合严格的早期截断机制（Early Terminations，如倾角过大的 bad_orientation 与底盘触地的 base_contact）以及路径走完即刻终止的 route_completed 逻辑，极大减少了无效状态空间下的冗余数据采样。同时，通过显式除以环境步长 (step_dt) 以消除数值污染，重构了极其纯粹的稀疏奖励梯度（如 waypoint_reached），彻底避免了策略陷入“原地刷分”的局部最优解。整套 MDP 架构使得 PPO 算法的整体训练收敛速度和样本利用效率显著超越常规基线框架。
* 🎯 基于航点的高速导航跟踪 (Waypoint-Based High-Speed Navigation): 抛弃了传统的速度跟踪方案，采用 WaypointCommand 状态机以取代传统的全局速度指令。配合严密的最大速度截断机制。该设计不仅极大提升了机器人趋近航点的线速度，还保证了高动态运动下的航向精准度。
* 🧗 航点驱动的动态地形课程 (Waypoint-Driven Terrain Curriculum): 构建了涵盖平地、随机起伏、斜坡、离散障碍箱 (Hurdles) 与台阶 (Platforms) 的多层级地形生成器。在课程学习 (Curriculum Learning) 的升降级策略上，彻底剥离了易受局部奖励污染的回合累计得分 (Episode Reward Sum)，转而采用真实航点到达率作为评估指标，确保了环境难度递增与核心导航任务的绝对对齐。
* ⚖️ 专精跑酷的强化学习奖励约束 (Parkour-Specific Reward Formulation): * 高标准姿态与稳定性约束: 对底盘触地 (base_contact) 采取严格的惩罚机制，并在机体倾角（俯仰/横滚）超过安全阈值（约 43°）时立即触发回合终止 (bad_orientation)，从根本上杜绝了策略利用物理引擎惯性产生的作弊滑行行为。

## 📁 核心目录结构

```text
unitree-go2-parkour-rl/
├── scripts/                            # 训练与测试启动脚本
│   ├── train.py
│   └── play.py
└── unitree_rl_lab/
    └── tasks/
        └── locomotion/                 # 核心环境
            ├── parkour_env_cfg.py      # 环境主配置 (Env & Play)
            ├── scene_cfg.py            # 传感器、机器人与地形场景设置
            ├── terrain_cfg.py          # 各子地形生成比例与难度区间设计
            ├── obs_cfg.py              # 策略网络 (Policy) 与价值网络 (Critic) 观测配置
            └── mdp/                    # 强化学习
                ├── parkour_rewards.py  # 自定义奖励函数库
                ├── curriculums_parkour.py # 航点感知的课程学习逻辑
                ├── parkour_terminations.py# 专属终止条件
                └── mdp_components_cfg.py  # MDP 动作、命令等组件映射
