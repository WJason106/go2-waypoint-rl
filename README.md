# 🐕 Unitree Go2 跑酷强化学习框架 (Parkour RL Framework)

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-4.x-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个高度定制化的强化学习（RL）框架，专为 **Unitree Go2** 机器狗设计，用于实现跑酷和复杂地形下的运动控制。本项目基于 NVIDIA Isaac Lab 和 `unitree_rl_lab` 深度二次开发。

![演示]
![alt text](<picture.png>)

## ✨ 核心特性

本项目对标准的四足机器人 MDP（马尔可夫决策过程）进行了深度改造，以支持敏捷的跑酷动作：

* 🎯 **航点导航与状态机 (Waypoint Navigation):** 定制了 `WaypointCommand` 状态机来追踪导航进度。奖励与终止条件不再依赖简单的速度追踪，而是严格绑定到真实的航点完成度（包含稳步前进的 `waypoint_progress` 以及到达航点时的稀疏暴击奖励 `waypoint_reached`）。
* 🧗 **动态地形课程学习 (Dynamic Terrain Curriculum):** 重写了地形生成器，包含平地、随机起伏、斜坡、离散障碍箱 (Hurdles) 和复杂台阶 (Platforms)。环境难度的升降级严格基于**真实的航点命中数量**，与容易被污染的回合总奖励 (Episode Reward Sum) 完全解耦。
* ⚖️ **跑酷专属奖励函数 (Parkour-Specific Rewards):** * **稳定性保障:** 对躯干触地 (Base Contact) 和危险姿态进行极其严格的惩罚（例如：俯仰/横滚角 > 43度立马判定失败，不给靠惯性滑行的机会）。
  * **精准落足:** 引入地形感知的边缘踩踏惩罚 (`feet_edge`) 和基于高斯核的腾空高度奖励 (`foot_clearance`)，确保机器狗干净利落跨越障碍。
  * **平滑与部署:** 针对相邻步力矩变化 (`delta_torques`) 和动作输出频率 (`action_rate`) 施加惩罚，减轻 Sim-to-Real 的思想包袱。

## 📁 核心目录结构

```text
unitree-go2-parkour-rl/
├── scripts/                            # 训练与测试启动脚本
│   ├── train.py
│   └── play.py
├── unitree_rl_lab/
│   └── tasks/
│       └── locomotion/                 # 核心环境
│           ├── parkour_env_cfg.py      # 环境主配置 (Env & Play)
│           ├── scene_cfg.py            # 传感器、机器人与地形场景设置
│           ├── terrain_cfg.py          # 各子地形生成比例与难度区间设计
│           ├── obs_cfg.py              # 策略网络 (Policy) 与价值网络 (Critic) 观测配置
│           └── mdp/                    # 强化学习
│               ├── parkour_rewards.py  # 自定义奖励函数库
│               ├── curriculums_parkour.py # 航点感知的课程学习逻辑
│               ├── parkour_terminations.py# 专属终止条件
│               └── mdp_components_cfg.py  # MDP 动作、命令等组件映射
