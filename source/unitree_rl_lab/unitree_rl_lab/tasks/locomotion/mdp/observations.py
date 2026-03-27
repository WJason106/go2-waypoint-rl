from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def waypoint_rel_xy(env: ManagerBasedRLEnv, command_name: str = "waypoint") -> torch.Tensor:
    """当前 waypoint 在机体坐标系下的相对位置 (x, y)。"""
    cmd = env.command_manager.get_term(command_name)
    return cmd.waypoint_rel_xy


def waypoint_distance(env: ManagerBasedRLEnv, command_name: str = "waypoint") -> torch.Tensor:
    """当前 waypoint 距离（标量），输出 shape=(N,1)。"""
    cmd = env.command_manager.get_term(command_name)
    return cmd.waypoint_distance.unsqueeze(-1)
