from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Waypoint-aware terrain curriculum
# =============================================================================
# 这份 curriculum 的核心修复有两点：
#
# 1) 不再从 reward_manager._episode_sums 反推完成率
#    ------------------------------------------------
#    Isaac Lab 的 RewardManager 在累计 episode reward 时，会把每一项 reward 都乘上：
#        value * weight * dt
#    然后才写进 _episode_sums。
#
#    所以如果再拿 _episode_sums["waypoint_reached"] 去反推“完成了多少个 waypoint”，
#    结果一定会被 reward weight 和环境 step_dt 污染。
#
#    对你当前环境：
#      sim.dt = 0.005
#      decimation = 4
#      step_dt = 0.02
#      waypoint_reached.weight = 5.0
#      bonus = 1.0
#      num_waypoints = 3
#
#    那么 3 个 waypoint 全部到达时，episode sum 实际只有：
#      3 * 5.0 * 1.0 * 0.02 = 0.3
#    而不是 15.0。
#
#    因此课程应该直接读取 command term 内维护的“真实命中数”：
#      waypoint_reached_count / num_waypoints
#
# 2) 不再直接改 terrain_levels，而是调用 update_env_origins()
#    ----------------------------------------------------------
#    TerrainImporter.update_env_origins() 会同时维护：
#      - terrain_levels
#      - env_origins
#
#    如果只改 terrain_levels 而不改 env_origins，日志上看起来 level 变了，
#    但机器人 reset 时的真实出生地可能根本没变，课程学习等于没有生效。
# =============================================================================


def _to_env_ids_tensor(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | slice,
) -> torch.Tensor:
    """把 env_ids 统一转换成 LongTensor，方便后续索引。"""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def terrain_levels_waypoint(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "waypoint",
    promote_ratio: float = 0.60,
    demote_ratio: float = 0.20,
    level_step: int = 1,
):
    """根据 waypoint 完成率更新 terrain level。

    参数
    ----
    env:
        ManagerBasedRLEnv 实例。
    env_ids:
        这次即将 reset 的环境 ID。
        Isaac Lab 会在 _reset_idx(env_ids) 里先调用 curriculum_manager.compute(env_ids)，
        然后才去 reset 各个 manager，所以这里拿到的 command term 状态仍然是“旧 episode”的。
    command_name:
        WaypointCommand 在 CommandsCfg 中的名字。
    promote_ratio:
        completion_ratio >= promote_ratio 时，环境升一级。
    demote_ratio:
        completion_ratio < demote_ratio 时，环境降一级。
    level_step:
        每次升 / 降的等级步长。
        官方 update_env_origins() 每次只会 +/-1，所以这里通过循环调用来兼容更大的步长。

    返回
    ----
    dict[str, float]
        返回给 CurriculumManager 记录到日志里。
    """
    env_ids = _to_env_ids_tensor(env, env_ids)
    if env_ids.numel() == 0:
        return {
            "mean_completion_ratio": 0.0,
            "promoted_fraction": 0.0,
            "demoted_fraction": 0.0,
            "mean_terrain_level": 0.0,
        }

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        # 没有 terrain importer，就没法做 terrain curriculum。
        return {
            "mean_completion_ratio": 0.0,
            "promoted_fraction": 0.0,
            "demoted_fraction": 0.0,
            "mean_terrain_level": 0.0,
        }

    if not hasattr(terrain, "update_env_origins"):
        raise AttributeError(
            "env.scene.terrain 缺少 update_env_origins()。"
            "当前 terrain 对象不支持基于 terrain origin 的 curriculum。"
        )

    # 如果 terrain_origins 为空，说明当前不是“按子地形 origin 采样环境”的模式。
    # 这时 update_env_origins() 也不会真正切换出生地。
    if getattr(terrain, "terrain_origins", None) is None:
        mean_level = 0.0
        if hasattr(terrain, "terrain_levels"):
            mean_level = torch.mean(terrain.terrain_levels[env_ids].float()).item()
        return {
            "mean_completion_ratio": 0.0,
            "promoted_fraction": 0.0,
            "demoted_fraction": 0.0,
            "mean_terrain_level": mean_level,
        }

    cmd_term = env.command_manager.get_term(command_name)

    if not hasattr(cmd_term, "waypoint_reached_count"):
        raise AttributeError(
            f"Command term '{command_name}' 缺少 waypoint_reached_count 属性，"
            "请先使用修复后的 waypoint_command.py。"
        )

    # num_waypoints 优先从 property 读取；如果 property 不存在，再回退到 cfg。
    if hasattr(cmd_term, "num_waypoints"):
        total_waypoints = float(max(int(cmd_term.num_waypoints), 1))
    else:
        total_waypoints = float(max(int(cmd_term.cfg.num_waypoints), 1))

    reached_count = cmd_term.waypoint_reached_count[env_ids].float()
    completion_ratio = reached_count / total_waypoints

    # 每个环境单独做升 / 降级，而不是先对 env_ids 求均值再“一刀切”。
    move_up = completion_ratio >= promote_ratio
    move_down = completion_ratio < demote_ratio

    # 官方 API 每次只会对 terrain_levels 做 +/-1，
    # 所以 level_step > 1 时通过重复调用实现。
    for _ in range(max(int(level_step), 1)):
        terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)

    mean_level = 0.0
    if hasattr(terrain, "terrain_levels"):
        mean_level = torch.mean(terrain.terrain_levels[env_ids].float()).item()

    return {
        "mean_completion_ratio": torch.mean(completion_ratio).item(),
        "promoted_fraction": torch.mean(move_up.float()).item(),
        "demoted_fraction": torch.mean(move_down.float()).item(),
        "mean_terrain_level": mean_level,
    }
