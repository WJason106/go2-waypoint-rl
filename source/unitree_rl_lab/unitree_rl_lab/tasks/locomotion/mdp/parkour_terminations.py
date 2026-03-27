from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Parkour-specific terminations
# =============================================================================
# route_completed:
#   当最后一个 waypoint 被首次命中后，当前 episode 立即 done。
#
# 为什么必须加这个终止项？
# - 你的 sparse reward `waypoint_reached` 本质上是个“到点一次性 bonus”；
# - 如果 episode 在终点后仍继续运行，而 command 内部又没有把终点状态锁住，
#   策略就可能学成“冲到最后一个点后原地刷奖励”；
# - 即使 command 端已经修好“终点不再重复领奖”，到点即 done 仍然是最干净的任务定义，
#   可以把 episode 边界与任务完成边界对齐。
# =============================================================================


def route_completed(
    env: "ManagerBasedRLEnv",
    command_name: str = "waypoint",
) -> torch.Tensor:
    """当整条 waypoint 路径完成时返回 True，shape = (num_envs,)。"""
    cmd_term = env.command_manager.get_term(command_name)

    if not hasattr(cmd_term, "path_completed"):
        raise AttributeError(
            f"Command term '{command_name}' 缺少 path_completed 属性，"
            "请先使用修复后的 waypoint_command.py。"
        )

    return cmd_term.path_completed.to(dtype=torch.bool)
