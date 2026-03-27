from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass


# =============================================================================
# WaypointCommand
# =============================================================================
# 这个命令项负责做 3 件事：
# 1) 为每个环境采样一串 waypoint；
# 2) 把“当前 active waypoint”转换成机体系观测 (rel_x_b, rel_y_b, dist)；
# 3) 维护 waypoint 状态机，用于奖励 / curriculum / termination 共享。
#
# 这次修改重点修了两个训练稳定性的硬伤：
# - 最后一个 waypoint 原先会被反复判定为 reached，从而反复刷 sparse reward；
# - 自定义 reset() 没有同步维护 CommandTerm 的 time_left，导致每个 episode 的第一步会被
#   super().compute(dt) 再重采样一次路径。
#
# 修复后的行为：
# - reached_this_step 只在“首次命中 waypoint 的那一帧”为 True；
# - 命中最后一个 waypoint 后，path_completed 会锁存为 True；
# - 终点后不会再重复计数 / 重复发 reached bonus；
# - reset() 会先记录旧 episode 的统计信息，再调用 super().reset() 正确刷新 time_left。
# =============================================================================
class WaypointCommand(CommandTerm):
    cfg: "WaypointCommandCfg"

    def __init__(self, cfg: "WaypointCommandCfg", env):

        # 可视化句柄：只有 debug_vis=True 时才会真正创建。
        self._goal_visualizer: VisualizationMarkers | None = None
        super().__init__(cfg, env)
        self._asset = self._env.scene[cfg.asset_name]

        num_envs = self.num_envs
        num_waypoints = cfg.num_waypoints
        device = self.device

        # ---------------------------------------------------------------------
        # waypoint 几何缓存
        # ---------------------------------------------------------------------
        # _waypoints_w:
        #   每个环境一条路径，shape = (N, K, 3)，坐标系为世界系。
        # _wp_index:
        #   当前正在追踪的 active waypoint 索引。
        # ---------------------------------------------------------------------
        self._waypoints_w = torch.zeros((num_envs, num_waypoints, 3), device=device)
        self._wp_index = torch.zeros((num_envs,), dtype=torch.long, device=device)

        # ---------------------------------------------------------------------
        # episode 级任务状态
        # ---------------------------------------------------------------------
        # _reached_count:
        #   本 episode 里累计首次命中的 waypoint 数量。
        # _path_completed:
        #   本 episode 是否已经完成整条路径。
        # reached_this_step:
        #   这一帧是否首次命中了一个 waypoint（脉冲量，只持续一帧）。
        # ---------------------------------------------------------------------
        self._reached_count = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._path_completed = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        self.reached_this_step = torch.zeros((num_envs,), dtype=torch.bool, device=device)

        # ---------------------------------------------------------------------
        # progress 奖励所需的距离缓存
        # ---------------------------------------------------------------------
        # waypoint_progress = _dist_prev - _dist_curr
        # 为了避免“到达后切目标”带来的虚假大 progress，
        # 在 waypoint 切换后会把 _dist_prev 对齐到新目标的 _dist_curr。
        # ---------------------------------------------------------------------
        self._dist_prev = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self._dist_curr = torch.zeros((num_envs,), dtype=torch.float32, device=device)

        # 命令输出：机体系下当前 waypoint 的相对 x / y 和平面距离
        self._command_out = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    # -------------------------------------------------------------------------
    # Isaac Lab CommandTerm 抽象接口
    # -------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """返回当前命令张量，shape = (N, 3)，内容为 (rel_x_b, rel_y_b, dist)。"""
        return self._command_out

    def _update_metrics(self):
        """这里暂时不做逐步 metrics 累积；episode 统计在 reset() 时直接返回。"""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定环境重新采样整条 waypoint 路径。"""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        root_pos_w = self._asset.data.root_pos_w[env_ids]
        num_envs = env_ids.shape[0]
        num_waypoints = self.cfg.num_waypoints

        x_min, x_max = self.cfg.x_range
        y_min, y_max = self.cfg.y_range

        # 沿 x 方向等间距排布 waypoint，再给每个 waypoint 添加随机 y 偏移。
        # 这样路径整体是“向前推进”的，但又保留少量左右摆动，便于学习导航。
        step_x = torch.linspace(x_min, x_max, num_waypoints, device=self.device).unsqueeze(0).repeat(num_envs, 1)
        noise_y = torch.empty((num_envs, num_waypoints), device=self.device).uniform_(y_min, y_max)

        if not self.cfg.forward_only:
            # 如果允许前后随机，则对整条路径的 x 方向做随机翻转。
            sign = torch.where(torch.rand((num_envs, 1), device=self.device) > 0.5, 1.0, -1.0)
            step_x = step_x * sign

        self._waypoints_w[env_ids, :, 0] = root_pos_w[:, 0:1] + step_x
        self._waypoints_w[env_ids, :, 1] = root_pos_w[:, 1:2] + noise_y

        # 这里先保持 waypoint 的 z 与机器人当前根部等高。
        # 对当前这套任务来说，导航目标主要体现在平面位置；真正的越障由地形 + locomotion reward 学出来。
        self._waypoints_w[env_ids, :, 2] = root_pos_w[:, 2:3]

        # 每次重采样都从第 0 个 waypoint 开始追踪。
        self._wp_index[env_ids] = 0

    def _update_command(self):
        """更新当前 active waypoint，以及对应的命令输出。"""
        # ---------------------------------------------------------------------
        # 1) 先保存旧距离，再清空“本帧命中”脉冲量
        # ---------------------------------------------------------------------
        self._dist_prev[:] = self._dist_curr
        self.reached_this_step[:] = False

        all_env_ids = torch.arange(self.num_envs, device=self.device)

        # 先基于“当前 active waypoint”计算一次命令和当前距离。
        # 这一步得到的 _dist_curr 是“切目标前”的距离，用来判断是否命中。
        self._calc_command_for_envs(all_env_ids)

        # ---------------------------------------------------------------------
        # 2) 只有尚未完成整条路径的环境，才允许继续命中 waypoint
        # ---------------------------------------------------------------------
        reached = (self._dist_curr <= self.cfg.waypoint_radius) & (~self._path_completed)
        if not torch.any(reached):
            return

        reached_ids = torch.nonzero(reached, as_tuple=False).squeeze(-1)

        # 这一帧首次命中 waypoint：用于 sparse reward 的脉冲信号
        self.reached_this_step[reached_ids] = True

        # 记录累计命中数：curriculum 直接读取这个量，不再通过 reward episode sum 倒推
        self._reached_count[reached_ids] += 1

        old_idx = self._wp_index[reached_ids].clone()
        last_idx = self.cfg.num_waypoints - 1
        hit_final = old_idx >= last_idx

        # ---------------------------------------------------------------------
        # 3) 非终点：推进到下一个 waypoint
        # ---------------------------------------------------------------------
        if torch.any(~hit_final):
            non_final_ids = reached_ids[~hit_final]
            self._wp_index[non_final_ids] += 1

        # ---------------------------------------------------------------------
        # 4) 终点：锁存 path_completed，禁止后续重复领奖
        # ---------------------------------------------------------------------
        if torch.any(hit_final):
            final_ids = reached_ids[hit_final]
            self._wp_index[final_ids] = last_idx
            self._path_completed[final_ids] = True

        # waypoint 索引变化后，要立刻重新计算新 active waypoint 的 command。
        self._calc_command_for_envs(reached_ids)

        # 把 _dist_prev 对齐到“新目标”的当前距离，避免切目标时出现虚假的大 progress。
        self._dist_prev[reached_ids] = self._dist_curr[reached_ids]

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        """重置指定环境的 waypoint 状态，并返回上一个 episode 的统计信息。"""
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # 先把“旧 episode”的统计信息取出来，供 logger / tensorboard 查看。
        extras = {
            "waypoints_reached": torch.mean(self._reached_count[env_ids].float()).item(),
            "route_completed": torch.mean(self._path_completed[env_ids].float()).item(),
        }

        # 清空 episode 级状态。注意：这些状态必须在 super().reset() 之前清掉，
        # 这样本轮 reset 之后所有 manager 看到的都是新 episode 的干净状态。
        self._reached_count[env_ids] = 0
        self._path_completed[env_ids] = False
        self.reached_this_step[env_ids] = False
        self._dist_prev[env_ids] = 0.0
        self._dist_curr[env_ids] = 0.0
        self._command_out[env_ids] = 0.0

        # 关键修复：
        # 这里必须调用 CommandTerm.reset()，让基类正确维护：
        # - time_left
        # - command_counter
        # - _resample()
        #
        # 原版本自己重载 reset() 而没有同步 time_left，导致每个 episode 的第一步里
        # super().compute(dt) 都会再次触发一次 _resample_command()，路径被偷偷换掉。
        super().reset(env_ids)

        # 基类 reset 只负责重采样，不会主动把 command_out / dist_curr 刷到最新。
        # 所以这里补一次显式计算，确保 reset 后立刻就能拿到正确 observation。
        self._calc_command_for_envs(env_ids)
        self._dist_prev[env_ids] = self._dist_curr[env_ids]

        return extras

    # -------------------------------------------------------------------------
    # 核心数学：世界系 waypoint -> 机体系命令
    # -------------------------------------------------------------------------

    def _calc_command_for_envs(self, env_ids: torch.Tensor):
        """把当前 active waypoint 转换到机体系，写入 _command_out。"""
        if env_ids.numel() == 0:
            return

        root_pos_w = self._asset.data.root_pos_w[env_ids]
        root_quat_w = self._asset.data.root_quat_w[env_ids]

        idx = self._wp_index[env_ids]
        wp = self._waypoints_w[env_ids, idx]
        rel_w = wp - root_pos_w

        # 这里只提取 yaw，把世界系平面向量旋到机体系平面。
        # 对四足导航任务来说，观测当前目标的水平相对位置已经足够；
        # 而 roll / pitch 主要交给 proprioception + height scanner 自己处理。
        qw = root_quat_w[:, 0]
        qx = root_quat_w[:, 1]
        qy = root_quat_w[:, 2]
        qz = root_quat_w[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # 世界系 -> 机体系（绕 z 轴旋转 -yaw）
        rel_x_b = cy * rel_w[:, 0] + sy * rel_w[:, 1]
        rel_y_b = -sy * rel_w[:, 0] + cy * rel_w[:, 1]

        # 当前 waypoint 的平面距离
        dist = torch.sqrt(rel_x_b**2 + rel_y_b**2 + 1e-8)

        self._command_out[env_ids, 0] = rel_x_b
        self._command_out[env_ids, 1] = rel_y_b
        self._command_out[env_ids, 2] = dist
        self._dist_curr[env_ids] = dist

    # -------------------------------------------------------------------------
    # 调试可视化：让 ParkourPlayCfg 里的 debug_vis=True 真正生效
    # -------------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        """创建 / 隐藏 waypoint 可视化 marker。"""
        if debug_vis:
            if self._goal_visualizer is None:
                marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/waypoint_goal")
                marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
                self._goal_visualizer = VisualizationMarkers(cfg=marker_cfg)
            self._goal_visualizer.set_visibility(True)
        else:
            if self._goal_visualizer is not None:
                self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """每一帧把当前 active waypoint 画到场景里。"""
        if self._goal_visualizer is None:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        current_wp = self._waypoints_w[env_ids, self._wp_index]
        self._goal_visualizer.visualize(current_wp)

    # -------------------------------------------------------------------------
    # 对外暴露的观测 / 奖励 / 课程 / 终止接口
    # -------------------------------------------------------------------------

    @property
    def waypoint_rel_xy(self) -> torch.Tensor:
        """机体系下当前 waypoint 的相对 (x, y)，shape = (N, 2)。"""
        return self._command_out[:, 0:2]

    @property
    def waypoint_distance(self) -> torch.Tensor:
        """当前 waypoint 的平面距离，shape = (N,)。"""
        return self._command_out[:, 2]

    @property
    def waypoint_progress(self) -> torch.Tensor:
        """本 step 向当前 waypoint 靠近的距离，正值表示接近目标。"""
        return self._dist_prev - self._dist_curr

    @property
    def waypoint_reached_count(self) -> torch.Tensor:
        """本 episode 内累计首次命中的 waypoint 数量。"""
        return self._reached_count

    @property
    def path_completed(self) -> torch.Tensor:
        """本 episode 是否已经完成整条路径。"""
        return self._path_completed

    @property
    def num_waypoints(self) -> int:
        """路径总 waypoint 数。"""
        return int(self.cfg.num_waypoints)


@configclass
class WaypointCommandCfg(CommandTermCfg):
    class_type: type = WaypointCommand

    asset_name: str = "robot"

    # 因为这里的一条路径希望在整个 episode 内保持不变，
    # 所以把基类的重采样时间设成一个非常大的值。
    # 这样只有在 episode reset 时，才会真正重采样新路径。
    resampling_time_range: tuple[float, float] = (9999.0, 9999.0)

    # 现在 WaypointCommand 已经实现了 debug vis，所以 play 模式下可以安全打开。
    debug_vis: bool = False

    num_waypoints: int = 3
    waypoint_radius: float = 0.35
    x_range: tuple[float, float] = (1.5, 5.0)
    y_range: tuple[float, float] = (-1.0, 1.0)
    forward_only: bool = True
