"""
ViPlanner 损失函数 (2D适配版)

损失 = Goal Loss + Obstacle Loss + Motion Loss + Fear Loss

完全参照 ViPlanner TrajCost.CostofTraj:
  Goal:    log(‖wp[-1] - goal‖ + 1) × 4.0
  Obs:     sum(代价图采样值) × 0.25     (3轨扩展, 全部点求和)
  Motion:  |实际步长 - 理想步长| × 1.5
  Fear:    BCE(碰撞概率) × 1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate(keypoints, step=0.1):
    """
    5关键点 → Hermite插值 → 51路径点
    同 ViPlanner TrajGeneratorFromPFreeRot
    """
    B, N, D = keypoints.shape
    pts = torch.cat([
        torch.zeros(B, 1, D, device=keypoints.device, requires_grad=keypoints.requires_grad),
        keypoints
    ], dim=1)
    N = N + 1
    xs = torch.arange(0, N - 1 + step, step, device=keypoints.device).unsqueeze(0).expand(B, -1)
    x = torch.arange(N, device=keypoints.device, dtype=keypoints.dtype).unsqueeze(0).expand(B, -1)
    m = (pts[:, 1:] - pts[:, :-1]) / (x[:, 1:] - x[:, :-1]).unsqueeze(-1)
    m = torch.cat([m[:, 0:1], (m[:, 1:] + m[:, :-1]) / 2, m[:, -1:]], dim=1)

    idxs = torch.searchsorted(x[0, :-1], xs[0, :]).clamp(0, N - 2)
    dx = (x[:, idxs + 1] - x[:, idxs]).unsqueeze(-1)
    t = ((xs - x[:, idxs]) / dx.squeeze(-1)).unsqueeze(-1)
    t2 = t * t
    t3 = t2 * t
    h00, h10, h01, h11 = 2*t3-3*t2+1, t3-2*t2+t, -2*t3+3*t2, t3-t2
    return (h00 * pts[:, idxs] + h10 * dx * m[:, idxs] +
            h01 * pts[:, idxs + 1] + h11 * dx * m[:, idxs + 1])  # (B, 51, 2)


class ViPlannerLoss(nn.Module):
    """
    ViPlanner 损失函数

    输入:
      keypoints: (B,5,2) 预测关键点
      fear:      (B,1)   碰撞概率
      goal:      (B,2)   目标位置 (x_右, z_前)
      cost_maps: list[(num_x, num_z)] 每样本独立的代价图
      aligners:  list[CostMapAligner]
    """
    def __init__(self, cfg):
        super().__init__()
        self.w_goal = cfg.loss.w_goal
        self.w_obs = cfg.loss.w_obs
        self.w_motion = cfg.loss.w_motion
        self.w_fear = cfg.loss.w_fear
        self.fear_ahead = cfg.loss.fear_ahead_dist
        self.obs_thresh = getattr(cfg.loss, 'obstacle_thresh', 1.25)
        self.robot_half_width = 0.3

    def forward(self, keypoints, fear, goal, cost_maps, aligners):
        B, device = keypoints.shape[0], keypoints.device
        wps = interpolate(keypoints)  # (B, 51, 2)

        # ═══════════════════════════════════════════════════
        # 1. Goal Loss — 同 ViPlanner
        # ═══════════════════════════════════════════════════
        gloss_M = torch.norm(goal[:, :2] - wps[:, -1, :2], dim=1)  # (B,)
        goal_loss = torch.mean(torch.log(gloss_M + 1.0))

        # ═══════════════════════════════════════════════════
        # 2. Obstacle Loss — 同 ViPlanner (2D适配)
        #    3轨扩展 → grid_sample → sum全部点 → 平均
        # ═══════════════════════════════════════════════════
        obs_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        for b in range(B):
            if cost_maps[b] is None or aligners[b] is None:
                continue
            cm = cost_maps[b].to(device)
            al = aligners[b]
            wp = wps[b]  # (51, 2) — 使用全部51个点

            # 切线 + 法线 (同 ViPlanner _compute_oloss)
            tangent = wp[1:] - wp[:-1]                           # (50, 2)
            t_norm = torch.norm(tangent, dim=1, keepdim=True)
            tangent = tangent / (t_norm + 1e-8)
            normal = tangent[:, [1, 0]] * torch.tensor([-1, 1], device=device)

            # 3轨扩展 (同 ViPlanner: ±robot_width/2)
            left   = wp[:-1] - self.robot_half_width * normal     # (50, 2)
            center = wp[:-1]
            right  = wp[:-1] + self.robot_half_width * normal

            # 合并: ViPlanner 用 torch.vstack([...]*3), 结果 (3*B, 50, 3)
            tracks = torch.cat([left, center, right], dim=0).unsqueeze(0)  # (1, 150, 2)

            # 坐标对齐 → grid_sample
            nc = al.pos_to_ind(tracks)
            sampled = F.grid_sample(
                cm.unsqueeze(0).unsqueeze(0), nc,
                mode='bicubic', padding_mode='border', align_corners=False,
            )[0, 0, 0]  # (150,)

            # ViPlanner 方式: sum over all points per track, no clipping
            # 但因我们代价图路面 ≈0.45, 减0.5将道路置0
            cost = (sampled - 0.5).clamp(min=0)  # 道路→0, 障碍物→保留
            tracks_sum = cost.reshape(3, 50).sum(dim=1)  # (3,) — 每条轨50个点求和
            obs_loss = obs_loss + tracks_sum.mean()       # 3轨平均
            n_valid += 1

        if n_valid > 0:
            obs_loss = obs_loss / n_valid  # batch平均

        # ═══════════════════════════════════════════════════
        # 3. Motion Loss — 同 ViPlanner
        #    对比理想步长 (从目标生成理想轨迹)
        # ═══════════════════════════════════════════════════
        num_p = wps.shape[1]  # 51

        # 从原点到目标的理想轨迹 (均匀分布)
        t = torch.linspace(0, 1, num_p, device=device).unsqueeze(0).expand(B, -1)
        ideal_wp = goal[:, None, :2] * t[:, :, None]  # (B, 51, 2)

        # 步长
        ideal_ds = torch.norm(ideal_wp[:, 1:] - ideal_wp[:, :-1], dim=2)  # (B, 50)
        wp_ds = torch.norm(wps[:, 1:] - wps[:, :-1], dim=2)               # (B, 50)

        # Motion Loss: sum of |actual - ideal| over path, mean over batch
        motion_loss = torch.mean(torch.sum(torch.abs(wp_ds - ideal_ds), dim=1))

        # ═══════════════════════════════════════════════════
        # 4. Fear Loss — 同 ViPlanner
        # ═══════════════════════════════════════════════════
        cum_len = torch.cumsum(wp_ds, dim=1)  # (B, 50)

        fear_labels = torch.zeros(B, 1, device=device)
        for b in range(B):
            if cost_maps[b] is None or aligners[b] is None:
                continue
            al = aligners[b]
            cm = cost_maps[b].to(device)
            wp = wps[b].unsqueeze(0)

            nc = al.pos_to_ind(wp)
            vals = F.grid_sample(
                cm.unsqueeze(0).unsqueeze(0), nc,
                mode='bicubic', padding_mode='border', align_corners=False,
            )[0, 0, 0]

            # 前 fear_ahead 米内的最大代价 > 阈值 → 碰撞标签=1
            mask = cum_len[b] <= self.fear_ahead
            if mask.any():
                fear_labels[b] = (vals[:-1][mask].max().item() > self.obs_thresh)
            else:
                fear_labels[b] = 0.0

        fear_loss = F.binary_cross_entropy(fear, fear_labels)

        # ═══════════════════════════════════════════════════
        # Total — 同 ViPlanner
        # ═══════════════════════════════════════════════════
        total = (self.w_goal * goal_loss +
                 self.w_obs * obs_loss +
                 self.w_motion * motion_loss +
                 self.w_fear * fear_loss)

        return total, {
            'goal': goal_loss.item(),
            'obs': obs_loss.item(),
            'motion': motion_loss.item(),
            'fear': fear_loss.item(),
        }