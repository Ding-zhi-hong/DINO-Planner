"""
代价图坐标对齐 — 同 ViPlanner 的 Pos2Ind

ViPlanner 中:
  路径点 (相机系) → TransformPoints(odom) → 世界系
  → Pos2Ind(世界系) → 归一化坐标 → F.grid_sample(代价图)

我们:
  路径点 (x_右, z_前) 已经和代价图在同一坐标系
  → Pos2Ind(路径点) → 归一化坐标 → F.grid_sample(代价图)

用法:
  aligner = CostMapAligner(meta)
  norm_coords = aligner.pos_to_ind(waypoints)  # (B, N, 2) → (B, 1, N, 2)
  sampled = F.grid_sample(cost_map, norm_coords, mode='bicubic')
"""

import torch
import numpy as np


class CostMapAligner:
    """
    度量代价图坐标对齐器

    功能 (同 ViPlanner CostMapPCD.Pos2Ind):
      将机器人坐标系的路径点 (x_右, z_前)
      转为代价图上的归一化坐标 [-1, 1]

    公式:
      grid_x = (x - x_start) / resolution   # 栅格坐标
      norm_x = (grid_x - num_x/2) / (num_x/2) # → [-1, 1]
    """

    def __init__(self, meta: dict):
        """
        meta: 来自 build_costmap() 的返回值
          x_start, z_start, num_x, num_z, resolution
        """
        self.x_start = meta['x_start']
        self.z_start = meta['z_start']
        self.num_x = meta['num_x']
        self.num_z = meta['num_z']
        self.resolution = meta['resolution']

    def pos_to_ind(self, points):
        """
        路径点 → 归一化坐标 (同 ViPlanner Pos2Ind)

        Args:
            points: (B, N, 2) — (x_右, z_前) 在机器人坐标系

        Returns:
            (B, 1, N, 2) — 归一化坐标 [-1, 1]
                           可用于 F.grid_sample
        """
        device = points.device
        xs = torch.tensor(self.x_start, device=device, dtype=torch.float64)
        zs = torch.tensor(self.z_start, device=device, dtype=torch.float64)
        nx = torch.tensor(self.num_x, device=device, dtype=torch.float64)
        nz = torch.tensor(self.num_z, device=device, dtype=torch.float64)
        res = torch.tensor(self.resolution, device=device, dtype=torch.float64)

        pts = points.to(torch.float64)

        # Step 1: 米 → 栅格坐标
        grid_x = (pts[..., 0:1] - xs) / res
        grid_z = (pts[..., 1:2] - zs) / res

        # Step 2: 栅格坐标 → 归一化 [-1, 1] (同 ViPlanner)
        norm_x = (grid_x - nx / 2.0) / (nx / 2.0)
        norm_z = (grid_z - nz / 2.0) / (nz / 2.0)

        # cost_map shape=(num_x, num_z) → grid_sample: [...,0]=W=z, [...,1]=H=x
        ind = torch.cat([norm_z, norm_x], dim=-1).to(torch.float32)
        return ind.unsqueeze(1)

    def ind_to_pos(self, indices):
        """
        归一化坐标 → 路径点 (逆变换, 调试用)

        Args:
            indices: (B, 1, N, 2) — 归一化坐标 [-1, 1]

        Returns:
            (B, N, 2) — (x_右, z_前)
        """
        device = indices.device
        xs = torch.tensor(self.x_start, device=device, dtype=torch.float64)
        zs = torch.tensor(self.z_start, device=device, dtype=torch.float64)
        nx = torch.tensor(self.num_x, device=device, dtype=torch.float64)
        nz = torch.tensor(self.num_z, device=device, dtype=torch.float64)
        res = torch.tensor(self.resolution, device=device, dtype=torch.float64)

        ind = indices.squeeze(1).to(torch.float64)

        # grid_sample 返回的是 (norm_z, norm_x)
        grid_z = ind[..., 0:1] * (nz / 2.0) + nz / 2.0
        grid_x = ind[..., 1:2] * (nx / 2.0) + nx / 2.0

        x = grid_x * res + xs
        z = grid_z * res + zs

        return torch.cat([x, z], dim=-1).to(torch.float32)

    def sample_cost_map(self, cost_map, waypoints, mode='bicubic'):
        """
        在代价图上采样路径点的代价 (同 ViPlanner)

        Args:
            cost_map: (B, 1, num_z, num_x) — 注意 grid_sample 格式
            waypoints: (B, N, 2) — (x_右, z_前)

        Returns:
            (B, 1, N) — 每个路径点的代价
        """
        norm_coords = self.pos_to_ind(waypoints)  # (B, 1, N, 2)
        sampled = torch.nn.functional.grid_sample(
            cost_map,                # (B, 1, num_z, num_x)
            norm_coords,             # (B, 1, N, 2)
            mode=mode,
            padding_mode='border',
            align_corners=False,
        )  # (B, 1, 1, N)
        return sampled.squeeze(2)  # (B, 1, N)

    def get_grid_extent(self):
        """代价图的范围 (调试用)"""
        x_min = self.x_start
        x_max = self.x_start + self.num_x * self.resolution
        z_min = self.z_start
        z_max = self.z_start + self.num_z * self.resolution
        return (x_min, x_max, z_min, z_max)

    def __repr__(self):
        return (f"CostMapAligner(x=[{self.x_start:.2f}, "
                f"{self.x_start + self.num_x * self.resolution:.2f}], "
                f"z=[{self.z_start:.2f}, "
                f"{self.z_start + self.num_z * self.resolution:.2f}], "
                f"res={self.resolution}, grid={self.num_x}×{self.num_z})")


def point_cost(x, z, cost_map, meta):
    """
    工具函数: 一个坐标点 (x_右, z_前) → 代价图上对应的代价

    参数:
        x, z: float — 机器人坐标系下的坐标 (米)
        cost_map: (num_x, num_z) numpy array
        meta: dict — 来自 build_costmap() 的返回值

    返回:
        float — 该点的遍历代价
    """
    aligner = CostMapAligner(meta)
    pt = torch.tensor([[x, z]], dtype=torch.float32).unsqueeze(0)  # (1,1,2)
    norm = aligner.pos_to_ind(pt)
    cm_t = torch.from_numpy(cost_map).float().unsqueeze(0).unsqueeze(0)
    cost = torch.nn.functional.grid_sample(
        cm_t, norm, mode='bilinear', padding_mode='border', align_corners=False
    )[0, 0, 0].item()
    return cost
