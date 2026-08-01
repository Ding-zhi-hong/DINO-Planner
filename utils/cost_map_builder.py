"""
Metric Cost Map Builder — 仿照 ViPlanner 的 SemCostMap 方法

输入: 语义图 + 深度图 (单帧, 无需精确内参)
输出: 度量代价图 (X-Z 平面, 机器人局部坐标系)

管线 (与 ViPlanner 对应):
  深度→3D点云 ─→ 语义分类→代价 ─→ 滤波 ─→ 投影到2D度量栅格
  ─→ 空洞填充 ─→ 高斯平滑 ─→ 距离变换梯度 ─→ 最终平滑

坐标系: x_右, z_前 (与模型预测的 waypoints 一致)
栅格: (i, j) ↔ (x_start + i×res, z_start + j×res)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import KDTree
import math

# ═══════════════════════════════════════════════════════════════
# 31 类语义元数据 (与 ViPlanner 完全一致)
# ═══════════════════════════════════════════════════════════════

OBSTACLE_LOSS = 2.0
VIPLANNER_SEM_META = [
    {"name": "sidewalk",    "loss": 0.0, "color": [0, 255, 0]},
    {"name": "crosswalk",   "loss": 0.0, "color": [0, 102, 0]},
    {"name": "floor",       "loss": 0.0, "color": [0, 204, 0]},
    {"name": "stairs",      "loss": 0.0, "color": [0, 153, 0]},
    {"name": "gravel",      "loss": 0.5, "color": [204, 255, 0]},
    {"name": "sand",        "loss": 0.5, "color": [153, 204, 0]},
    {"name": "snow",        "loss": 0.5, "color": [204, 102, 0]},
    {"name": "indoor_soft", "loss": 1.0, "color": [102, 153, 0]},
    {"name": "terrain",     "loss": 1.0, "color": [255, 255, 0]},
    {"name": "road",        "loss": 0.0, "color": [255, 128, 0]},
    {"name": "person",      "loss": 2.0, "color": [255, 0, 0]},
    {"name": "anymal",      "loss": 2.0, "color": [204, 0, 0]},
    {"name": "vehicle",     "loss": 2.0, "color": [153, 0, 0]},
    {"name": "on_rails",    "loss": 2.0, "color": [51, 0, 0]},
    {"name": "motorcycle",  "loss": 2.0, "color": [102, 0, 0]},
    {"name": "bicycle",     "loss": 2.0, "color": [102, 0, 0]},
    {"name": "building",    "loss": 2.0, "color": [127, 0, 255]},
    {"name": "wall",        "loss": 2.0, "color": [102, 0, 204]},
    {"name": "fence",       "loss": 2.0, "color": [76, 0, 153]},
    {"name": "bridge",      "loss": 2.0, "color": [51, 0, 102]},
    {"name": "tunnel",      "loss": 2.0, "color": [51, 0, 102]},
    {"name": "pole",        "loss": 2.0, "color": [0, 0, 255]},
    {"name": "traffic_sign","loss": 2.0, "color": [0, 0, 153]},
    {"name": "traffic_light","loss": 2.0, "color": [0, 0, 204]},
    {"name": "bench",       "loss": 2.0, "color": [0, 0, 102]},
    {"name": "vegetation",  "loss": 2.0, "color": [153, 0, 153]},
    {"name": "water_surface","loss":2.0, "color": [204, 0, 204]},
    {"name": "sky",         "loss": 2.0, "color": [102, 0, 51]},
    {"name": "background",  "loss": 2.0, "color": [102, 0, 51]},
    {"name": "furniture",   "loss": 2.0, "color": [0, 0, 51]},
    {"name": "door",        "loss": 2.0, "color": [153, 153, 0]},
    {"name": "ceiling",     "loss": 2.0, "color": [25, 0, 51]},
    {"name": "static",      "loss": 2.0, "color": [0, 0, 0]},
    {"name": "dynamic",     "loss": 2.0, "color": [32, 0, 32]},
]
COLOR_TO_ID = {tuple(c["color"]): i for i, c in enumerate(VIPLANNER_SEM_META)}
COST_TABLE = np.array([c["loss"] for c in VIPLANNER_SEM_META], dtype=np.float32)
LOSS_LEVELS = np.sort(np.unique(COST_TABLE))  # [0.0, 0.5, 1.0, 1.5, 2.0]


# ═══════════════════════════════════════════════════════════════
# Step 1: 深度 → 3D 点云
# ═══════════════════════════════════════════════════════════════

def estimate_intrinsics(H, W, fov_deg=70):
    """
    无精确内参时，从图像尺寸和 FOV 估计

    Args:
        H, W: 图像高宽
        fov_deg: 水平视场角 (默认 70°, 常见手机相机)
    """
    fx = W / (2 * np.tan(np.radians(fov_deg / 2)))
    fy = fx
    cx, cy = W / 2, H / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def depth_to_pc(depth, K=None):
    """
    深度图 → 3D 点云 (相机坐标系)

    depth: (H, W) float32, 单位米
    K: (3,3) 或 None (自动估计)
    返回: (N, 3) — (x_右, y_下, z_前)
    """
    H, W = depth.shape
    if K is None:
        K = estimate_intrinsics(H, W)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u, v = u.astype(np.float32), v.astype(np.float32)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    valid = np.isfinite(depth) & (depth > 0.05)
    pts = np.stack([x, y, z], axis=-1)
    return pts[valid], valid


# ═══════════════════════════════════════════════════════════════
# Step 2: 语义图 → 逐像素 Class ID + Cost
# ═══════════════════════════════════════════════════════════════

def get_pixel_costs(sem_img):
    """
    sem_img: (3,H,W) uint8
    返回: (H,W) float32, 每像素遍历代价
    """
    H, W = sem_img.shape[1:]
    class_id = np.full((H, W), -1, dtype=np.int8)
    for (r, g, b), cid in COLOR_TO_ID.items():
        mask = (sem_img[0] == r) & (sem_img[1] == g) & (sem_img[2] == b)
        class_id[mask] = cid
    cost = COST_TABLE[class_id]
    cost[class_id == -1] = OBSTACLE_LOSS
    return cost


# ═══════════════════════════════════════════════════════════════
# Step 3: 3D 点 → 投影到 2D 度量栅格 (同 ViPlanner)
# ═══════════════════════════════════════════════════════════════

def pc_to_grid(points_3d, point_costs, valid_depth_mask, res=0.04,
               robot_h=0.7, robot_h_factor=3.0):
    """
    3D 点云 → 2D 度量栅格 (X-Z 地面平面)

    points_3d: (N, 3) — 已从深度过滤的3D点
    point_costs: (H, W) — 每像素的代价 (全图)
    valid_depth_mask: (H, W) — 有效深度像素的 bool mask
    """
    # 用有效深度 mask 过滤代价到 3D 点对应的数量
    costs = point_costs[valid_depth_mask]  # (N,)
    pts = points_3d

    # 滤波: 剔除高出机器人、地面以下的点
    keep = (pts[:, 1] < robot_h * robot_h_factor) & (pts[:, 1] > -0.5)
    pts = pts[keep]
    costs = costs[keep]

    if len(pts) < 50:
        # 点太少就扩大接受范围或返回空
        return None, None, None, None, None

    # 确定栅格范围 (同 ViPlanner)
    margin = 1.0
    x_min, x_max = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    z_min, z_max = pts[:, 2].min() - margin, pts[:, 2].max() + margin

    num_x = int(np.ceil((x_max - x_min) / res / 10)) * 10
    num_z = int(np.ceil((z_max - z_min) / res / 10)) * 10

    x_start = (x_max + x_min) / 2 - num_x / 2 * res
    z_start = (z_max + z_min) / 2 - num_z / 2 * res

    # 投影: 同栅格多点 → 取最高代价 (保守策略)
    gx = np.round((pts[:, 0] - x_start) / res).astype(int)
    gz = np.round((pts[:, 2] - z_start) / res).astype(int)
    in_bounds = (gx >= 0) & (gx < num_x) & (gz >= 0) & (gz < num_z)
    gx, gz, gc = gx[in_bounds], gz[in_bounds], costs[in_bounds]

    grid_flat = gx * num_z + gz
    order = np.argsort(grid_flat)
    gx, gz, gc, gf = gx[order], gz[order], gc[order], grid_flat[order]

    _, idx = np.unique(gf, return_index=True)
    grid = np.full((num_x, num_z), -10.0, dtype=np.float32)
    grid[gx[idx], gz[idx]] = gc[idx]

    return grid, float(x_start), float(z_start), num_x, num_z


# ═══════════════════════════════════════════════════════════════
# Step 4-5: 填充 + 平滑 + 距离变换梯度 (完全 ViPlanner)
# ═══════════════════════════════════════════════════════════════

def fill_holes(grid, unknown_val=-10.0, fill_val=OBSTACLE_LOSS):
    """空洞填充: 近邻复制 / 全空则填障碍物"""
    filled = grid.copy()
    hole = grid == unknown_val
    if not hole.any():
        return filled

    known = np.argwhere(~hole)
    unknown = np.argwhere(hole)
    if len(known) > 0 and len(unknown) > 0:
        kdt = KDTree(known)
        _, nn = kdt.query(unknown, k=1)
        filled[hole] = grid[known[nn, 0], known[nn, 1]]
    filled[filled == unknown_val] = fill_val
    return filled


def dist_gradient(mask, lo=0.0, hi=0.5, log_scale=False):
    """距离变换梯度 (同 ViPlanner _distance_based_gradient)"""
    d = np.zeros(mask.shape, dtype=np.float32)
    d[mask] = 1.0
    dist = distance_transform_edt(d)
    if log_scale:
        dist[dist > 0] = np.log(dist[dist > 0] + math.e)
    elif dist.max() > 0:
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        dist = dist * (hi - lo) + lo
    return dist[mask]


def apply_gradients(grid, neg_reward=0.5, obs_thresh=0.5, sigma_final=3.0):
    """距离变换梯度 (同 ViPlanner _dense_grid_loss)"""
    g = grid.copy()
    r = 2  # round decimal

    # 可通行 (0.0) → 负梯度
    m0 = np.round(g, r) == 0.0
    if m0.any():
        g[m0] = dist_gradient(m0, 0, abs(neg_reward)) * -1

    # 障碍物 (>阈值) → log 梯度
    mo = g > obs_thresh * LOSS_LEVELS[-1]
    if mo.any():
        g[mo] = dist_gradient(mo, log_scale=True)

    # 中间等级 → 线性
    for i in range(1, len(LOSS_LEVELS) - 1):
        mi = np.round(g, r) == LOSS_LEVELS[i]
        if mi.any():
            g[mi] = dist_gradient(mi, LOSS_LEVELS[i], LOSS_LEVELS[i + 1])

    if g.min() < 0:
        g += abs(g.min())
    return gaussian_filter(g, sigma=sigma_final, mode='reflect')


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def build_costmap(depth_map, sem_img, K=None,
                  resolution=0.04,
                  fov_deg=70,
                  negative_reward=0.5,
                  sigma_smooth=2.5,
                  sigma_final=3.0,
                  obstacle_threshold=0.5,
                  cost_override=None):
    """
    输入: 深度图 + 语义图
    输出: 度量代价图 (X-Z 平面, 机器人局部)

    当 K=None 时从 fov_deg 估计内参
    cost_override: (H,W) float32, 若提供则跳过 RGB 颜色匹配
    """
    H, W = depth_map.shape
    if K is None:
        K = estimate_intrinsics(H, W, fov_deg)

    # Step 1: 深度 → 3D 点
    pts_3d, valid = depth_to_pc(depth_map, K)

    # Step 2: 语义 → 代价 (或外部传入)
    costs = cost_override if cost_override is not None else get_pixel_costs(sem_img)

    # Step 3: 3D → 2D 度量栅格
    grid_info = pc_to_grid(pts_3d, costs, valid, res=resolution)
    if grid_info[0] is None:
        return None, None
    grid_arr, xs, zs, nx, nz = grid_info

    # Step 4: 填充 + 平滑
    # fill_holes 会用最近邻已知区域填充视野外的未知区域,
    # 让它们和可见道路一样自然地经过后续梯度管线处理
    grid_arr = fill_holes(grid_arr)
    grid_arr = gaussian_filter(grid_arr, sigma=sigma_smooth, mode='reflect')

    # Step 5: 距离变换梯度
    cost_arr = apply_gradients(grid_arr,
                                neg_reward=negative_reward,
                                obs_thresh=obstacle_threshold,
                                sigma_final=sigma_final)

    meta = {
        'x_start': xs, 'z_start': zs,
        'num_x': nx, 'num_z': nz,
        'resolution': resolution,
    }
    return cost_arr, meta


# ═══════════════════════════════════════════════════════════════
# Pos2Ind (训练时用 F.grid_sample 采样)
# ═══════════════════════════════════════════════════════════════

def waypoints_to_grid_norm(wp, x_start, z_start, res, nx, nz):
    """
    路径点 (x_右,z_前) → 归一化坐标 [-1,1]

    同 ViPlanner Pos2Ind:
      H = (points - origin) / resolution
      H = (H - [nx/2, nz/2]) / [nx/2, nz/2]
    """
    gx = (wp[:, 0] - x_start) / res
    gz = (wp[:, 1] - z_start) / res
    return np.stack([
        (gx - nx/2) / (nx/2),
        (gz - nz/2) / (nz/2),
    ], axis=-1)


# ═══════════════════════════════════════════════════════════════
# 命令行
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True)
    ap.add_argument("--sem", required=True)
    ap.add_argument("--out", default="cost_map.npy")
    ap.add_argument("--fov", type=float, default=70)
    args = ap.parse_args()

    depth = np.load(args.depth).squeeze()
    sem = np.load(args.sem)
    cm, meta = build_costmap(depth, sem, fov_deg=args.fov)

    if cm is not None:
        np.save(args.out, cm)
        xs, zs = meta['x_start'], meta['z_start']
        nx, nz = meta['num_x'], meta['num_z']
        print(f"Cost map: {nx}×{nz}, range=[{cm.min():.3f}, {cm.max():.3f}]")
        print(f"Grid: x=[{xs:.2f},{xs+nx*0.04:.2f}], z=[{zs:.2f},{zs+nz*0.04:.2f}]")
    else:
        print("Failed: too few valid points")
