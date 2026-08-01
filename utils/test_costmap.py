"""
测试 完整管线: RGB → 语义 + 深度 → 度量代价图
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.semantic import SemanticSegmentor
from utils.depth import DepthEstimator
from utils.cost_map_builder import build_costmap

IMAGE = r'C:\Users\34287\Desktop\000001.jpg'
OUT = r'C:\Users\34287\Desktop\test_costmap.png'

img = cv2.imread(IMAGE)
if img is None:
    raise FileNotFoundError(IMAGE)
print(f"Image: {img.shape[1]}x{img.shape[0]}")

# 语义
seg = SemanticSegmentor()
sem_img, cost_override = seg.predict(img)
print(f"Semantic cost: [{cost_override.min():.2f}, {cost_override.max():.2f}]")

# 深度
de = DepthEstimator()
depth = de.predict(img)
print(f"Depth: [{depth.min():.2f}, {depth.max():.2f}m]")

# 代价图
cost_map, meta = build_costmap(depth, sem_img, cost_override=cost_override, fov_deg=70)
if cost_map is None:
    print("Failed to build cost map")
    exit(1)

xs, zs, nx, nz, res = (meta['x_start'], meta['z_start'],
                         meta['num_x'], meta['num_z'], meta['resolution'])
print(f"Cost map: {cost_map.shape}, [{cost_map.min():.3f}, {cost_map.max():.3f}]")
print(f"Grid: x=[{xs:.2f},{xs+nx*res:.2f}]m, z=[{zs:.2f},{zs+nz*res:.2f}]m, res={res}m")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# RGB
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("RGB")
axes[0, 0].axis('off')

# 语义
axes[0, 1].imshow(sem_img.transpose(1, 2, 0))
axes[0, 1].set_title(f"Semantic [{cost_override.min():.2f}, {cost_override.max():.2f}]")
axes[0, 1].axis('off')

# 深度
vmax = np.percentile(depth[depth>0], 95)
im_d = axes[1, 0].imshow(depth, cmap='plasma', vmin=0, vmax=vmax)
axes[1, 0].set_title(f"Depth [0, {vmax:.1f}m]")
axes[1, 0].axis('off')
plt.colorbar(im_d, ax=axes[1, 0], fraction=0.046)

# 度量代价图
extent = [xs, xs+nx*res, zs, zs+nz*res]
im_c = axes[1, 1].imshow(cost_map.T, cmap='jet', vmin=0, vmax=2.5,
                          origin='lower', extent=extent)
axes[1, 1].set_title(f"Metric Cost Map [{cost_map.min():.2f}, {cost_map.max():.2f}]")
axes[1, 1].set_xlabel("X (right) [m]")
axes[1, 1].set_ylabel("Z (forward) [m]")
plt.colorbar(im_c, ax=axes[1, 1], fraction=0.046)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
plt.show()
