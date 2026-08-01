"""
测试 SemanticSegmentor — 语义图 + 代价图可视化
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.semantic import SemanticSegmentor

IMAGE = r'C:\Users\34287\Desktop\000001.jpg'
OUT = r'C:\Users\34287\Desktop\test_semantic.png'

img = cv2.imread(IMAGE)
if img is None:
    raise FileNotFoundError(IMAGE)
print(f"Image: {img.shape[1]}x{img.shape[0]}")

seg = SemanticSegmentor()
sem_img, cost_map = seg.predict(img)
print(f"Semantic: {sem_img.shape}")
print(f"Cost range: [{cost_map.min():.2f}, {cost_map.max():.2f}]")
print(f"  traversable (<=0.5)={(cost_map<=0.5).mean()*100:.1f}%")
print(f"  obstacle (=2.0)={(cost_map>=2.0).mean()*100:.1f}%")

# 检测到的类
detected = set()
cost_table = seg.get_cost_table()
for name, cost in cost_table.items():
    for c in [list(cost_map.flatten())]:  # skip, just check existence via unique values
        pass
unique_costs = np.unique(cost_map)
print(f"Unique costs: {sorted(unique_costs)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原图
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("RGB")
axes[0].axis('off')

# 语义
sem_viz = sem_img.transpose(1, 2, 0)
axes[1].imshow(sem_viz)
axes[1].set_title("VIPlanner Semantic (31 classes)")
axes[1].axis('off')

# 代价
im = axes[2].imshow(cost_map, cmap='RdYlGn_r', vmin=0, vmax=2)
plt.colorbar(im, ax=axes[2], fraction=0.046, ticks=[0, 0.5, 1, 1.5, 2])
axes[2].set_title("Traversal Cost")
axes[2].axis('off')

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
plt.show()
