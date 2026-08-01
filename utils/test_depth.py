"""
测试 DepthEstimator — 深度图可视化
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.depth import DepthEstimator

IMAGE = r'C:\Users\34287\Desktop\000001.jpg'
OUT = r'C:\Users\34287\Desktop\test_depth.png'

img = cv2.imread(IMAGE)
if img is None:
    raise FileNotFoundError(IMAGE)
print(f"Image: {img.shape[1]}x{img.shape[0]}")

de = DepthEstimator()
depth = de.predict(img)
print(f"Depth: {depth.shape}, [{depth.min():.2f}, {depth.max():.2f}m]")
print(f"  median={np.median(depth):.2f}m, >1m={(depth>1).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("RGB")
axes[0].axis('off')

vmax = np.percentile(depth[depth>0], 95)
im = axes[1].imshow(depth, cmap='plasma', vmin=0, vmax=vmax)
axes[1].set_title(f"Depth [0, {vmax:.1f}m]")
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
plt.show()
