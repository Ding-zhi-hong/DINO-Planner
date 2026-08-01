"""
深度估计工具 — DepthAnythingV2 Metric → 米制深度图

用法:
  from utils.depth import DepthEstimator
  de = DepthEstimator()
  depth = de.predict(image_bgr)  # (H, W) float32 米
"""

import numpy as np
import cv2
import torch
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRIC_PATH = os.path.join(BASE, 'Depth-Anything-V2', 'metric_depth')
CKPT = os.path.join(BASE, 'Depth-Anything-V2', 'checkpoints',
                    'depth_anything_v2_metric_vkitti_vits.pth')


class DepthEstimator:
    """DepthAnythingV2 Metric 单目深度估计 (米制)"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self):
        if self._model is not None:
            return
        import sys
        sys.path.insert(0, METRIC_PATH)
        from depth_anything_v2.dpt import DepthAnythingV2
        self._model = DepthAnythingV2(encoder='vits', features=64,
                                       out_channels=[48, 96, 192, 384])
        state = torch.load(CKPT, map_location='cuda')
        # metric 权重可能缺少一些 key, 用 strict=False 加载
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Depth] missing keys: {missing}")
        if unexpected:
            print(f"[Depth] unexpected keys: {unexpected}")
        self._model = self._model.cuda().eval()
        print("[Depth] DepthAnythingV2 Metric loaded")

    def predict(self, image_bgr):
        """
        image_bgr: (H,W,3) BGR uint8
        返回: (H,W) float32 米制深度
        """
        self._load()
        with torch.no_grad():
            depth = self._model.infer_image(image_bgr)
        return depth.astype(np.float32)
