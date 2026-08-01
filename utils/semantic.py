"""
语义分割工具 — Mask2Former (ModelScope) → 31类 VIPlanner 颜色+代价

用法:
  from utils.semantic import SemanticSegmentor
  seg = SemanticSegmentor()
  sem_img, cost_map = seg.predict(image_bgr)  # (3,H,W), (H,W)
"""

import numpy as np
import cv2
import torch
import os

# COCO 171 → VIPlanner 31 类映射
COCO_TO_VIP = {
    "road": ["road"], "sidewalk": ["pavement-merged"],
    "floor": ["floor-other-merged", "floor-wood", "platform", "playingfield", "rug-merged"],
    "gravel": ["gravel"], "stairs": ["stairs"], "sand": ["sand"], "snow": ["snow"],
    "person": ["person"],
    "anymal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["car", "bus", "truck", "boat"],
    "on_rails": ["train", "railroad"], "motorcycle": ["motorcycle"], "bicycle": ["bicycle"],
    "building": ["building-other-merged", "house", "roof"],
    "wall": ["wall-other-merged", "curtain", "mirror-stuff", "wall-brick",
             "wall-stone", "wall-tile", "wall-wood", "window-blind", "window-other"],
    "fence": ["fence-merged"], "bridge": ["bridge"],
    "pole": ["fire hydrant", "parking meter"], "traffic_sign": ["stop sign"],
    "traffic_light": ["traffic light"], "bench": ["bench"],
    "vegetation": ["potted plant", "flower", "tree-merged", "mountain-merged", "rock-merged"],
    "terrain": ["grass-merged", "dirt-merged"],
    "water_surface": ["river", "sea", "water-other"],
    "sky": ["sky-other-merged", "airplane"],
    "dynamic": ["backpack", "umbrella", "handbag", "tie", "suitcase", "book",
                "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "microwave", "oven", "toaster", "sink",
                "refrigerator", "banana", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "fruit",
                "food-other-merged", "apple", "mouse", "remote", "keyboard",
                "cell phone", "laptop", "scissors", "teddy bear", "hair drier",
                "toothbrush", "net", "paper-merged"],
    "static": ["banner", "cardboard", "light", "tent", "unknown"],
    "furniture": ["chair", "couch", "bed", "dining table", "toilet", "clock",
                  "vase", "blanket", "pillow", "shelf", "cabinet",
                  "table-merged", "counter", "tv"],
    "door": ["door-stuff"], "ceiling": ["ceiling-merged"],
    "indoor_soft": ["towel"],
}

# VIPlanner 31类颜色
VIP_COLORS = {
    "sidewalk": [0, 255, 0], "crosswalk": [0, 102, 0], "floor": [0, 204, 0],
    "stairs": [0, 153, 0], "gravel": [204, 255, 0], "sand": [153, 204, 0],
    "snow": [204, 102, 0], "indoor_soft": [102, 153, 0], "terrain": [255, 255, 0],
    "road": [255, 128, 0],
    "person": [255, 0, 0], "anymal": [204, 0, 0], "vehicle": [153, 0, 0],
    "on_rails": [51, 0, 0], "motorcycle": [102, 0, 0], "bicycle": [102, 0, 0],
    "building": [127, 0, 255], "wall": [102, 0, 204], "fence": [76, 0, 153],
    "bridge": [51, 0, 102], "tunnel": [51, 0, 102],
    "pole": [0, 0, 255], "traffic_sign": [0, 0, 153], "traffic_light": [0, 0, 204],
    "bench": [0, 0, 102],
    "vegetation": [153, 0, 153], "water_surface": [204, 0, 204],
    "sky": [102, 0, 51], "background": [102, 0, 51],
    "furniture": [0, 0, 51], "door": [153, 153, 0], "ceiling": [25, 0, 51],
    "static": [0, 0, 0], "dynamic": [32, 0, 32],
}

# VIPlanner 31类→遍历代价
VIP_COST = {
    "sidewalk": 0.0, "crosswalk": 0.0, "floor": 0.0, "stairs": 0.0,
    "gravel": 0.5, "sand": 0.5, "snow": 0.5,
    "indoor_soft": 1.0, "terrain": 1.0,
    "road": 0.0,
    "person": 2.0, "anymal": 2.0, "vehicle": 2.0, "on_rails": 2.0,
    "motorcycle": 2.0, "bicycle": 2.0,
    "building": 2.0, "wall": 2.0, "fence": 2.0, "bridge": 2.0, "tunnel": 2.0,
    "pole": 2.0, "traffic_sign": 2.0, "traffic_light": 2.0, "bench": 2.0,
    "vegetation": 2.0, "water_surface": 2.0,
    "sky": 2.0, "background": 2.0,
    "furniture": 2.0, "door": 2.0, "ceiling": 2.0,
    "static": 2.0, "dynamic": 2.0,
}

MODEL_DIR = (r'C:\Users\34287\.cache\modelscope\hub\models'
             r'\facebook\mask2former-swin-tiny-coco-panoptic')


class SemanticSegmentor:
    """Mask2Former 语义分割 → VIPlanner 31类颜色+代价"""

    _instance = None
    _model = None
    _processor = None
    _id_to_vip = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self):
        if self._model is not None:
            return
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_DIR).cuda().eval()
        self._processor = Mask2FormerImageProcessor.from_pretrained(MODEL_DIR)
        # 构建 COCO id → VIPlanner 名称映射
        self._id_to_vip = {}
        for cid, cname in self._model.config.id2label.items():
            matched = False
            for vip_name, keywords in COCO_TO_VIP.items():
                if any(kw in cname for kw in keywords):
                    self._id_to_vip[cid] = vip_name
                    matched = True
                    break
            if not matched:
                self._id_to_vip[cid] = "static"
        print("[Semantic] Mask2Former loaded")

    def predict(self, image_bgr):
        """
        image_bgr: (H,W,3) BGR uint8
        返回: sem_img (3,H,W) uint8 VIPlanner 颜色
              cost_map (H,W) float32 遍历代价 [0, 2]
        """
        self._load()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=image_rgb, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self._model(**inputs)
        sem_seg = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_bgr.shape[:2]]
        )[0].cpu().numpy()

        out_color = np.zeros((*sem_seg.shape, 3), dtype=np.uint8)
        out_cost = np.ones(sem_seg.shape, dtype=np.float32) * 2.0

        for coco_id, vip_name in self._id_to_vip.items():
            mask = sem_seg == coco_id
            if not mask.any():
                continue
            out_color[mask] = VIP_COLORS.get(vip_name, [0, 0, 0])
            out_cost[mask] = VIP_COST.get(vip_name, 2.0)

        return out_color.transpose(2, 0, 1), out_cost

    @staticmethod
    def get_cost_table():
        """返回 {类名: 代价} 映射, 供其他模块使用"""
        return dict(VIP_COST)
