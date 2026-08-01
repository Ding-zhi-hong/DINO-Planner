"""
纯推理脚本：输入图片 + 目标点 → 输出 5 个路径关键点

最小依赖：仅加载模型权重，无需语义/深度/代价图/可视化。
适用于将模型集成到导航管线中。

用法:
  python predict.py --image path/to/image.jpg --goal_x 2.0 --goal_z 5.0
  python predict.py --image path/to/image.jpg --goal_x 2.0 --goal_z 5.0 \\
                    --ckpt results/single_dinov2_run/checkpoints/best.pth

输出:
  5 个关键点 (k, 2): 每行一个 (x_right, z_forward)，单位米
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SingleDINOv2Nav


def load_config(config_path):
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    class Cfg:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Cfg(v))
                else:
                    setattr(self, k, v)
    return Cfg(cfg_dict)


def load_model(ckpt_path, cfg, device):
    model = SingleDINOv2Nav(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f'[Model] Loaded from {ckpt_path}', file=sys.stderr)
    return model


def preprocess(image_bgr, target_h=360, target_w=640):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_w, target_h))
    rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).float().unsqueeze(0)


def predict(model, image_bgr, goal_xy, device):
    """
    单张图片推理

    Returns:
        keypoints: (5, 2) numpy — 5 个路径关键点 (x_right, z_forward)
        fear: float — 碰撞概率 [0, 1]
    """
    rgb_t = preprocess(image_bgr).to(device)
    goal_t = torch.from_numpy(np.array(goal_xy, dtype=np.float32)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        keypoints, fear = model(rgb_t, goal_t)

    kp = keypoints.cpu().numpy()[0]                       # (5, 2)
    return kp, fear.cpu().numpy()[0, 0]


def main():
    parser = argparse.ArgumentParser(description='SingleDINOv2Nav 纯推理')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--goal_x', type=float, required=True, help='目标点 x (右方向, 米)')
    parser.add_argument('--goal_z', type=float, required=True, help='目标点 z (前方向, 米)')
    parser.add_argument('--ckpt',
                        default='results/single_dinov2_run/checkpoints/best.pth',
                        help='模型权重路径')
    parser.add_argument('--config', default='config/single_dinov2.yaml',
                        help='模型配置文件')
    parser.add_argument('--cpu', action='store_true', help='强制 CPU')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')

    cfg = load_config(args.config)
    model = load_model(args.ckpt, cfg, device)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f'[Error] Cannot read image: {args.image}', file=sys.stderr)
        sys.exit(1)

    goal_xy = np.array([args.goal_x, args.goal_z], dtype=np.float32)

    keypoints, fear = predict(model, image_bgr, goal_xy, device)

    # ── 输出到 stdout ──
    print(f'# fear={fear:.4f}')
    print(f'# keypoints shape={keypoints.shape}')
    for pt in keypoints:
        print(f'{pt[0]:.6f} {pt[1]:.6f}')


if __name__ == '__main__':
    main()