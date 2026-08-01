"""
推理脚本：输入一张图片 + 目标点 → 可视化预测轨迹 + 代价地图

模型: SingleDINOv2Nav
  - 仅需 RGB 图像, 无需深度图输入
  - DINOv2(冻结) + DPT 多尺度融合 → 1024×12×20 → Decoder → 5关键点 + 恐惧值

用法:
  python infer.py --image path/to/image.jpg --goal_x 2.0 --goal_z 5.0
  python infer.py --image path/to/image.jpg --goal_x 2.0 --goal_z 5.0 \\
                  --ckpt results/single_dinov2_run/checkpoints/best.pth

可视化（3 面板）:
  左：原始 RGB
  中：预测轨迹（鸟瞰，x-z 平面）
  右：代价地图 + 预测轨迹叠加
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SingleDINOv2Nav
from utils import DepthEstimator, SemanticSegmentor, build_costmap, CostMapAligner


def load_config(config_path='config/single_dinov2.yaml'):
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
    print(f'[Model] Loaded from {ckpt_path}')
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] Parameters: total={total:,}  trainable={trainable:,}')
    return model


def preprocess_for_model(rgb_bgr, goal, device,
                         seg=None, de=None):
    """
    输入：BGR 图像 → 模型输入 tensor + 代价图 + 对齐器

    SingleDINOv2Nav 仅需要 RGB + goal。
    深度+语义只用于构建代价图（可视化损失），不输入模型。

    返回：
      rgb_t:    (1,3,360,640) tensor — 模型输入
      goal_t:   (1,2) tensor
      cost_map: (nx,nz) numpy — 代价图（可视化用）
      aligner:  CostMapAligner
    """
    # ── RGB 原图 (模型唯一视觉输入) ──
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)          # (H,W,3)
    target_h, target_w = 360, 640
    rgb_resized = cv2.resize(rgb_rgb, (target_w, target_h))      # (360,640,3)
    rgb_input = rgb_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3,360,640)

    # ── 深度估计 (仅用于代价图) ──
    if de is None:
        de = DepthEstimator()
    depth_map = de.predict(rgb_bgr)                              # (H,W)

    # ── 构建代价地图 (仅用于可视化) ──
    if seg is None:
        seg = SemanticSegmentor()
    sem_img, _ = seg.predict(rgb_bgr)                            # (3,H,W), (H,W)
    cost_map, meta = build_costmap(depth_map, sem_img)
    if cost_map is None:
        print('[Warn] Cost map build failed, using dummy')
        cost_map = np.zeros((100, 100), dtype=np.float32)
        meta = {'x_start': -2.0, 'z_start': -2.0,
                'num_x': 100, 'num_z': 100, 'resolution': 0.04}

    aligner = CostMapAligner(meta)

    # ── 构造模型输入 tensor (SingleDINOv2Nav: 仅 RGB + goal) ──
    rgb_t = torch.from_numpy(rgb_input).float().unsqueeze(0).to(device)   # (1,3,360,640)
    goal_t = torch.from_numpy(goal).float().unsqueeze(0).to(device)       # (1,2)

    return rgb_t, goal_t, cost_map, aligner


def infer(model, rgb_t, goal_t, device):
    """
    模型推理 → 5 关键点 + 恐惧值

    Returns:
        keypoints: (5, 2) numpy — 5 个路径关键点
        fear: float — 碰撞概率
    """
    with torch.no_grad():
        keypoints, fear = model(rgb_t, goal_t)
    return keypoints.cpu().numpy()[0], fear.cpu().numpy()[0, 0]


def visualize(image_bgr, keypoints, goal, fear, cost_map, aligner, save_path):
    """
    3 面板可视化：
      左图：原始 RGB
      中图：预测轨迹（5 关键点，鸟瞰，x-z 平面）
      右图：代价地图 + 预测轨迹叠加
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x_min, x_max, z_min, z_max = aligner.get_grid_extent()
    cm_viz = cost_map

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── 左：RGB ──
    axes[0].imshow(image_rgb)
    axes[0].set_title(f'Input Image | Fear: {fear:.3f}')
    axes[0].axis('off')

    # ── 中：轨迹 ──
    axes[1].plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    axes[1].plot(keypoints[:, 0], keypoints[:, 1], 'b.-', linewidth=2, markersize=8, label='Pred')
    axes[1].plot(0, 0, 'k^', markersize=12, label='Robot')
    axes[1].set_xlabel('Right (x) [m]')
    axes[1].set_ylabel('Forward (z) [m]')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axis('equal')
    axes[1].set_title('Trajectory (5 keypoints, Bird\'s Eye)')

    # ── 右：代价图 + 轨迹 ──
    im = axes[2].imshow(cm_viz.T, origin='lower',
                        extent=[x_min, x_max, z_min, z_max],
                        cmap='hot', alpha=0.85)
    axes[2].plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')
    axes[2].plot(keypoints[:, 0], keypoints[:, 1], 'b.-', linewidth=2, markersize=6, label='Pred')
    axes[2].plot(0, 0, 'k^', markersize=10, label='Robot')
    axes[2].set_xlabel('Right (x) [m]')
    axes[2].set_ylabel('Forward (z) [m]')
    axes[2].legend(fontsize=8, loc='upper right')
    axes[2].set_title('Cost Map + Trajectory')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[Vis] Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='SingleDINOv2Nav 推理')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--goal_x', type=float, required=True, help='目标点 x (右方向, 米)')
    parser.add_argument('--goal_z', type=float, required=True, help='目标点 z (前方向, 米)')
    parser.add_argument('--ckpt',
                        default='results/last.pth',
                        help='模型权重路径')
    parser.add_argument('--config', default='config/single_dinov2.yaml',
                        help='模型配置文件')
    parser.add_argument('--out', default='infer_result.png', help='输出图片路径')
    parser.add_argument('--cpu', action='store_true', help='强制 CPU')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # ── 加载模型 ──
    cfg = load_config(args.config)
    model = load_model(args.ckpt, cfg, device)

    # ── 读取图片 ──
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f'[Error] Cannot read image: {args.image}')
        sys.exit(1)
    print(f'[Image] {args.image} ({image_bgr.shape[1]}x{image_bgr.shape[0]})')

    goal = np.array([args.goal_x, args.goal_z], dtype=np.float32)
    print(f'[Goal] x={args.goal_x:.2f}m, z={args.goal_z:.2f}m')

    # ── 预处理 → 模型推理 ──
    de = DepthEstimator()
    print('[Pipeline] Depth estimation + Cost map (visualization only)...')
    rgb_t, goal_t, cost_map, aligner = preprocess_for_model(
        image_bgr, goal, device, de=de,
    )

    print('[Pipeline] Model inference (SingleDINOv2Nav)...')
    keypoints, fear = infer(model, rgb_t, goal_t, device)

    print(f'[Result] Keypoints: {keypoints.tolist()}')
    print(f'[Result] Fear: {fear:.4f}')

    # ── 可视化 ──
    visualize(image_bgr, keypoints, goal, fear, cost_map, aligner, args.out)


if __name__ == '__main__':
    main()