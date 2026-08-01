"""
CARLA 数据集 — 加载原始 CityWalker 数据

每个样本目录 (D:/carla/traindata/sample_trajectory_*):
  000004.jpg     原始 RGB (3,360,640)   ← 现在作为模型输入
  sem.npy        (3,360,640)  语义图    ← 不再使用 (保留供以后参考)
  depth.npy      (1,360,640)  深度图 (float32, 米)
  cost_map.npy   (nx,nz)      代价图
  cost_meta.npy  dict         代价图参数
  goal.txt       一行两列     目标点 (x_right, z_forward)
"""
import os, glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils.cost_map_align import CostMapAligner


class CARLADataset(Dataset):
    def __init__(self, data_dir, mode='train', val_ratio=0.1, viz=False):
        """
        data_dir: D:/carla/data
        mode: train / val
        val_ratio: 验证集比例
        viz: 可视化模式 (额外加载 RGB 副本)
        """
        self.mode = mode
        self.viz = viz
        self.samples = sorted(glob.glob(os.path.join(data_dir, 'sample_*')))
        assert len(self.samples) > 0, f'No samples found in {data_dir}'

        # 按模式分割
        random.seed(42)
        random.shuffle(self.samples)
        split = int(len(self.samples) * (1 - val_ratio))

        if mode == 'train':
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]

        print(f'{mode}: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        sid = os.path.basename(sample_dir)

        # ── 加载 RGB 原图 (模型输入) ──
        jpg_path = os.path.join(sample_dir, '000004.jpg')
        rgb_uint8 = cv2.imread(jpg_path)
        rgb_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_BGR2RGB)          # (H, W, 3) uint8
        if rgb_uint8.shape[0] != 360 or rgb_uint8.shape[1] != 640:
            rgb_uint8 = cv2.resize(rgb_uint8, (640, 360))               # resize to 360×640
        rgb = rgb_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0   # (3, 360, 640) float32

        # ── 加载深度图 ──
        depth = np.load(os.path.join(sample_dir, 'depth.npy'))
        if depth.shape[1] != 360 or depth.shape[2] != 640:
            H, W = 360, 640
            depth_resized = np.zeros((1, H, W), dtype=np.float32)
            depth_resized[0] = cv2.resize(depth[0], (W, H), interpolation=cv2.INTER_NEAREST)
            depth = depth_resized

        # ── 代价图 ──
        cost_map = np.load(os.path.join(sample_dir, 'cost_map.npy'))
        cost_meta = np.load(os.path.join(sample_dir, 'cost_meta.npy'),
                            allow_pickle=True).item()

        # ── 目标点 (goal.txt: x_right z_forward) ──
        goal_path = os.path.join(sample_dir, 'goal.txt')
        if os.path.exists(goal_path):
            goal = np.loadtxt(goal_path, dtype=np.float32)
        else:
            with open(os.path.join(sample_dir, 'input_data.txt')) as f:
                lines = f.readlines()
            goal_line = lines[5].strip().split()
            goal = np.array([float(goal_line[1]), float(goal_line[0])], dtype=np.float32)

        # ── 对齐器 ──
        aligner = CostMapAligner(cost_meta)

        # ── 输入标准化 ──
        depth_norm = depth / 20.0

        # ── 注意: key 名仍为 'sem' 以兼容 train.py ──
        sample = {
            'sem': torch.from_numpy(rgb).float(),           # (3, 360, 640), [0, 1]
            'depth': torch.from_numpy(depth_norm).float(),
            'goal': torch.from_numpy(goal).float(),
            'cost_map': torch.from_numpy(cost_map).float(),
            'aligner': aligner,
        }

        if self.viz:
            sample['rgb'] = rgb_uint8                           # (360, 640, 3) uint8
            sample['sid'] = sid

        return sample


def custom_collate(batch):
    """batch_size>1: tensor stack, cost_map/aligner 保持 list"""
    out = {}
    for k in batch[0].keys():
        if k in ('cost_map', 'aligner', 'rgb', 'sid'):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def create_dataloader(data_dir, mode='train', batch_size=16, viz=False):
    dataset = CARLADataset(data_dir, mode, viz=viz)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=2,
        collate_fn=custom_collate,
    )