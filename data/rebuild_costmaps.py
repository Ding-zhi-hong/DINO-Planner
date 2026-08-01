"""
重新构建所有样本的代价地图

使用修改后的 cost_map_builder (FOV 外部=0.0, 而非 0.2)
覆盖每个样本原有的 cost_map.npy / cost_meta.npy

用法:
  cd mymodel
  python -m data.rebuild_costmaps
"""
import os, glob, time, sys
import numpy as np
from tqdm import tqdm

# 确保能找到 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cost_map_builder import build_costmap

DATA_DIR = r'D:/carla/traindata'
BATCH_SIZE = 64          # 每批处理数量，控制内存


def rebuild_one(sample_dir):
    """为单个样本重建代价地图"""
    depth_path = os.path.join(sample_dir, 'depth.npy')
    sem_path   = os.path.join(sample_dir, 'sem.npy')

    depth = np.load(depth_path).squeeze()          # (360, 640) float32 米
    sem   = np.load(sem_path)                       # (3, 360, 640) float32 [0,1]
    sem_u8 = (sem * 255).astype(np.uint8)           # → (3, 360, 640) uint8

    cost_map, meta = build_costmap(depth, sem_u8)

    if cost_map is not None:
        np.save(os.path.join(sample_dir, 'cost_map.npy'), cost_map)
        np.save(os.path.join(sample_dir, 'cost_meta.npy'), meta)
        return True, cost_map.shape
    else:
        return False, None


def main():
    samples = sorted(glob.glob(os.path.join(DATA_DIR, 'sample_*')))
    print(f'共找到 {len(samples)} 个样本')

    ok = 0
    fail = 0
    shapes = {}

    for s in tqdm(samples, desc='重建代价地图', unit='样本'):
        sid = os.path.basename(s)
        success, shape = rebuild_one(s)
        if success:
            ok += 1
            k = str(shape)
            shapes[k] = shapes.get(k, 0) + 1
        else:
            fail += 1
            tqdm.write(f'  ⚠  失败: {sid} (点云点太少)')

    print('=' * 50)
    print(f'完成! 成功: {ok}, 失败: {fail}')
    if shapes:
        for k, v in sorted(shapes.items()):
            print(f'  尺寸 {k}: {v} 个样本')
    print('=' * 50)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'总耗时: {time.time() - t0:.1f} 秒')