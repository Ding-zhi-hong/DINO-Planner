"""
批量重建代价图 — 用新方法覆盖 traindata 中的 cost_map.npy / cost_meta.npy

用法:
  python scripts/rebuild_costmaps.py                     # 全量 3853 样本
  python scripts/rebuild_costmaps.py --num 100            # 前 100 个测试
  python scripts/rebuild_costmaps.py --workers 4          # 4 进程并行
  python scripts/rebuild_costmaps.py --dry-run            # 只统计不执行
"""

import os, sys
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

# 添加项目根目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

TRAINDATA_DIR = r'D:\carla\traindata'
OUT_FILES = ['cost_map.npy', 'cost_meta.npy']


def rebuild_one(sample_dir):
    """单个样本: 加载 sem + depth → 新方法 build_costmap → 覆盖保存"""
    sid = os.path.basename(sample_dir)

    try:
        # ── 加载预计算数据 ──
        sem_path = os.path.join(sample_dir, 'sem.npy')
        depth_path = os.path.join(sample_dir, 'depth.npy')
        if not os.path.exists(sem_path) or not os.path.exists(depth_path):
            return f'{sid}: missing sem/depth'

        sem = np.load(sem_path)            # (3, 360, 640) float32 [0,1]
        depth = np.load(depth_path)        # (1, 360, 640) float32

        # ── 恢复语义图为 uint8 ──
        sem_u8 = np.round(sem * 255).astype(np.uint8)

        # ── 深度 squeeze ──
        depth_map = depth[0]               # (360, 640)

        # ── 新方法构建代价图 ──
        from utils.cost_map_builder import build_costmap, get_pixel_costs

        cost_override = get_pixel_costs(sem_u8)  # (360, 640) float32
        cm, meta = build_costmap(
            depth_map, sem_u8,
            cost_override=cost_override,
            fov_deg=70,
        )

        if cm is None:
            return f'{sid}: cost map None'

        # ── 覆盖保存 ──
        np.save(os.path.join(sample_dir, 'cost_map.npy'), cm)
        np.save(os.path.join(sample_dir, 'cost_meta.npy'), meta)

    except Exception as e:
        return f'{sid}: {e}'

    return None  # None = OK


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量重建代价图')
    parser.add_argument('--data_dir', default=TRAINDATA_DIR)
    parser.add_argument('--num', type=int, default=None, help='处理样本数')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--workers', type=int, default=2, help='并行进程数')
    parser.add_argument('--dry-run', action='store_true', help='只统计不执行')
    args = parser.parse_args()

    samples = sorted(glob.glob(os.path.join(args.data_dir, 'sample_*')))
    samples = samples[args.start:]
    if args.num:
        samples = samples[:args.num]

    print(f"数据集: {args.data_dir}")
    print(f"总样本: {len(samples)}, 并行: {args.workers}")

    # 检查哪些已经有 cost_map
    has_old = sum(1 for s in samples if os.path.exists(os.path.join(s, 'cost_map.npy')))
    print(f"已有 cost_map: {has_old}/{len(samples)} (将被覆盖)")

    if args.dry_run:
        print("Dry-run: 未执行任何操作")
        return

    t0 = __import__('time').time()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(rebuild_one, samples),
            total=len(samples),
            desc="Rebuilding cost maps",
        ))

    ok = sum(1 for r in results if r is None)
    fails = [r for r in results if r is not None]
    elapsed = __import__('time').time() - t0

    print(f"\n✅ 完成: {ok} OK, ❌ {len(fails)} failed / {ok + len(fails)} total")
    print(f"⏱ 耗时: {elapsed:.0f}s ({ok/elapsed:.0f} samples/s)")
    if fails:
        print("❌ 错误列表 (前20):")
        for f in fails[:20]:
            print(f"  {f}")


if __name__ == '__main__':
    main()