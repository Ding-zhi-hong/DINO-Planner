"""
ViPlanner 训练循环
"""
import argparse
import os
import yaml
import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SingleDINOv2Nav
from loss import ViPlannerLoss
from data import create_dataloader


def visualize_val(model, loader, loss_fn, epoch, result_dir, device):
    """验证集可视化: 左RGB + 中轨迹 + 右代价图"""
    model.eval()
    vis_dir = os.path.join(result_dir, 'vis', f'epoch_{epoch:03d}')
    os.makedirs(vis_dir, exist_ok=True)

    vis_count = 0
    max_vis = 20

    for batch in loader:
        if vis_count >= max_vis:
            break

        sem = batch['sem'].to(device)
        depth = batch['depth'].to(device)
        goal = batch['goal'].to(device)
        cm_list = batch['cost_map']
        al_list = batch['aligner']
        rgb_list = batch.get('rgb')
        sid_list = batch.get('sid', ['?'] * len(sem))

        with torch.no_grad():
            kp_pred, fear = model(sem, goal)
            _, loss_dict = loss_fn(kp_pred, fear, goal, cm_list, al_list)

            # 每个样本独立的 obs
            per_sample_obs = []
            for b in range(len(sem)):
                cm_b = [cm_list[b]]
                al_b = [al_list[b]]
                kp_b = kp_pred[b:b+1]
                g_b = goal[b:b+1]
                _, ld_b = loss_fn(kp_b, fear[b:b+1], g_b, cm_b, al_b)
                per_sample_obs.append(ld_b['obs'])

        for b in range(len(sem)):
            if vis_count >= max_vis:
                break

            sid = sid_list[b] if sid_list else '?'
            kp = kp_pred[b].cpu().numpy()
            g = goal[b].cpu().numpy()
            rgb_img = rgb_list[b] if rgb_list else None

            cm = cm_list[b]
            al = al_list[b]
            x_min, x_max, z_min, z_max = al.get_grid_extent()

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            if rgb_img is not None:
                axes[0].imshow(rgb_img)
            else:
                axes[0].text(0.5, 0.5, 'No RGB', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title(f'{sid} | Fear: {fear[b].item():.3f}')
            axes[0].axis('off')

            sample_obs = per_sample_obs[b] if b < len(per_sample_obs) else loss_dict['obs']
            axes[1].plot(g[0], g[1], 'r*', markersize=15, label='Goal')
            axes[1].plot(kp[:, 0], kp[:, 1], 'b.-', linewidth=2, markersize=8, label='Pred')
            axes[1].plot(0, 0, 'k^', markersize=12, label='Robot')
            axes[1].set_xlabel('Right (x) [m]')
            axes[1].set_ylabel('Forward (z) [m]')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].axis('equal')
            axes[1].set_title(f'Trajectory | obs={sample_obs:.3f}')

            cm_viz = cm.cpu().numpy() if hasattr(cm, 'cpu') else np.array(cm)
            im = axes[2].imshow(cm_viz.T, origin='lower',
                                extent=[x_min, x_max, z_min, z_max],
                                cmap='hot', alpha=0.85)
            axes[2].plot(g[0], g[1], 'r*', markersize=12, label='Goal')
            axes[2].plot(kp[:, 0], kp[:, 1], 'b.-', linewidth=2, markersize=6, label='Pred')
            axes[2].plot(0, 0, 'k^', markersize=10, label='Robot')
            axes[2].set_xlabel('Right (x) [m]')
            axes[2].set_ylabel('Forward (z) [m]')
            axes[2].legend(fontsize=8, loc='upper right')
            axes[2].set_title('Cost Map')
            plt.colorbar(im, ax=axes[2], shrink=0.8)

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{sid}.png'), dpi=120, bbox_inches='tight')
            plt.close()
            vis_count += 1

    print(f'  Saved {vis_count} visualizations to {vis_dir}')
    model.train()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/single_dinov2.yaml')
    parser.add_argument('--data_dir', default='D:/carla/traindata')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    class Cfg:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Cfg(v))
                else:
                    setattr(self, k, v)
    cfg = Cfg(cfg_dict)
    if args.epochs:
        cfg.training.max_epochs = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(os.path.join(result_dir, 'checkpoints'), exist_ok=True)

    # 模型 (SingleDINOv2Nav — RGB 单输入)
    model = SingleDINOv2Nav(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Model: trainable={trainable:,} / frozen={total-trainable:,} / total={total:,} params')

    # 损失
    criterion = ViPlannerLoss(cfg)

    # 优化器 — 统一LR (同 ViPlanner)
    if cfg.training.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.lr,
                                     momentum=cfg.training.momentum,
                                     weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,
                                      weight_decay=cfg.training.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.training.lr_factor,
        patience=cfg.training.lr_patience, min_lr=cfg.training.min_lr
    )

    # 数据
    train_loader = create_dataloader(args.data_dir, 'train', batch_size=cfg.training.batch_size, viz=False)
    val_loader = create_dataloader(args.data_dir, 'val', batch_size=cfg.training.batch_size, viz=True)

    # 断点续训
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'Resumed from epoch {ckpt["epoch"]}')

    # ═══════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg.training.max_epochs):
        model.train()
        t0 = time.time()
        train_loss, n = 0.0, 0

        pbar = tqdm(train_loader, desc=f'E{epoch:03d}')
        for batch in pbar:
            sem = batch['sem'].to(device)
            depth = batch['depth'].to(device)
            goal = batch['goal'].to(device)
            cost_maps = batch['cost_map']
            aligners = batch['aligner']

            # SingleDINOv2Nav: 仅需 RGB + goal, 不需要 depth
            kp_pred, fear = model(sem, goal)
            loss, ld = criterion(kp_pred, fear, goal, cost_maps, aligners)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            n += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'goal': f'{ld["goal"]:.3f}',
                'obs': f'{ld["obs"]:.3f}',
                'mot': f'{ld["motion"]:.3f}',
            })

        train_loss /= n

        # 验证
        model.eval()
        val_loss, vn = 0.0, 0
        val_ld = {'goal': 0, 'obs': 0, 'motion': 0, 'fear': 0}
        for batch in val_loader:
            sem = batch['sem'].to(device)
            depth = batch['depth'].to(device)
            goal = batch['goal'].to(device)
            cost_maps = batch['cost_map']
            aligners = batch['aligner']

            with torch.no_grad():
                kp_pred, fear = model(sem, goal)
                loss, ld = criterion(kp_pred, fear, goal, cost_maps, aligners)

            val_loss += loss.item()
            for k in val_ld:
                val_ld[k] += ld[k]
            vn += 1

        val_loss /= vn
        for k in val_ld:
            val_ld[k] /= vn

        scheduler.step(val_loss)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(ckpt, os.path.join(result_dir, 'checkpoints', 'last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(result_dir, 'checkpoints', 'best.pth'))

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        print(f'E{epoch:03d} | '
              f'train={train_loss:.4f} val={val_loss:.4f} | '
              f'goal={val_ld["goal"]:.4f} '
              f'obs={val_ld["obs"]:.4f} mot={val_ld["motion"]:.4f} '
              f'fear={val_ld["fear"]:.4f} | '
              f'lr={lr:.2e} {dt:.0f}s | best={best_loss:.4f}')

        # 可视化
        visualize_val(model, val_loader, criterion, epoch, result_dir, device)

    print(f'Training done. Best val loss: {best_loss:.4f}')


if __name__ == '__main__':
    train()