"""
DINOv2Encoder — 冻结的 DINOv2 骨干 + 适配器

输入:  (B, 3, 360, 640) RGB
输出:  (B, feat_dim, 12, 20) 空间特征图

加载官方 DINOv2 预训练权重 (torch hub cache).
冻结 backbone, 只训练 adapter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 注册 DINOv2 hub 路径 ──
_DINOV2_HUB = r'C:\Users\34287\.cache\torch\hub\facebookresearch_dinov2_main'


def _load_dinov2(model_size='vitb'):
    """
    从本地 torch hub cache 加载 DINOv2 模型.
    """
    # 确保 hub 能发现本地 repo
    import sys
    if _DINOV2_HUB not in sys.path:
        sys.path.insert(0, _DINOV2_HUB)

    if model_size == 'vitb':
        from dinov2.hub.backbones import dinov2_vitb14
        return dinov2_vitb14()
    elif model_size == 'vits':
        from dinov2.hub.backbones import dinov2_vits14
        return dinov2_vits14()
    elif model_size == 'vitl':
        from dinov2.hub.backbones import dinov2_vitl14
        return dinov2_vitl14()
    elif model_size == 'vitg':
        from dinov2.hub.backbones import dinov2_vitg14
        return dinov2_vitg14()
    else:
        raise ValueError(f"Unknown model_size: {model_size}")


class DINOv2Encoder(nn.Module):
    """
    DINOv2 冻结编码器 + 适配器头

    Args:
        feat_dim:   输出通道数 (默认 512)
        model_size: 模型大小 'vits' | 'vitb' | 'vitl' | 'vitg' (默认 'vitb')
    """
    def __init__(self, feat_dim=512, model_size='vitb'):
        super().__init__()

        # ── 加载 DINOv2 骨干 ──
        print(f'[DINOv2Encoder] Loading dinov2_{model_size}14 from hub ...')
        self.backbone = _load_dinov2(model_size)
        embed_dim = self.backbone.embed_dim  # vitb → 768, vits → 384, vitl → 1024, vitg → 1536

        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # ── 适配器: 投影 + 上采样 ──
        # DINOv2 输出 cls token + patch tokens (N+1, embed_dim)
        # Patch 数: (360/14) * (640/14) ≈ 26 * 46 ≈ 1196
        # 适配器: LayerNorm → Conv1x1 投影 → 重排 → 插值到 12x20
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, feat_dim, kernel_size=1)

        print(f'[DINOv2Encoder] embed_dim={embed_dim}, feat_dim={feat_dim}, frozen={sum(p.numel() for p in self.backbone.parameters()):,} params')

    def forward(self, x):
        """
        x: (B, 3, 360, 640) RGB, [0, 1]
        returns: (B, feat_dim, 12, 20)
        """
        B = x.shape[0]
        _, _, H, W = x.shape

        # ── 填充到 14 的倍数 ──
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        Hp_pad = (H + pad_h) // 14  # 360→364→26 patches
        Wp_pad = (W + pad_w) // 14  # 640→644→46 patches

        # ── DINOv2 提取 patch tokens ──
        with torch.no_grad():
            out = self.backbone.forward_features(x)
            patch_tokens = out['x_norm_patchtokens']  # (B, N, embed_dim)

        # ── LayerNorm ──
        patch_tokens = self.norm(patch_tokens)

        # ── 重排为 2D 特征图 ──
        feat = patch_tokens[:, :Hp_pad * Wp_pad]
        feat = feat.permute(0, 2, 1).reshape(B, -1, Hp_pad, Wp_pad)

        # ── 投影到 feat_dim ──
        feat = self.proj(feat)

        # ── 插值到 (12, 20) ──
        feat = F.interpolate(feat, size=(12, 20), mode='bilinear', align_corners=False)

        return feat  # (B, feat_dim, 12, 20)