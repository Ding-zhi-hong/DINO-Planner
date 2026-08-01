"""
SingleDINOv2Nav — 单个冻结 DINOv2 + 完整 DPT 多尺度融合

核心思路:
  1. 冻结 DINOv2 (从本地 checkpoint 加载), 提取 4 层中间层 patch tokens
  2. 完整 DPT 结构 (Conv1×1 + Resize + RefineNet ×4) → 1024×12×20
  3. 原始 Decoder 完全复用 (输入 1024×12×20)

对比原始:
  原始: DINOv2(最后层) → 512 + PlannerNet(深度) → 512 → Concat → 1024×12×20
  新:   DINOv2(4中间层) → 完整 DPT 融合 → 1024×12×20          (无需深度图!)

输入: RGB (B,3,360,640) + Goal (B,2)
输出: keypoints (B,5,2) + fear (B,1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# ─────────────────────────────────────────────
# DINOv2 本地加载路径
# ─────────────────────────────────────────────
DINOV2_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'checkpoints', 'dinov2_vitb14_pretrain.pth'
)


# ═════════════════════════════════════════════
# 1. 完整 DPT 基础组件
# ═════════════════════════════════════════════

class ResidualConvUnit(nn.Module):
    """DPT RefineNet 中的残差卷积单元"""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x  # 残差


class FeatureFusionBlock(nn.Module):
    """DPT RefineNet 融合块 — 从深到浅逐步融合"""
    def __init__(self, features):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x, residual=None, size=None):
        """
        x:        当前层的特征 (深层)
        residual: 上层的特征 (浅层, 更高分辨率)
        size:     目标分辨率
        """
        x = self.resConfUnit1(x)

        if residual is not None:
            residual = self.resConfUnit2(residual)
            # 残差上采样到当前层分辨率
            residual = F.interpolate(residual, size=x.shape[2:],
                                     mode='bilinear', align_corners=True)
            x = x + residual

        if size is not None:
            x = F.interpolate(x, size=size,
                              mode='bilinear', align_corners=True)
        return x


# ═════════════════════════════════════════════
# 2. DINOv2 多尺度编码器 (冻结, 本地权重)
# ═════════════════════════════════════════════

class DINOv2MultiScaleEncoder(nn.Module):
    """
    仿 DepthAnythingV2: 从 DINOv2 提取 4 层中间层特征

    从本地 checkpoint 加载, 冻结。
    输入: (B, 3, H, W) RGB, [0,1]
    输出: list of 4 (patch_tokens, cls_token)
    """
    def __init__(self, model_size='vitb', checkpoint_path=None):
        super().__init__()

        if checkpoint_path is None:
            checkpoint_path = DINOV2_CKPT

        # 中间层索引 (与 DepthAnything 完全一致)
        # model_size 已经是 'vitb'/'vits' 等，直接做 key
        self.intermediate_layers = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }[model_size]

        self.embed_dim = {
            'vits': 384, 'vitb': 768, 'vitl': 1024, 'vitg': 1536,
        }[model_size]

        # ── 构建 DINOv2 模型结构 ──
        # 直接从模块导入，不经过 torch.hub
        import sys
        import importlib

        # 尝试从本地 dinov2 包加载
        # 如果 torch.hub 有缓存也可以用, 但用户指定本地权重
        # 先用 torch.hub 构建模型结构, 然后加载本地权重
        print(f'[DINOv2] Loading model structure from local torch.hub cache...')
        # source='local' + 指定缓存目录路径, 避免 GitHub 网络连接
        hub_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub')
        dinov2_cache = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
        self.backbone = torch.hub.load(dinov2_cache,
                                       f'dinov2_{model_size}14',
                                       source='local',
                                       skip_validation=True,
                                       trust_repo=True)

        # ── 加载本地权重 ──
        if os.path.exists(checkpoint_path):
            print(f'[DINOv2] Loading local weights from {checkpoint_path}')
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # 去掉可能的前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    k = k[9:]
                elif k.startswith('model.'):
                    k = k[6:]
                new_state_dict[k] = v
            missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f'  [DINOv2] missing keys: {len(missing)}')
            if unexpected:
                print(f'  [DINOv2] unexpected keys: {len(unexpected)}')
        else:
            print(f'[DINOv2] WARNING: checkpoint not found at {checkpoint_path}, using random init')

        # ── 冻结 ──
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        total = sum(p.numel() for p in self.backbone.parameters())
        print(f'[DINOv2] {model_size} | embed_dim={self.embed_dim} | '
              f'intermediate_layers={self.intermediate_layers} | '
              f'frozen: {total:,} params')

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB, [0,1]
        returns: list of (patch_tokens, cls_token), len=4
        """
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layers,
            return_class_token=True,
        )
        return features


# ═════════════════════════════════════════════
# 3. 完整 DPT 融合头 (带 RefineNet)
# ═════════════════════════════════════════════

class FullDPTHead(nn.Module):
    """
    完整 DPT 头 (与 DepthAnythingV2 一致)

    4 层 DINOv2 中间特征:
      → Conv1×1 投影到不同通道
      → 重采样到不同分辨率
      → CLS token 注入
      → RefineNet ×4 逐层融合 (从深到浅)
      → 上采样到 1024×12×20

    输入: list of 4 (patch_tokens, cls_token)
    输出: (B, 1024, 12, 20)
    """
    def __init__(self, embed_dim=768, features=64):
        super().__init__()

        # 4 层的投影通道 (跟 DepthAnything 一样)
        out_channels = [features, features*2, features*4, features*4]
        # = [256, 512, 1024, 1024]

        # ── Conv1×1 投影 ──
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        # ── 重采样层 ──
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        # ── CLS token 注入 (use_clstoken 模式) ──
        self.readout_projects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        # ── scratch: 统一通道到 features ──
        self.scratch_layer1_rn = nn.Conv2d(out_channels[0], features, 1)
        self.scratch_layer2_rn = nn.Conv2d(out_channels[1], features, 1)
        self.scratch_layer3_rn = nn.Conv2d(out_channels[2], features, 1)
        self.scratch_layer4_rn = nn.Conv2d(out_channels[3], features, 1)

        # ── RefineNet ×4 渐进融合 (从深到浅) ──
        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        # ── 最终输出: 输出 1024 通道以兼容 Decoder ──
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 1024, kernel_size=1),  # (B,1024,H,W)
        )

        # ── 固定上采样到 (12,20) ──
        self.to_12x20 = nn.Upsample(size=(12, 20), mode='bilinear',
                                    align_corners=False)

    def forward(self, features, patch_h, patch_w):
        """
        features: list of 4 (patch_tokens, cls_token)
        patch_h, patch_w: DINOv2 特征图尺寸 (≈26, 46)
        """
        outs = []

        for i, (patch_tokens, cls_token) in enumerate(features):
            B, N, D = patch_tokens.shape

            # ── CLS token 注入 ──
            cls_expanded = cls_token.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([patch_tokens, cls_expanded], dim=-1)  # (B,N,2D)
            x = self.readout_projects[i](x)                      # (B,N,D)

            # ── 重排为 2D ──
            x = x.permute(0, 2, 1)                               # (B,D,N)
            x = x.reshape(B, D, patch_h, patch_w)                 # (B,D,H_p,W_p)

            # ── 投影 + 重采样 ──
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            outs.append(x)

        # ── 逐层渐进融合 (从深到浅) ──
        # layer_4_rn (最深层, 小分辨率)
        layer_1_rn = self.scratch_layer1_rn(outs[0])
        layer_2_rn = self.scratch_layer2_rn(outs[1])
        layer_3_rn = self.scratch_layer3_rn(outs[2])
        layer_4_rn = self.scratch_layer4_rn(outs[3])

        # RefineNet 从深到浅融合 (lane_4 是最深层)
        path_4 = self.refinenet4(layer_4_rn,
                                 size=layer_3_rn.shape[2:])  # 上采样到 layer_3 尺寸
        path_3 = self.refinenet3(path_4, layer_3_rn,
                                 size=layer_2_rn.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2_rn,
                                 size=layer_1_rn.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1_rn)          # 保持原尺寸

        # ── 输出投影 → 1024ch → 12×20 ──
        out = self.output_conv(path_1)                         # (B,1024,H,W)
        out = self.to_12x20(out)                                # (B,1024,12,20)

        return out


# ═════════════════════════════════════════════
# 4. 原始 Decoder (完全复用, 不改一行)
# ═════════════════════════════════════════════

class Decoder(nn.Module):
    """原始 ViPlanner 解码器 — 完全复用"""
    def __init__(self, in_channels=1024, goal_channels=16, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fg = nn.Linear(2, goal_channels)

        self.conv1 = nn.Conv2d(in_channels + goal_channels, 512,
                               kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 128, 1024)   # 256 × 8 × 16 = 32768 → 不对
        # 实际: Conv2(512→256, k=3, p=0) 输入 12×20 → 输出 (12-2=10, 20-2=18)?
        # 等一下 Conv1(k=5,p=1): (12+2-5)/1+1 = 10, (20+2-5)/1+1 = 18
        # Conv2(k=3,p=0): (10+0-3)/1+1 = 8, (18+0-3)/1+1 = 16
        # 256 × 8 × 16 = 32768
        # 嗯, 256*128=32768, 所以 256*8*16 不对...让我算一下
        # 实际上: 12x20 输入
        # conv1:  (12-5+2)/1+1 = 10, (20-5+2)/1+1 = 18? 不对
        # 公式: out = (in + 2*padding - kernel)/stride + 1
        # conv1: (12 + 2 - 5)/1 + 1 = 10, (20 + 2 - 5)/1 + 1 = 18
        # conv2: (10 + 0 - 3)/1 + 1 = 8, (18 + 0 - 3)/1 + 1 = 16
        # 256 * 8 * 16 = 32768... 但注释写着 256*128=32768
        # 8*16=128, 所以 256*128=32768 ✅
        # 所以 fc1 是 nn.Linear(256*128=32768, 1024)
        self.fc1 = nn.Linear(32768, 1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 2)

        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        goal_emb = self.fg(goal[:, 0:2])
        goal_emb = goal_emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])

        x = torch.cat((x, goal_emb), dim=1)       # (B, 1024+16=1040, 12, 20)
        x = self.relu(self.conv1(x))               # (B, 512, 10, 18)
        x = self.relu(self.conv2(x))               # (B, 256, 8, 16)

        x = torch.flatten(x, 1)                    # (B, 32768)
        f = self.relu(self.fc1(x))                 # (B, 1024)

        kp = self.relu(self.fc2(f))                # (B, 512)
        kp = self.fc3(kp)                          # (B, 10)
        keypoints = kp.reshape(-1, self.k, 2).clamp(-10, 10)

        c = self.relu(self.frc1(f))                # (B, 128)
        fear = self.sigmoid(self.frc2(c))          # (B, 1)

        return keypoints, fear


# ═════════════════════════════════════════════
# 5. 主模型: SingleDINOv2Nav
# ═════════════════════════════════════════════

class SingleDINOv2Nav(nn.Module):
    """
    单 DINOv2 导航模型

    输入 RGB → 输出轨迹 (无需深度图):
      DINOv2(冻结,4中间层) → 完整DPT融合 → 1024×12×20 → Decoder → 5关键点 + 恐惧
    """
    def __init__(self, cfg, dinov2_checkpoint=None):
        super().__init__()

        self.knodes = cfg.model.knodes
        self.goal_channels = cfg.model.in_channel

        # 1. DINOv2 多尺度编码器 (冻结)
        self.encoder = DINOv2MultiScaleEncoder(
            model_size='vitb',
            checkpoint_path=dinov2_checkpoint,
        )

        # 2. 完整 DPT 融合头 (训练)
        self.dpt_head = FullDPTHead(
            embed_dim=self.encoder.embed_dim,
            features=64,
        )

        # 3. 原始 Decoder (完全复用, 不改)
        self.decoder = Decoder(
            in_channels=1024,
            goal_channels=self.goal_channels,
            k=self.knodes,
        )

        # 统计参数量
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f'[SingleDINOv2Nav] trainable={trainable:,} | '
              f'frozen={frozen:,} | total={total:,}')

    def forward(self, rgb, goal):
        """
        rgb:  (B, 3, 360, 640) RGB [0,1]
        goal: (B, 2) 目标点 (x_右, z_前)

        returns: keypoints (B,5,2), fear (B,1)
        """
        B, _, H, W = rgb.shape

        # ── Pad 到 14 的倍数 (DINOv2 patch_embed 要求) ──
        patch_size = 14
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode='reflect')

        H_p = (H + pad_h) // patch_size  # 364/14 = 26
        W_p = (W + pad_w) // patch_size  # 644/14 = 46

        # ── Step 1: DINOv2 多尺度编码 (冻结) ──
        features = self.encoder(rgb)
        # features: 4 × [(B, N_patches, 768), (B, 768)]

        # ── Step 2: 完整 DPT 融合 → 1024×12×20 ──
        fused = self.dpt_head(features, H_p, W_p)  # (B, 1024, 12, 20)

        # ── Step 3: 原始 Decoder → 轨迹 + 恐惧 ──
        keypoints, fear = self.decoder(fused, goal)

        return keypoints, fear