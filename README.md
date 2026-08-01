# SingleDINOv2Nav — 单 DINOv2 端到端导航模型
    2 
    3 基于 **ViPlanner** 思想、使用**单个冻结 DINOv2** 骨干的端到端局部导航模型。输入一张 **RGB 图像 + 一个目标点**，直接输出 **5 个路径关键点 + 碰撞概率（fear）**，**无
      需深度图输入**。
    4
    5 本项目的核心改动（对比原始 ViPlanner）在于：用 **冻结的 DINOv2 多尺度特征 + 完整 DPT（RefineNet）融合**替换原来的「DINOv2 最后一层 + PlannerNet 深度编码器」双编码器
      结构，从而完全去掉深度图依赖，仅凭 RGB 即可完成规划。
    6
    7 > ⚠️ 项目目录名中的 “fov置0” 表示代价图/相机 **FOV 相关的感知管线参数被置 0/简化处理**，代价图仅用于训练时的损失计算与可视化，不参与模型前向输入。
    8
    9 ---
   10
   11 ## 模型结构
   12
   13 ```
   14 输入 RGB (B,3,360,640)                    ── 唯一视觉输入
   15         │
   16         ▼
   17 DINOv2(vitb14) 冻结 ═ 提取 4 层中间特征   ── 层索引 [2,5,8,11]
   18         │
   19         ▼
   20 完整 DPT 融合头 (Conv1×1 + Resize + RefineNet×4)
   21         │  ← CLS token 注入 (use_clstoken 模式)
   22         ▼
   23 1024 × 12 × 20 空间特征图
   24         │
   25         ▼
   26 原始 ViPlanner Decoder (复用, 未改动)
   27         ├── 5 个路径关键点 keypoints (B,5,2)
   28         └── 碰撞概率 fear (B,1)
   29 ```
   30
   31 ### 对比原始模型
   32
   33 | | 原始 ViPlanner | 本项目 (SingleDINOv2Nav) |
   34 |---|---|---|
   35 | 视觉编码 | DINOv2(最后层)→512 + PlannerNet(深度)→512 | DINOv2(4个中间层)→完整 DPT 融合 |
   36 | 深度图 | 需要 | **不需要**（仅 RGB） |
   37 | 特征输出 | 512+512 → Concat → 1024×12×20 | DPT → 1024×12×20 |
   38 | Decoder | 原始 | 完全复用 |
   39
   40 ### 关键组件 (`model/`)
   41
   42 - `dinov2_encoder.py` — 冻结 DINOv2 骨干 + 适配器（从本地 torch hub cache 加载）
   43 - `encoder.py` — 随机初始化的 ResNet34 风格编码器（参照 PlannerNet，备用）
   44 - `single_dinov2_nav.py` — 主模型：`DINOv2MultiScaleEncoder` + `FullDPTHead` + 原始 `Decoder`
   45
   46 ---
   47
   48 ## 项目结构
   49
   50 ```
   51 .
   52 ├── config/
   53 │   └── single_dinov2.yaml      # 模型 / 损失 / 训练超参数
   54 ├── data/
   55 │   ├── dataset.py              # CARLA 数据集加载 (RGB + goal + cost_map)
   56 │   ├── preprocess*.py          # 离线预处理 (语义/深度/测试数据)
   57 │   ├── rebuild_costmaps.py     # 代价图重建
   58 │   └── verify_coords.py        # 坐标校验
   59 ├── model/
   60 │   ├── single_dinov2_nav.py    # 主模型 (DINOv2 + DPT + Decoder)
   61 │   ├── dinov2_encoder.py       # 冻结 DINOv2 编码器
   62 │   └── encoder.py              # ResNet 风格编码器 (备用)
   63 ├── utils/
   64 │   ├── depth.py                # DepthAnythingV2 深度估计
   65 │   ├── semantic.py             # Mask2Former 语义分割
   66 │   ├── cost_map_builder.py     # 代价图构建 (FOV 相关)
   67 │   └── cost_map_align.py       # 代价图坐标对齐器
   68 ├── scripts/
   69 │   └── rebuild_costmaps.py     # 批量重建代价图 (多进程)
   70 ├── evaluation/
   71 │   ├── evaluate_p0.py          # 论文 P0 核心指标评测
   72 │   └── results_p0*.json        # 评测结果
   73 ├── train.py                    # 训练 + 验证可视化
   74 ├── infer.py                    # 单图推理 (RGB + 轨迹 + 代价图 3 面板)
   75 ├── predict.py                  # 纯推理 (仅输出 5 关键点, 最小依赖)
   76 ├── batch_infer.py              # 批量推理
   77 ├── benchmark.py                # 模型原始推理速度基准
   78 ├── benchmark_speed.py          # 端到端推理速度基准
   79 ├── loss.py                     # ViPlanner 损失 (Goal+Obs+Motion+Fear)
   80 ├── checkpoints/
   81 │   └── dinov2_vitb14_pretrain.pth   # DINOv2 预训练权重 (冻结)
   82 ├── results/                    # 训练输出 (checkpoints / 可视化)
   83 └── Depth-Anything-V2/          # 深度估计依赖子模块
   84 ```
   85
   86 ---
   87
   88 ## 损失函数 (`loss.py`)
   89
   90 完全参照 ViPlanner `TrajCost.CostofTraj`，四项加权求和：
   91
   92 ```python
   93 Loss = w_goal·Goal + w_obs·Obstacle + w_motion·Motion + w_fear·Fear
   94 ```
   95
   96 - **Goal Loss**：`log(‖wp[-1] - goal‖ + 1)` —— 终点误差
   97 - **Obstacle Loss**：3 轨扩展（机器人半宽 ±0.3m）在代价图上采样求和，道路代价减 0.5 置 0
   98 - **Motion Loss**：Hermite 插值后 51 个路径点的实际步长 vs 理想步长差
   99 - **Fear Loss**：前 `fear_ahead_dist` 米内最大代价超阈值则标记碰撞，BCE 监督
  100
  101 5 个关键点经 **Hermite 插值**得到 51 个路径点（同 ViPlanner `TrajGeneratorFromPFreeRot`）。
  102
  103 ---
  104
  105 ## 快速开始
  106
  107 ### 环境要求
  108
  109 - Python 3.10+
  110 - PyTorch + CUDA
  111 - 依赖：`tqdm`, `numpy`, `opencv-python`, `matplotlib`, `pyyaml`, `torchvision`
  112 - DINOv2 预训练权重（放入 `checkpoints/dinov2_vitb14_pretrain.pth`，也可通过 torch hub cache 自动加载）
  113
  114 ### 1. 训练
  115
  116 ```bash
  117 python train.py                                # 使用默认配置
  118 python train.py --config config/single_dinov2.yaml
  119 python train.py --data_dir D:/carla/carla_data
  120 python train.py --epochs 20                    # 覆盖 max_epochs
  121 python train.py --resume results/single_dinov2_run/checkpoints/last.pth   # 断点续训
  122 ```
  123
  124 训练过程会保存 `best.pth` / `last.pth`，并对验证集生成 3 面板可视化（RGB / 轨迹 / 代价图）。
  125
  126 ### 2. 推理
  127
  128 ```bash
  129 # 完整推理: RGB + 目标 → 3 面板可视化 (RGB, 轨迹, 代价图)
  130 python infer.py --image path/to/image.jpg --goal_x 2.0 --goal_z 5.0 \
  131                 --ckpt results/single_dinov2_run/checkpoints/best.pth
  132
  133 # 纯推理: 最小依赖, 仅输出 5 个关键点 (适合集成到导航管线)
  134 python predict.py --image path/to/image.jpg --goal_x 2.0 --goal_z 5.0 \
  135                   --ckpt results/single_dinov2_run/checkpoints/best.pth
  136
  137 # 批量推理
  138 python batch_infer.py --num 500 --out_dir batch_infer_results
  139 ```
  140
  141 坐标约定：`goal_x` 为**右方向 (x)**，`goal_z` 为**前方向 (z)**，单位米。
  142
  143 ### 3. 基准测试
  144
  145 ```bash
  146 python benchmark.py         # 模型原始 5 关键点输出速度
  147 python benchmark_speed.py   # 端到端推理速度 (含预处理)
  148 ```
  149
  150 ### 4. 评测
  151
  152 ```bash
  153 python evaluation/evaluate_p0.py \
  154     --ckpt results/single_dinov2_run/checkpoints/best.pth \
  155     --data_dir D:/testdata/processed
  156 ```
  157
  158 ---
  159
  160 ## 配置说明 (`config/single_dinov2.yaml`)
  161
  162 | 字段 | 默认值 | 说明 |
  163 |---|---|---|
  164 | `model.knodes` | 5 | 输出的路径关键点数 |
  165 | `model.in_channel` | 16 | Goal 嵌入通道数 |
  166 | `model.img_input_size` | [360, 640] | 输入图像分辨率 (H, W) |
  167 | `loss.w_goal` | 2.0 | 终点损失权重 |
  168 | `loss.w_obs` | 3.0 | 障碍代价损失权重 |
  169 | `loss.w_motion` | 1.0 | 运动平滑损失权重 |
  170 | `loss.w_fear` | 0.0 | 碰撞概率损失权重 |
  171 | `loss.fear_ahead_dist` | 2.5 | 前方检查距离 (m) |
  172 | `loss.obstacle_thresh` | 1.25 | 碰撞判定阈值 |
  173 | `training.*` | — | batch_size / lr / optimizer / scheduler |
  174
  175 ---
  176
  177 ## 评测结果
  178
  179 在 CARLA 测试集上评测（`evaluate_p0.py`，指标：FDE / ADE / 碰撞率 / 成功率 / FPS）：
  180
  181 | 数据集 | 样本数 | ADE (m) | FDE (m) | 碰撞率 | 成功率 | FPS |
  182 |---|---|---|---|---|---|---|
  183 | testdata (主) | 1615 | 0.091 | 0.059 | 4.0% | 95.7% | ~19.7 |
  184 | testdata2 (泛化) | 962  | 0.137 | 0.099 | 20.2% | 79.5% | ~19.7 |
  185
  186 - **成功判定**：终点误差 FDE < 0.5m **且** 路径无碰撞（代价 > 1.25）
  187 - 推理速度约 **50 ms/step**（含模型推理，不含感知管线）；`benchmark_speed.py` 与 ViPlanner 口径一致
  188 - 详细分桶结果见 `evaluation/results_p0.json`
  189
  190 > 具体数值会随训练轮次、数据划分与随机种子变化，请以实际运行 `evaluate_p0.py` 的结果为准。
  191
  192 ---
  193
  194 ## 数据格式 (CARLA 数据集)
  195
  194 ## 数据格式 (CARLA 数据集)
  195
     … +119 lines (ctrl+o to expand)
