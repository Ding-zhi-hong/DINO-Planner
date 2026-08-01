"""推理速度基准测试 — 模型原始 5 关键点输出"""
import time, sys, numpy as np, cv2, torch
sys.path.insert(0, '.')
from model import SingleDINOv2Nav
from predict import load_config, load_model, preprocess

CKPT = 'results/single_dinov2_run/checkpoints/best.pth'
CONFIG = 'config/single_dinov2.yaml'
TEST_IMAGE = 'D:/carla/traindata/sample_trajectory_20260718_191609_00210470/000004.jpg'

device = torch.device('cuda')
print(f'device={device}')

cfg = load_config(CONFIG)
model = load_model(CKPT, cfg, device)

image_bgr = cv2.imread(TEST_IMAGE)
rgb_t = preprocess(image_bgr).to(device)
goal_t = torch.from_numpy(np.array([3.0, 5.0], dtype=np.float32)).float().unsqueeze(0).to(device)

# warmup
N_warmup = 10
for _ in range(N_warmup):
    with torch.no_grad():
        kp, fear = model(rgb_t, goal_t)
torch.cuda.synchronize()

# benchmark (只测模型推理, 无后处理)
N = 500
start = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        kp, fear = model(rgb_t, goal_t)
torch.cuda.synchronize()
end = time.perf_counter()

total = end - start
avg_ms = total / N * 1000
fps = N / total

total_p = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

print()
print('═══ Benchmark Results ═══')
print(f'  Python:       {sys.version.split()[0]}')
print(f'  Torch:        {torch.__version__}')
print(f'  GPU:          {torch.cuda.get_device_name()}')
print(f'  Params:       {total_p:,} total, {trainable_p:,} trainable')
print(f'  Warmup:       {N_warmup}')
print(f'  Runs:         {N}')
print(f'  Total time:   {total:.3f} s')
print(f'  Avg per step: {avg_ms:.2f} ms')
print(f'  FPS:          {fps:.1f}')
print('══════════════════════════')