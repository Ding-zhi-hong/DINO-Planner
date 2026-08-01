"""
离线预处理: 对 CityWalker 视频逐帧提取语义和深度

流程:
  1. 读取视频帧
  2. Mask2Former → 31类语义RGB编码
  3. DepthAnythingV2 → 深度图
  4. 保存到对应目录

用法:
  python -m data.preprocess --video_dir dataset/videos \
                             --sem_dir dataset/sem_rgb \
                             --depth_dir dataset/depth
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import torch


def build_semantic_pipeline():
    """
    构建 Mask2Former 语义分割管线
    返回: callable, 输入 RGB (H,W,3) → 输出语义RGB (H,W,3)
    """
    try:
        from mmdet.apis import init_detector, inference_detector
        from mmdet.evaluation import INSTANCE_OFFSET

        # COCO 171类 → 31类ViPlanner 颜色映射
        from viplanner.config.coco_sem_meta import get_class_for_id_mmdet
        from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler

        config = 'mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
        checkpoint = 'mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
        model = init_detector(config, checkpoint, device='cuda:0')

        viplanner_meta = VIPlannerSemMetaHandler()
        coco_mapping = get_class_for_id_mmdet(model.dataset_meta['classes'])

        def predict(image):
            """image: (H, W, 3) BGR uint8 → (3, H, W) 31类语义RGB uint8"""
            result = inference_detector(model, image)
            sem_seg = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
            out = np.zeros((*sem_seg.shape, 3), dtype=np.uint8)
            for cls_id in np.unique(sem_seg):
                label = cls_id % INSTANCE_OFFSET
                cls_name = coco_mapping.get(label, 'static')
                color = viplanner_meta.class_color.get(cls_name, [0, 0, 0])
                out[sem_seg == cls_id] = color
            return out.transpose(2, 0, 1)  # (3, H, W)

        return predict

    except ImportError:
        print('[WARNING] mmdet not installed, using dummy semantic')
        def predict(image):
            return np.zeros((3, image.shape[0], image.shape[1]), dtype=np.uint8)
        return predict


def build_depth_pipeline():
    """
    构建 DepthAnythingV2 深度估计管线
    返回: callable, 输入 RGB (H,W,3) → 输出深度 (1, H, W)
    """
    try:
        import sys
        sys.path.append('Depth-Anything-V2')
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        model = DepthAnythingV2(**model_configs['vits'])
        model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth',
                                         map_location='cuda'))
        model = model.cuda().eval()

        def predict(image):
            """image: (H, W, 3) RGB uint8 → (1, H, W) float32"""
            depth = model.infer_image(image)  # (H, W)
            return depth[None, :, :].astype(np.float32)  # (1, H, W)

        return predict

    except ImportError:
        print('[WARNING] DepthAnythingV2 not installed, using dummy depth')
        def predict(image):
            return np.ones((1, image.shape[0], image.shape[1]), dtype=np.float32)
        return predict


def process_video(video_path, sem_dir, depth_dir, sem_pipeline, depth_pipeline):
    """处理单个视频文件"""
    import cv2

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=video_name) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔 6 帧处理一次 (5fps ≈ 视频 30fps 采样)
            if frame_idx % 6 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 语义
                key = f'{video_name}_{frame_idx//6:06d}'
                sem_path = os.path.join(sem_dir, f'{key}.npy')
                if not os.path.exists(sem_path):
                    sem = sem_pipeline(frame_rgb)
                    np.save(sem_path, sem)

                # 深度
                depth_path = os.path.join(depth_dir, f'{key}.npy')
                if not os.path.exists(depth_path):
                    depth = depth_pipeline(frame_rgb)
                    np.save(depth_path, depth)

            frame_idx += 1
            pbar.update(1)

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--sem_dir', required=True)
    parser.add_argument('--depth_dir', required=True)
    parser.add_argument('--num_videos', type=int, default=None)
    args = parser.parse_args()

    sem_pipeline = build_semantic_pipeline()
    depth_pipeline = build_depth_pipeline()

    videos = sorted([f for f in os.listdir(args.video_dir) if f.endswith('.mp4')])
    if args.num_videos:
        videos = videos[:args.num_videos]

    for v in videos:
        process_video(
            os.path.join(args.video_dir, v),
            args.sem_dir,
            args.depth_dir,
            sem_pipeline,
            depth_pipeline,
        )


if __name__ == '__main__':
    main()
