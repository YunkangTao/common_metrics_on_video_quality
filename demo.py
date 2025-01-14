import hashlib
import json
import os
from typing import List

import cv2
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional
from tqdm import tqdm

from calculate_fvd import calculate_fvd
from calculate_lpips import calculate_lpips
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
from calculate_fid import Calculate_fid
from calculate_clipscore import calculate_clipscore


def get_all_mp4_files(root_dir: str, max_videos = None) -> List[str]:
    """
    递归获取给定目录及其子目录下所有的 .mp4 文件。
    """
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                full_path = os.path.join(dirpath, filename)
                mp4_files.append(full_path)
    mp4_files.sort()  # 对列表进行排序
    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]
    return mp4_files


def read_video_frames(video_path: str, max_frames: int = None, crop_size: int = 224) -> torch.Tensor:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 将 BGR 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # 可选：计算并打印帧的哈希值
        # frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        # print(f"Frame {frame_count} hash: {frame_hash}")

        # 转换为 PIL 图像
        frame_pil = torchvision.transforms.functional.to_pil_image(frame)
        width, height = frame_pil.size

        if crop_size > width or crop_size > height:
            raise ValueError(f"Crop size {crop_size} is larger than frame size {width}x{height} in video {video_path}")

        # 计算从右下角开始裁剪的左上角坐标
        left = width - crop_size
        top = height - crop_size
        # 执行裁剪
        frame_cropped = torchvision.transforms.functional.crop(frame_pil, top, left, crop_size, crop_size)
        # 转换为张量并归一化到 [0,1]
        frame_tensor = torchvision.transforms.functional.to_tensor(frame_cropped)
        frames.append(frame_tensor)

        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video {video_path}")

    # print(f"Video {video_path}: {len(frames)} frames read, target max_frames={max_frames}")

    # 如果指定了最大帧数，进行填充或截断
    if max_frames is not None:
        if len(frames) < max_frames:
            # 使用全零张量填充
            pad_frames = [torch.zeros_like(frames[0]) for _ in range(max_frames - len(frames))]
            frames.extend(pad_frames)
        else:
            frames = frames[:max_frames]

    # 堆叠帧为 [frames, channels, size, size]
    video_tensor = torch.stack(frames)
    # print(f"Final video tensor shape: {video_tensor.shape}")
    return video_tensor


def generate_video_tensor(root_dir: str, crop_size: int = 224, max_frames: int = 30, max_videos: int = 1000) -> torch.Tensor:
    """
    生成包含所有视频的张量。

    Args:
        root_dir (str): 包含 MP4 文件的根目录。
        crop_size (int): 每帧要裁剪的大小。
        max_frames (int): 每个视频的帧数。

    Returns:
        torch.Tensor: 张量形状为 [num, frames, channels, size, size]。
    """
    video_paths = get_all_mp4_files(root_dir, max_videos)
    num_videos = len(video_paths)
    if num_videos == 0:
        raise ValueError("指定目录下未找到 MP4 文件。")

    # 用于收集视频张量的占位列表
    video_tensors = []

    print(f"找到 {num_videos} 个视频。正在处理...")
    for video_path in tqdm(video_paths, desc="处理视频"):
        try:
            video_tensor = read_video_frames(video_path, max_frames=max_frames, crop_size=crop_size)
            video_tensors.append(video_tensor)
        except Exception as e:
            print(f"处理 {video_path} 时出错: {e}")

    # 堆叠所有视频张量为 [num, frames, channels, size, size]
    final_tensor = torch.stack(video_tensors)
    return final_tensor


def main(videos1, videos2, device, output_path, video_paths1, video_paths2):
    result = {}
    # only_final = False
    only_final = True
    # result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=only_final)
    result['ssim'] = calculate_ssim(videos1, videos2, only_final=only_final)
    result['psnr'] = calculate_psnr(videos1, videos2, only_final=only_final)
    result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=only_final)
    result['fid'] = Calculate_fid()
    result['clipscore'] = calculate_clipscore(video_paths1, video_paths2)

    # with open(output_path, 'w', encoding='utf-8') as file:
    #     json.dump(result, file, indent=5, ensure_ascii=False)
    print(result)


if __name__ == '__main__':
    # ps: pixel value should be in [0, 1]!
    # NUMBER_OF_VIDEOS = 10
    # VIDEO_LENGTH = 49
    # CHANNEL = 3
    # SIZE = 512
    # videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    # videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

    videos1_path = '/home/lingcheng/RealEstate10KAfterProcess/test_clips'
    videos1 = generate_video_tensor(root_dir=videos1_path, crop_size=512, max_frames=None, max_videos=None)

    videos2_path = "/home/chenyang_lei/video_diffusion_models/EasyAnimateCameraControl/output_dir_20250107_inpainting_with_mask_all_realestate/test_clips"
    # videos2_path = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/RealEstate10K/test_clips/"
    videos2 = generate_video_tensor(root_dir=videos2_path, crop_size=512, max_frames=None, max_videos=100)

    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    output_path = 'output_20250107_all_realestate.json'

    main(videos1, videos2, device, output_path, videos1_path, videos2_path)
