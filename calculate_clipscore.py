import os
import glob
from Clipscore import Video_Metric
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def calculate_clipscore(video_path1, video_path2):
    metric = Video_Metric()
    video_paths1 = glob.glob(os.path.join(video_path1, '*.mp4'))
    video_paths2 = glob.glob(os.path.join(video_path2, '*.mp4'))
    video_paths1.sort()
    video_paths2.sort()
    num_videos = min(len(video_paths1), len(video_paths2))
    video_paths1 = video_paths1[:num_videos]
    video_paths2 = video_paths2[:num_videos]
    sum_score = 0
    for i in range(num_videos):
        video_path1 = video_paths1[i]
        video_path2 = video_paths2[i]
        scores, avg_score = metric.clip_v2v(video_path1, video_path2)
        sum_score += avg_score
    sum_score /= num_videos
    # print(f'{num_videos} videos')
    # print(f'Average ClipScore: {avg_score}')
    return avg_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path1', type = str, help = "video path 1", default = '/home/chenyang_lei/video_diffusion_models/common_metrics_on_video_quality/baselines_result/cameractrl_svd_1000_test/gt_videos')
    parser.add_argument('--path2', type = str, help = "video path 2", default = '/home/chenyang_lei/video_diffusion_models/common_metrics_on_video_quality/baselines_result/cameractrl_svd_1000_test/gt_videos')
    args = parser.parse_args()
    metric = Video_Metric()
    video_paths1 = glob.glob(os.path.join(args.path1, '*.mp4'))
    video_paths2 = glob.glob(os.path.join(args.path2, '*.mp4'))
    video_paths1.sort()
    video_paths2.sort()
    num_videos = len(video_paths2)
    video_paths1 = video_paths1[:num_videos]
    sum_score = 0
    for i in range(num_videos):
        video_path1 = video_paths1[i]
        video_path2 = video_paths2[i]
        scores, avg_score = metric.clip_v2v(video_path1, video_path2)
        sum_score += avg_score
        print(scores)
    sum_score /= num_videos
    print(f'{num_videos} videos')
    print(f'Average ClipScore: {avg_score}')