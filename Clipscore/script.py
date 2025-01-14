import os
import glob
from clipscore import Video_Metric
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def calculate_clipscore(video_path1, video_path2):
    metric = Video_Metric()
    video_paths1 = glob.glob(os.path.join(video_path1, '/*')).sort()
    video_paths2 = glob.glob(os.path.join(video_path2, '/*')).sort()
    num_videos = len(video_paths1)
    sum_score = 0
    for i in range(num_videos):
        video_path1 = video_paths1[i]
        video_path2 = video_paths2[i]
        scores, avg_score = metric.clip_v2v(video_path1, video_path2)
        sum_score += avg_score
    sum_score /= num_videos
    print(f'{num_videos} videos')
    print(f'Average ClipScore: {avg_score}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path1', type = str, help = "video path 1", default = '../input_video.mp4')
    parser.add_argument('--path2', type = str, help = "video path 2", default = '../input_video.mp4')
    args = parser.parse_args()
    metric = Video_Metric()
    video_paths1 = glob.glob(os.path.join(args.path1, '/*'))
    video_paths2 = glob.glob(os.path.join(args.path2, '/*'))
    num_videos = len(video_paths1)
    sum_score = 0
    for i in range(num_videos):
        video_path1 = video_paths1[i]
        video_path2 = video_paths2[i]
        scores, avg_score = metric.clip_v2v(video_path1, video_path2)
        sum_score += avg_score
    sum_score /= num_videos
    print(f'{num_videos} videos')
    print(f'Average ClipScore: {avg_score}')