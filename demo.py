import json
import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips


def get_original_videos(videos_path):
    # read videos from videos_path
    return videos


def get_generated_videos(videos_path):
    # read videos from videos_path
    return videos


def main(videos1, videos2, device, output_path):
    result = {}
    # only_final = False
    only_final = True
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
    # result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=only_final)
    result['ssim'] = calculate_ssim(videos1, videos2, only_final=only_final)
    result['psnr'] = calculate_psnr(videos1, videos2, only_final=only_final)
    result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=only_final)

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # ps: pixel value should be in [0, 1]!
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 59
    CHANNEL = 3
    SIZE = 512
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

    videos1_path = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/RealEstate10K/test_clips/"
    vidoes1 = get_original_videos(videos1_path)

    videos2_path = ""
    vidoes2 = get_generated_videos(videos2_path)

    device = torch.device("cuda")
    # device = torch.device("cpu")

    output_path = 'output1.json'

    main(videos1, videos2, device, output_path)
