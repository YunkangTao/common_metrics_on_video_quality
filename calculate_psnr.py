import numpy as np
import torch
from tqdm import tqdm
import math

def img_psnr(img1, img2):
    # [0,1]
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def trans(x):
    return x

def calculate_psnr(videos1, videos2, only_final=False):
    print("calculate_psnr...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    psnr_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].numpy()
            img2 = video2[clip_timestamp].numpy()
            
            # calculate psnr of a video
            psnr_results_of_a_video.append(img_psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)
    
    psnr_results = np.array(psnr_results)
    
    psnr = []
    psnr_std = []

    if only_final:
        psnr.append(np.mean(psnr_results))
        # psnr_std.append(np.std(psnr_results))
    else:
        for i in range(len(psnr_results)):
            psnr.append(np.mean(psnr_results[i]))
            # psnr_std.append(np.std(psnr_results[i]))

    result = {
        "value": psnr,
        # "value_std": psnr_std,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

    result = calculate_psnr(videos1, videos2)
    print("[psnr avg]", result["value"])
    # print("[psnr std]", result["value_std"])

    videos_random1 = torch.rand(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos_random2 = videos_random1.clone()
    result = calculate_psnr(videos_random1, videos_random2)
    print("[psnr avg]", result["value"])
    # print("[psnr std]", result["value_std"])

if __name__ == "__main__":
    main()