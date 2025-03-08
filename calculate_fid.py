import numpy as np
import torch
from tqdm import tqdm
import torchvision
from scipy.linalg import sqrtm
from torchvision.models import Inception_V3_Weights
def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fid(videos1, videos2, device, only_final=False):


    print("calculate_fid...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    inception_model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    # inception_model.to(device)
    # i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fid_results = []

    for i in tqdm(range(videos1.shape[0])):
    
        videos_clip1 = videos1[i].permute(1,0,2,3)
        videos_clip2 = videos2[i].permute(1,0,2,3)
        # print(videos_clip2.shape)
        fid = calculate_video_fid(videos_clip1, videos_clip2, inception_model)
    
        # calculate FVD when timestamps[:clip]
        fid_results.append(fid)

    if only_final:
        fid_results = np.mean(fid_results)
    result = {
        "value": fid_results,
    }

    return result

def calculate_video_fid(real_tensor, fake_tensor, inception_model):
    inception_model.eval()
    with torch.no_grad():
        real_features = inception_model(real_tensor).cpu().numpy()
        fake_features = inception_model(fake_tensor).cpu().numpy()
    
    # Calculate statistics for both sets of features
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    # Calculate FID
    covmean = sqrtm(sigma1.dot(sigma2) + 1e-6, disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    mean_diff = mu1 - mu2
    fid = np.sum(mean_diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, 576, 320, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, 576, 320, requires_grad=False)
    device = torch.device("cuda:0")
    result = calculate_fid(videos1, videos2, device)
    print("[fvd-videogpt ]", result["value"])

    videos_random1 = torch.rand(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, 576, 320, requires_grad=False)
    videos_random2 = videos_random1.clone()
    result = calculate_fid(videos_random1, videos_random2, device)
    print("[fvd-styleganv]", result["value"])


if __name__ == "__main__":
    main()
