'''
adapted from: https://blog.csdn.net/weixin_54338498/article/details/134387853
'''

import cv2
import torch
import clip
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from decord import VideoReader, cpu
import numpy as np

class Video_Metric:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(self.device)
        self.model, self.preprocess = clip.load("ViT-B/32", device = self.device)

    def clip_i2i(self, image1, image2):
        image1 = self.preprocess(image1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(image2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image1_features = self.model.encode_image(image1)
            image2_features = self.model.encode_image(image2)
        image1_features /= image1_features.norm(dim = -1, keepdim = True)
        image2_features /= image2_features.norm(dim = -1, keepdim = True)
        return max(0, (image1_features @ image2_features.T).squeeze(0).cpu().item())

    def clip_v2v(self, video_path1 : str, video_path2 : str):
        video1 = VideoReader(video_path1, ctx = cpu(0))
        video2 = VideoReader(video_path2, ctx = cpu(0))
        frames = len(video1)

        scores = []
        for i in range(frames):
            image1 = Image.fromarray(video1[i].asnumpy())
            image2 = Image.fromarray(video2[i].asnumpy())
            scores.append(self.clip_i2i(image1, image2))
        
        avg_score = sum(scores) / frames
        return scores, avg_score