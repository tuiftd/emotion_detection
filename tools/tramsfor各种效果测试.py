import os
import math
import sys
import json
import torch
import cv2 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
class openCV2PIL:
    def __init__(self,transform=None):
        self.transform = transform
        self.img = None 
    def cv2pil(self,cv2_img): 
        opencv_image_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(opencv_image_rgb)
        return pil_image
    def pil2cv(self,pil_img):
        pil_img = np.array(pil_img)
        opencv_image_bgr = cv2.cvtColor(pil_img, cv2.COLOR_RGB2BGR)
        return opencv_image_bgr
    def __call__(self,cv2_img,transform=None):
        self.img = cv2_img
        self.transform = transform
        if  self.img is None:
            raise ValueError("cv2_img is None")
        self.img = self.cv2pil(self.img)
        if type(self.img) == np.ndarray:
            raise ValueError("img is not PIL Image")
        if self.transform is not None:
            self.img = self.transform(self.img)
        else:
            pass
        if type(self.img) == torch.Tensor:
            self.img =transforms.ToPILImage()(self.img)
        self.img = self.pil2cv(self.img)
        return self.img
def main():
    trans = openCV2PIL()
    img_path = "images\OIP-C (3).jfif"
    trans_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(0.1, 0.5), ratio=(1, 1), value=0),
    ])
    img = cv2.imread(img_path)
    cv2.imshow("ori_img",img)
    img = trans(img,trans_list)
    print(img.shape)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()

#& C:/Users/gd/AppData/Local/anaconda3/envs/yolo/python.exe c:/Users/gd/Desktop/facecheck/tramsfor各种效果测试.py