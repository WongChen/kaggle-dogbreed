import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scipy.misc
from random import random



def center_crop(img, target_size=200):
    shape = img.shape
    min_l = np.min(shape[:-1])
    ratio = float(target_size)/min_l
    img = scipy.misc.imresize(img, ratio)
    h, w = img.shape[:-1]
    if h>w:
        b = (h-w)/2 + int(np.round(random()*15))
        img = img[b:b+target_size, :, :]
    if h<w:
        b = (w-h)/2 + int(np.round(random()*15))
        img = img[:, b:b+target_size, :]
    return cv2.resize(img, (200, 200))

class DogDataset(Dataset):
    def __init__(self, root_dir, label_csv_name):
        self.root_dir = root_dir
        df = pd.read_csv(label_csv_name) 
        self.id = list(df.id)
        self.class_mapping = {label:idx for idx, label in enumerate(np.unique(df.breed))}
        df['class_idx'] = df['breed'].map(self.class_mapping)
        self.labels = list(df.class_idx)
    def __len__(self):
        return len(self.id)
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root_dir, self.id[idx]+'.jpg'), -1)
        img = center_crop(img).astype(np.float32).reshape([3,200,200])
        return img, self.labels[idx]



