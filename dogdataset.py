import cv2
from torchvision import transforms, utils
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scipy.misc
from random import random



def center_crop(img, target_size=224):
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
    return cv2.resize(img, (target_size, target_size))

class TestDataset(Dataset):
    def __init__(self, root_dir, label_csv_name):
        self.root_dir = root_dir
        df = pd.read_csv(label_csv_name) 
        self.id = list(df.id)
        self.class_mapping = {label:idx for idx, label in enumerate(np.unique(df.breed))}
        self.ce_weight = np.array([np.sum(df.breed==dog) for dog in np.unique(df.breed)], dtype=np.float32)
        self.ce_weight = np.sum(self.ce_weight) / self.ce_weight
        self.ce_weight = self.ce_weight/ np.sum(self.ce_weight)
        df['class_idx'] = df['breed'].map(self.class_mapping)
        
        self.labels = list(df.class_idx)
    def __len__(self):
        return len(self.id)
    def __getitem__(self, idx):
        target_size = 224
        img = cv2.imread(os.path.join(self.root_dir, self.id[idx]+'.jpg'), -1)
        img = center_crop(img, target_size=target_size).astype(np.float32).transpose((2,0,1))
        return img, self.labels[idx]

class TrainDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        target_size = 224
        img = cv2.imread(os.path.join(self.root_dir, '%s.png'%idx), -1)
        img = img.transpose((2,0,1)).astype(np.float32)
        return img, self.labels[idx]


