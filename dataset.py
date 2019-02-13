import os
import numpy as np
import pickle
import torch
from torch.utils import data
from skimage import transform


max_depth_val = 10
img_size = (224, 224)
depth_size = (25, 32)
max_img_val = 255.0


class NyuV2(data.Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, 'images')))
    
    def __getitem__(self, index):
        f_img = open(os.path.join(self.root_dir, 'images', '{:05d}.p'.format(index)), 'rb')
        img = pickle.load(f_img)
        f_img.close()
        
        f_depth = open(os.path.join(self.root_dir, 'depths', '{:05d}.p'.format(index)), 'rb')
        depth = pickle.load(f_depth)
        f_depth.close()
        
        sample = {'image': img, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
