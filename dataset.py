import os
import numpy as np
import pickle
import torch
from torch.utils import data
from skimage import transform


max_depth_val = 9.9955
img_size = (224, 224)
depth_size = (25, 32)
max_img_val = 255.0


def img_transform(img):
    img = transform.resize(img, img_size, preserve_range=True).astype('float32') / max_img_val
    return np.moveaxis(img, -1, 0)

def depth_transform(depth):
    depth = transform.resize(depth, depth_size, preserve_range=True).astype('float32') / max_depth_val
    return depth.flatten()

class NyuV2(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, 'images')))
    
    def __getitem__(self, index):
        f_img = open(os.path.join(self.root_dir, 'images', '{:05d}.p'.format(index)), 'rb')
        img = img_transform(pickle.load(f_img))
        f_img.close()
        
        f_depth = open(os.path.join(self.root_dir, 'depths', '{:05d}.p'.format(index)), 'rb')
        depth = depth_transform(pickle.load(f_depth))
        f_depth.close()
        
        return img, depth
