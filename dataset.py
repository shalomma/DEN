import os
import pickle
from torch.utils import data


class NyuV2(data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, 'images')))
    
    def __getitem__(self, index):
        with open(os.path.join(self.root_dir, 'images', '{:05d}.p'.format(index)), 'rb') as f_img:
            img = pickle.load(f_img)

        with open(os.path.join(self.root_dir, 'depths', '{:05d}.p'.format(index)), 'rb') as f_depth:
            depth = pickle.load(f_depth)

        sample = {'image': img, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
