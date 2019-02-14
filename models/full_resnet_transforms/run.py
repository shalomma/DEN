import os
import time
import numpy as np
import pickle
import torch
from torch.utils import data
from torch import nn, optim
from torchvision.models import resnet152
from torchvision import transforms
import matplotlib.pyplot as plt
import logging
from shutil import copy
import skimage

from train_logger import TrainHandler
from dataset import NyuV2
from modeling import train_model
from dbe import DBELoss
from den import DEN
import utils
import transforms_nyu

seed = 2
torch.manual_seed(seed)

# Experiment
exp_name = 'full_resnet_transforms'
exp_dir = os.path.join('./models/', exp_name)
if os.path.exists(exp_dir):
    print('Enter new experiment name!')
    exit()
else:
    print('Preparing experiment directory...')
    os.mkdir(exp_dir)
    copy('run.py', exp_dir)
    copy('modeling.py', exp_dir)
    

# logger
logging.basicConfig(filename=os.path.join(exp_dir, 'training.log'), level=logging.INFO)
logger = logging.getLogger('root')
logger.addHandler(TrainHandler())

# dirs and files
data_path = './data/nyu_v2/'

# params
depth_size = (25, 32)
input_size = 224

# hyperparams
early_stopping_th = 50
n_epochs = 250
batch_size = 16


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

transformers = {
    'train': transforms.Compose([transforms_nyu.Scale(),
                               transforms_nyu.RandomRescale(480),
                               transforms_nyu.RandomCrop(input_size),
                               transforms_nyu.RandomHorizontalFlip(0.5),
                               transforms_nyu.ToTensor()]),

    'val': transforms.Compose([transforms_nyu.Scale(),
                               transforms_nyu.ToTensor()])
}

nyu = {
    'train': NyuV2(os.path.join(data_path, 'train'), transform=transformers['train']),

    'val': NyuV2(os.path.join(data_path, 'val'), transform=transformers['val'])
}

dataloaders = {
    'train': data.DataLoader(nyu['train'],
                             batch_size=batch_size, shuffle=True),
    'val': data.DataLoader(nyu['val'],
                           batch_size=batch_size, shuffle=True)
}


model = resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, depth_size[0] * depth_size[1])
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss(reduction='sum')

train_model(model, dataloaders, criterion, optimizer, n_epochs, device, exp_dir, early_stopping_th)
