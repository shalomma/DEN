import os
import time
import numpy as np
import pickle
import torch
from torch.utils import data
from torch import nn, optim
from torchvision.models import resnet152
import matplotlib.pyplot as plt
import logging
from shutil import copy

from train_logger import TrainHandler
from dataset import NyuV2
from modeling import train_model
from dbe import DBELoss
from den import DEN
import utils

seed = 2
torch.manual_seed(seed)

# Experiment
exp_name = 'den_dbe'
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

# hyperparams
early_stopping_th = 10
n_epochs = 200
batch_size = 16
wts_file = './models/full_resnet/149_resnet_model.pt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

dataloaders = {
    'train': data.DataLoader(NyuV2(os.path.join(data_path, 'train')),
                               batch_size=batch_size, shuffle=True),
    'val': data.DataLoader(NyuV2(os.path.join(data_path, 'val')),
                              batch_size=batch_size, shuffle=True)
}


model = DEN(wts_file)
params_to_update = utils.params_to_update(model)
model = model.to(device)

optimizer = optim.Adam(params_to_update, lr=1e-4)
criterion = DBELoss()
# criterion = nn.MSELoss(reduction='sum')

train_model(model, dataloaders, criterion, optimizer, n_epochs, device, exp_dir, early_stopping_th)
