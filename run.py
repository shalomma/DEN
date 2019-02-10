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


seed = 2
torch.manual_seed(seed)

# Experiment
exp_name = 'early_try'
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
early_stopping_th = 3
n_epochs = 240
batch_size = 16


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

dataloaders = {
    'train': data.DataLoader(NyuV2(os.path.join(data_path, 'train')),
                               batch_size=batch_size, shuffle=True),
    'val': data.DataLoader(NyuV2(os.path.join(data_path, 'val')),
                              batch_size=batch_size, shuffle=True)
}


def init_ft_model(model):
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(2048, depth_size[0] * depth_size[1])
    
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
            
    return model, params_to_update


model = resnet152(pretrained=True)
model, params_to_update = init_ft_model(model)
model = model.to(device)

optimizer = optim.Adam(params_to_update, lr=1e-4)
# criterion = DBELoss()
criterion = nn.MSELoss(reduction='sum')


train_model(model, dataloaders, criterion, optimizer, n_epochs, device, exp_dir, early_stopping_th)
