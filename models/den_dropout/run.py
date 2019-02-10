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


seed = 2
torch.manual_seed(seed)

# Experiment
exp_name = 'den_dropout'
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
early_stopping_th = 40
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


def params_to_update(model):
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
            
    return params_to_update


def plot_grad_flow(named_parameters, saved_file):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.data.abs().mean())
            max_grads.append(p.data.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(saved_file)


model = DEN(wts_file)
params_to_update = params_to_update(model)
model = model.to(device)

optimizer = optim.Adam(params_to_update, lr=1e-4)
# criterion = DBELoss()
criterion = nn.MSELoss(reduction='sum')

train_model(model, dataloaders, criterion, optimizer, n_epochs, device, exp_dir, early_stopping_th)
