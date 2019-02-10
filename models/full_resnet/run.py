import os
import time
import numpy as np
import pickle
import torch
from torch.utils import data
from torch import nn, optim
from torchvision.models import resnet152
import matplotlib.pyplot as plt

from dataset import NyuV2
from modeling import train_model
from dbe import DBELoss


seed = 2
torch.manual_seed(seed)

# logger
import logging
from train_logger import TrainHandler
logging.basicConfig(filename='./logs/training_resnet_rev4.log', level=logging.INFO)
logger = logging.getLogger('root')
logger.addHandler(TrainHandler())

# Experiment
exp_name = 'ft_resnet'
exp_dir = os.path.join('./models/', exp_name)
os.mkdir(exp_dir)

# dirs and files
data_path = './data/nyu_v2/'

# params
batch_size = 16
depth_size = (25, 32)
n_epochs = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

dataloaders = {
    'train': data.DataLoader(NyuV2(os.path.join(data_path, 'train')),
                               batch_size=batch_size, shuffle=True),
    'val': data.DataLoader(NyuV2(os.path.join(data_path, 'val')),
                              batch_size=batch_size, shuffle=True)
}

model = resnet152(pretrained=True)
model.fc = nn.Linear(2048, depth_size[0] * depth_size[1])
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = DBELoss()
criterion = nn.MSELoss(reduction='sum')


train_model(model, dataloaders, criterion, optimizer, n_epochs, device)
