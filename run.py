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

# logger
import logging
from train_logger import TrainHandler
logging.basicConfig(filename='./logs/training_resnet_rev3_dbe.log', level=logging.INFO)
logger = logging.getLogger('root')
logger.addHandler(TrainHandler())

# dirs and files
data_path = './data/nyu_v2/'
model_file = './models/ft_resnet.pt'
batch_size = 16
# img_size = (224, 224)
depth_size = (25, 32)
# max_img_val = 255.0
# max_depth_val = 9.9955
n_epochs = 400
seed = 2
torch.manual_seed(seed)
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
criterion = DBELoss()


train_model(model, dataloaders, criterion, optimizer, n_epochs, device)
