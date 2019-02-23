import os
import torch
from torch.utils import data
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet152
import logging
from shutil import copy

from train_logger import TrainHandler
from dataset import NyuV2
from modeling import train_model
from dbe import DBELoss
from den_gen2 import DEN
import utils
import transforms_nyu


print("PyTorch Version: ",torch.__version__)


seed = 2
torch.manual_seed(seed)

# Experiment
exp_name = 'den_gen2_v2'
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
test_crop = (427, 561)
depth_size = (25, 32)

# hyperparams
early_stopping_th = 50
n_epochs = 500
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

transformers = {
    'train': transforms.Compose([transforms_nyu.Normalize(),
                                 transforms_nyu.RandomRescale(0.1),
                                 transforms_nyu.RandomRotate(10),
                                 transforms_nyu.RandomCrop(test_crop),
                                 transforms_nyu.RandomHorizontalFlip(0.5),
                                 transforms_nyu.ScaleDown(),
                                 transforms_nyu.ToTensor()]),

    'val': transforms.Compose([transforms_nyu.Normalize(),
                               transforms_nyu.CenterCrop(test_crop),
                               transforms_nyu.ScaleDown(),
                               transforms_nyu.ToTensor()])
}

nyu = {
    'train': NyuV2(os.path.join(data_path, 'train'), transform=transformers['train']),

    'val': NyuV2(os.path.join(data_path, 'val'), transform=transformers['val'])
}

dataloaders = {
    'train': data.DataLoader(nyu['train'], num_workers=8,
                             batch_size=batch_size, shuffle=True),

    'val': data.DataLoader(nyu['val'], num_workers=8,
                           batch_size=batch_size, shuffle=True)
}

resnet_wts = './models/pretrained_resnet/model.pt'
model = DEN(resnet_wts)
model = model.to(device)

params_to_update = utils.params_to_update(model)
optimizer = optim.Adam(model.parameters(), lr=16e-5)
criterion = nn.MSELoss(reduction='sum')

train_model(model, dataloaders, criterion, optimizer, n_epochs, device, exp_dir, early_stopping_th)
