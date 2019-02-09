from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
import pickle
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



logger = logging.getLogger('root')

# dirs and files
models_dir = './models/'
best_model_file = 'best_resnet_model.pt'
model_file = '_resnet_model.pt'
train_loss_pkl = './models/train_loss_dbe.pkl'
val_loss_pkl = './models/val_loss_dbe.pkl'

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()

    val_loss_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0

    for epoch in range(1, num_epochs+1):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        since_epoch = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            
            to_print = 'Loss: {:.4f} RMSE: {:.4f}'.format(epoch_loss, epoch_loss**0.5) 
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(models_dir, '{:03}{}'.format(epoch, model_file)))
            if phase == 'val':
                logger.info('Validation: ' + to_print)
                val_loss_history.append(epoch_loss)
            elif phase == 'train':
                logger.info('Training:   ' + to_print)
                train_loss_history.append(epoch_loss)
                
        time_elapsed = time.time() - since_epoch
        logger.info('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
        # save losses
        if epoch % 20 == 0:
            pickle.dump(train_loss_history, open(train_loss_pkl, 'wb'))
            pickle.dump(val_loss_history, open(val_loss_pkl, 'wb'))
    
    # end of training
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    torch.save(best_model_wts, os.path.join(models_dir, best_model_file))
