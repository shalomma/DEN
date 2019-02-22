import torch
import time
import os
import copy
import logging
import pickle

logger = logging.getLogger('root')

# dirs and files
model_file = '_model.pt'
train_loss_pkl = 'train_loss.pkl'
val_loss_pkl = 'val_loss.pkl'

criterion_test = torch.nn.MSELoss(reduction='sum')

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, save_dir, early_stopping_th):
        
    since = time.time()

    val_loss_history = []
    train_loss_history = []
    early_stopping_counter = early_stopping_th
    best_loss = 1000.0

    for epoch in range(1, num_epochs+1):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))
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
            for data in dataloaders[phase]:
                inputs = data['image'].to(device).float()
                labels = data['depth'].to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    elif phase == 'val':
                        loss = criterion_test(outputs, labels)

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            to_print = 'Loss: {:.4f} RMSE: {:.4f}'.format(epoch_loss, epoch_loss**0.5) 
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 torch.save(best_model_wts, os.path.join(save_dir, '{:03}{}'.format(epoch, model_file)))
                early_stopping_counter = early_stopping_th
            elif phase == 'val':
                early_stopping_counter -= 1
            if phase == 'val':
                logger.info('Validation: ' + to_print)
                val_loss_history.append(epoch_loss)
            elif phase == 'train':
                logger.info('Training:   ' + to_print)
                train_loss_history.append(epoch_loss)
            
        # early stopper
        if early_stopping_counter == 0:
            logger.info('Early stopping...')
            break
                
        time_elapsed = time.time() - since_epoch
        logger.info('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
        # update saved losses
        if epoch % 2 == 0:
            pickle.dump(train_loss_history, open(os.path.join(save_dir, train_loss_pkl), 'wb'))
            pickle.dump(val_loss_history, open(os.path.join(save_dir, val_loss_pkl), 'wb'))
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, '{:03}{}'.format(epoch, model_file)))
    
    # end of training
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600,
                                                                      (time_elapsed // 60) % 60,
                                                                      time_elapsed % 60))
    logger.info('Best Val Loss: {:.4f} RMSE: {:.4f}'.format(best_loss, best_loss**0.5))
