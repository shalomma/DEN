import os
import numpy as np
import h5py
import pickle


# data path
data_path = './data/nyu_v2/'
data_file = 'nyu_depth_v2_labeled.mat'
# read mat file
f = h5py.File(os.path.join(data_path, data_file))

N = len(f['images'])
train_size = 1200
images_conv = []
depths_conv = []
for n in range(N):
    if n % 200 == 0:
        print('{}...'.format(n))
    
    if n < train_size:
        group = 'train'
        index = n
    else:
        group = 'val'
        index = n - train_size
    
    img = f['images'][n]

    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T
    
    img__ = img_.astype('float32')
    images_conv.append(img__)
    img_file = open(os.path.join(data_path, group, 'images', '{:05d}.p'.format(index)), 'wb')
    pickle.dump(img__, img_file)
    img_file.close()
    
    depth = f['depths'][n]

    # reshape
    depth_ = np.empty([480, 640])
    depth_[:,:] = depth[:,:].T
    
    depth__ = depth_.astype('float32')
    depths_conv.append(depth__)
    depth_file = open(os.path.join(data_path, group, 'depths', '{:05d}.p'.format(index)), 'wb')
    pickle.dump(depth__, depth_file)
    depth_file.close()
