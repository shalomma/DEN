import os
import skimage.io as io
import numpy as np
import h5py
import pickle


# data path
data_path = './data/'
data_file = 'nyu_depth_v2_labeled.mat'
# read mat file
f = h5py.File(os.path.join(data_path, data_file))

N = len(f['images'])
images_conv = []
depths_conv = []
for i in range(N):
    if i % 200 == 0:
        print('{}...'.format(i))
    
    img = f['images'][i]

    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T

    img__ = img_.astype('float32')
    images_conv.append(img__)
    img_file = open(os.path.join(data_path, 'raw_images', '{:05d}.p'.format(i)), 'wb')
    pickle.dump(img__, img_file)
    img_file.close()
    
    depth = f['depths'][i]

    # reshape for imshow
    depth_ = np.empty([480, 640])
    depth_[:,:] = depth[:,:].T
    
    depths_conv.append(depth_)
    depth_file = open(os.path.join(data_path, 'raw_depths', '{:05d}.p'.format(i)), 'wb')
    pickle.dump(depth_, depth_file)
    depth_file.close()