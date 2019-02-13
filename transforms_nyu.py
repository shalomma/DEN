from torch import from_numpy
import numpy as np
from skimage import transform


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.output_size = (output_size, output_size)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        depth = depth[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'depth': depth}
    
    
class RandomRescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        if output_size < 350:
            print('Rescale output size is too small')

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        h, w = image.shape[:2]
        output_height =  np.random.randint(self.output_size - 100, self.output_size + 100)
        new_h, new_w = output_height, output_height * w / h

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='reflect', anti_aliasing=True)
        depth = transform.resize(depth, (new_h, new_w), mode='reflect', anti_aliasing=True)

        return {'image': img, 'depth': depth}

    
class Scale(object):
    """
    Rescale the image in a sample to a given size.

    """

    def __init__(self):
        pass
        

    def __call__(self, sample):
        
        img, depth = sample['image'], sample['depth']
        img = img.astype('float') / 255.0
        depth = depth.astype('float') / 10.0

        return {'image': img, 'depth': depth}

    
class RandomHorizontalFlip(object):
    """
    Horizontaly Flip the image in a given rate.

    """

    def __init__(self, rate):
        self.rate = rate
        

    def __call__(self, sample):
        
        img, depth = sample['image'], sample['depth']
        
        rnd = np.random.uniform(0,1)
        if rnd < self.rate:
            img = np.fliplr(img).copy()
            depth = np.fliplr(depth).copy()

        return {'image': img, 'depth': depth}

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        # rescale and flatten the depth map
        depth = transform.resize(depth, (25, 32), mode='reflect', anti_aliasing=True)
        depth = np.ravel(depth)
        
        return {'image': from_numpy(image),
                'depth': from_numpy(depth)}
