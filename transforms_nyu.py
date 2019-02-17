from torch import from_numpy, stack
import numpy as np
from skimage import transform
from torchvision import transforms


class Normalize(object):
    """
    Rescale the image in a sample to a given size.

    """

    def __call__(self, sample):
        
        img, depth = sample['image'], sample['depth']
        img = img.astype('float') / 255.0
        depth = depth.astype('float') / 10.0

        return {'image': img, 'depth': depth}


class RandomCrop(object):
    """
    Crop randomly the image and depth map in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    
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

    
class RandomRotate(object):
    """
    Rotate randomly the image and depth map in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, max_angle=5):
        self.max_angle = max_angle


    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        angle = np.random.uniform(0, self.max_angle)
        image = transform.rotate(image, angle, resize=False,
                                 mode='reflect', cval=0, clip=True, preserve_range=False)
        depth = transform.rotate(depth, angle, resize=False,
                                 mode='reflect', cval=0, clip=True, preserve_range=False)

        return {'image': image, 'depth': depth}


class CenterCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size


    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        h, w = image.shape[:2]
        h_c, w_c = int(h/2), int(w/2)

        new_h, new_w = self.output_size
        half_new_h, half_new_w = int(new_h/2), int(new_w/2)

        image = image[h_c - half_new_h: h_c + half_new_h,
                      w_c - half_new_w: w_c + half_new_w]

        depth = depth[h_c - half_new_h: h_c + half_new_h,
                      w_c - half_new_w: w_c + half_new_w]

        return {'image': image, 'depth': depth}


class RandomRescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, max_rate=0.1):
        self.max_rate = max_rate

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        h, w = image.shape[:2]
        rate = np.random.uniform(0, self.max_rate)
        output_height = h * (1 + rate)
        new_h, new_w = output_height, output_height * w / h


        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='reflect', anti_aliasing=True)
        depth = transform.resize(depth, (new_h, new_w), mode='reflect', anti_aliasing=True)

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
        
        # flatten the depth map
        depth = np.ravel(depth)
        
        return {'image': from_numpy(image),
                'depth': from_numpy(depth)}

    
class ScaleDown(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']


        target_image_shape = (224, 224)
        image = transform.resize(image, target_image_shape, mode='reflect', anti_aliasing=True)
        
        # rescale and flatten the depth map
        target_depth_shape = (25, 32)
        depth = transform.resize(depth, target_depth_shape, mode='reflect', anti_aliasing=True)
        
        return {'image': image, 'depth': depth}    


class FDCPreprocess(object):
    """
    Preprocess images and depth for the FDC module
    images: 
    1. crop in four corners at a give ratio
    2. resize to (224, 224)
    3. cast to torch tensor and stack
    depths:
    1. resize to (25, 32)
    2. flatten
    3. to torch tensor
    """

    def __init__(self, crop_ratios):
        self.crop_ratios = crop_ratios

    def __call__(self, sample):
        img, depth = sample['image'], sample['depth']

        h, w, _ = img.shape
        four_crop = []
        for r in range(len(self.crop_ratios)):
            ratio = self.crop_ratios[r]
            h_crop, w_crop = [round(h * ratio), round(w * ratio)]
            for i in range(4):
                if i == 0:  # Top-left
                    crop = img[:h_crop, :w_crop]
                elif i == 1:  # Top-right
                    crop = img[:h_crop, -w_crop:]
                elif i == 2:  # Bottom-left
                    crop = img[-h_crop:, :w_crop]
                elif i == 3:  # Bottom-right
                    crop = img[-h_crop:, -w_crop:]

                crop = transform.resize(crop, (224, 224), mode='reflect',
                                        anti_aliasing=True, preserve_range=True).astype('float32')
                four_crop.append(crop)

        stacked_images = transforms.Lambda(lambda crops: stack([transforms.ToTensor()(c) for c in crops]))(four_crop)

        depth = transform.resize(depth, (25, 32), mode='reflect',
                                 anti_aliasing=True, preserve_range=True).astype('float32')
        depth = np.ravel(depth)
        depth = from_numpy(depth)

        return {'stacked_images': stacked_images, 'depth': depth}
