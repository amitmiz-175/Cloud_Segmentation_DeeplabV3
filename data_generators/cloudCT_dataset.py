"""
Written by Amit Mizrahi & Amit Ben-Aroush
Project Cloud Segmentation in VISL (Vision and Image Sciences Laboratory)
Technion Institute of Technology


The class 'CloudCTdataset' implements a dataset object, that inherits from
the torch.utils.data.Dataset class.
It uses a csv file to read the location of the images in the dataset.

In this file there's also implementations of a few augmentations that can be
used, and are called in the dataloader.
"""

import torch
import _pickle as cPickle
import pandas as pd
import numpy as np
import random
import torchvision.transforms.functional as TF


torch.manual_seed(17)
from torch.utils.data import Dataset


class CloudCTdataset(Dataset):

    def __init__(self, csv_file, subset=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            subset (Dataset): a subset of the dataset, for example - train set
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_map = pd.read_csv(csv_file)
        self.subset = subset
        self.transform = transform

    def __len__(self):
        if self.subset is None:
            return len(self.dataset_map)
        else:
            return len(self.subset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.subset is None:
            with open(self.dataset_map['Image Path'][idx], 'rb') as image_pkl:
                image = cPickle.load(image_pkl)
            with open(self.dataset_map['Mask Path'][idx], 'rb') as mask_pkl:
                mask = cPickle.load(mask_pkl)

            sample = {'image': image, 'mask': mask}
        else:
            sample = self.subset[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class Resize(object):
    """
    change image shape to 301x301 (the desired size).
    if size is smaller, zero padding is performed.
    """

    def __call__(self, sample, desired_size=301):
        image, mask = sample['image'], sample['mask']
        if image.shape[0] >= desired_size and image.shape[1] >= desired_size:
            new_im = image[0:desired_size, 0:desired_size, :]
            new_mask = mask[0:desired_size, 0:desired_size]
        elif image.shape[0] < desired_size <= image.shape[1]:
            new_im = np.zeros((desired_size, desired_size, 3))
            new_im[:image.shape[0], :, :] = image
            new_mask = np.zeros((desired_size, desired_size))
            new_mask[:image.shape[0], :] = mask
        elif image.shape[0] >= desired_size > image.shape[1]:
            new_im = np.zeros((desired_size, desired_size, 3))
            new_im[:, :image.shape[1], :] = image
            new_mask = np.zeros((desired_size, desired_size))
            new_mask[:, :image.shape[1]] = mask
        elif image.shape[0] < desired_size and image.shape[1] < desired_size:
            new_im = np.zeros((desired_size, desired_size, 3))
            new_im[:image.shape[0], :image.shape[1], :] = image
            new_mask = np.zeros((desired_size, desired_size))
            new_mask[:image.shape[0], :image.shape[1]] = mask
        return {'image': new_im, 'mask': new_mask}


class Normalize(object):
    """
    conversion of an image and a mask from numpy.ndarray to torch.Tensor
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image_norm = TF.normalize(image, mean=self.mean, std=self.std)
        return {'image': image_norm, 'mask': mask}


class ToTensor(object):
    """
    conversion of an image and a mask from numpy.ndarray to torch.Tensor
    """

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image_tensor = TF.to_tensor(image)
        mask_tensor = TF.to_tensor(mask)
        return {'image': image_tensor, 'mask': mask_tensor}


class Zoom(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        final_size = [301, 301]
        crop_range = [200, 300]

        crop_size = random.randint(crop_range[0], crop_range[1])
        cropped_image = TF.center_crop(image, crop_size)
        cropped_mask = TF.center_crop(mask, crop_size)
        zoomed_image = TF.resize(cropped_image, final_size)
        zoomed_mask = TF.resize(cropped_mask, final_size)
        return {'image': zoomed_image, 'mask': zoomed_mask}


class Rotate(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        rotation_angles = [90, 180, 270]

        angle = random.choice(rotation_angles)
        rotate_image = TF.rotate(image, angle)
        rotate_mask = TF.rotate(mask, angle)
        return {'image': rotate_image, 'mask': rotate_mask}


class HorizontalFlip(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        hflipped_image = TF.hflip(image)
        hflipped_mask = TF.hflip(mask)
        return {'image': hflipped_image, 'mask': hflipped_mask}


class VerticalFlip(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        vflipped_image = TF.vflip(image)
        vflipped_mask = TF.vflip(mask)
        return {'image': vflipped_image, 'mask': vflipped_mask}


class AffineTransform(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        affine_angle = [-180, -90, 90, 180]
        affine_shear_angle = [0,1,2,3]
        affine_translation = (1, 20)

        rotation_angle = random.choice(affine_angle)
        shear_angle = random.choice(affine_shear_angle)
        translation = (random.randint(affine_translation[0], affine_translation[1]),
                       random.randint(affine_translation[0], affine_translation[1]))
        affined_image = TF.affine(img=image, angle=rotation_angle, translate=translation, scale=1.0, shear=shear_angle)
        affined_mask = TF.affine(img=mask, angle=rotation_angle, translate=translation, scale=1.0, shear=shear_angle)
        return {'image': affined_image, 'mask': affined_mask}


class GaussianAugmentation(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        sigma = (random.randint(1, 100)) / 100
        gaussian_image = TF.gaussian_blur(img=image, kernel_size=5, sigma=sigma)
        return {'image': gaussian_image, 'mask': mask}



