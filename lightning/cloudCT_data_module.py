import os
import numpy as np
import torchvision
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import data_generators.cloudCT_dataset as cloudCT_dataset
from data_generators.cloudCT_dataset import CloudCTdataset


class CloudCTDataModule(pl.LightningDataModule):
    def __init__(self, config, train_csv_path, test_csv_path, batch_size,
                 num_workers, val_length=0.2, test_length=0.2):
        super().__init__()
        self.config = config
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_length = val_length
        self.test_length = test_length

    def calc_dataset_mean_and_std(self, dataset):
        data = []
        size = len(dataset)
        for i in range(size):
            data.append(dataset[i]['image'])

        R_mean = np.mean(data[:][:][:][0])
        G_mean = np.mean(data[:][:][:][1])
        B_mean = np.mean(data[:][:][:][2])

        R_std = np.std(data[:][:][:][0])
        G_std = np.std(data[:][:][:][1])
        B_std = np.std(data[:][:][:][2])

        mean = (R_mean, G_mean, B_mean)
        std = (R_std, G_std, B_std)

        return mean, std

    def setup(self, stage=None):
        # Step 1. Load Dataset

        csv_train_file = os.path.join(self.config['dataset']['base_path'], 'dataset/CloudCT_train_dataset.csv')
        csv_test_file = os.path.join(self.config['dataset']['base_path'], 'dataset/CloudCT_test_dataset.csv')

        if stage == 'fit' or stage is None:
            trainable_dataset = CloudCTdataset(csv_file=csv_train_file)

            train_size = int(len(trainable_dataset) * 0.8)
            val_size = len(trainable_dataset) - train_size
            len_of_sets = [train_size, val_size]
            train_subset, validation_subset = random_split(trainable_dataset, len_of_sets)

            train_set = CloudCTdataset(csv_file=csv_train_file, subset=train_subset)

            mean, std = self.calc_dataset_mean_and_std(train_set)

            train_transformations_list = [cloudCT_dataset.Resize(), cloudCT_dataset.ToTensor(), cloudCT_dataset.Zoom(),
                                          cloudCT_dataset.Rotate(), cloudCT_dataset.HorizontalFlip(),
                                          cloudCT_dataset.VerticalFlip(), cloudCT_dataset.AffineTransform(),
                                          cloudCT_dataset.GaussianAugmentation(),
                                          cloudCT_dataset.Normalize(mean=mean, std=std)]
            train_transforms = torchvision.transforms.Compose(train_transformations_list)

            self.train_set = CloudCTdataset(csv_file=csv_train_file, subset=train_subset, transform=train_transforms)
            self.validation_set = CloudCTdataset(csv_file=csv_train_file, subset=validation_subset, transform=train_transforms)

        if stage == 'test' or stage is None:
            test_transformations_list = [cloudCT_dataset.Resize(), cloudCT_dataset.ToTensor(),
                                         cloudCT_dataset.Normalize(mean=mean, std=std)]
            test_transforms = torchvision.transforms.Compose(test_transformations_list)

            self.test_set = CloudCTdataset(csv_file=csv_test_file, transform=test_transforms)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.config['training']['batch_size'], shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.validation_set, batch_size=self.config['training']['batch_size'], shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.config['training']['batch_size'], shuffle=False, drop_last=True)

