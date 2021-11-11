"""
Written by Amit Mizrahi & Amit Ben-Aroush
Project Cloud Segmentation in VISL (Vision and Image Sciences Laboratory)
Technion Institute of Technology


Here the dataloader is created.
The dataset is split into train, validation and test (and information of each
set is saved into a csv file.

There is also a function to calculate the normalization parameters of the dataset.
"""

import numpy as np
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import data_generators.cloudCT_dataset as cloudCT_dataset
from data_generators.cloudCT_dataset import CloudCTdataset


def main():
    train_loader, val_loader, test_loader, num_classes = initialize_data_loader()
    print("Dataloader finished!")


def initialize_data_loader(config):

    csv_train_file = os.path.join(config['dataset']['base_path'], 'dataset/CloudCT_train_dataset.csv')
    csv_test_file = os.path.join(config['dataset']['base_path'], 'dataset/CloudCT_test_dataset.csv')

    full_train_dataset = CloudCTdataset(csv_file=csv_train_file)
    train_size = int(len(full_train_dataset)*0.8)
    val_size = len(full_train_dataset) - train_size
    len_of_sets = [train_size, val_size]
    train_subset, validation_subset = random_split(full_train_dataset, len_of_sets)

    train_set = CloudCTdataset(csv_file=csv_train_file, subset=train_subset)

    mean, std = calc_dataset_mean_and_std(train_set)

    train_transform = T.Compose(
            [cloudCT_dataset.Resize(), cloudCT_dataset.ToTensor()])

    test_transform = T.Compose([cloudCT_dataset.Resize(), cloudCT_dataset.ToTensor()])

    train_set = CloudCTdataset(csv_file=csv_train_file, subset=train_subset, transform=train_transform)
    validation_set = CloudCTdataset(csv_file=csv_train_file, subset=validation_subset, transform=train_transform)
    test_set = CloudCTdataset(csv_file=csv_test_file, transform=test_transform)

    num_classes = 3
    train_loader = DataLoader(dataset=train_set, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=validation_set, batch_size=config['training']['batch_size'], shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=config['training']['batch_size'], shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, num_classes, mean, std


def calc_dataset_mean_and_std(dataset):
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


if __name__ == '__main__':
    main()
