"""
Written by Amit Mizrahi & Amit Ben-Aroush
Project Cloud Segmentation in VISL (Vision and Image Sciences Laboratory)
Technion Institute of Technology


This is the data generator.
Here we load the raw data that is received from the manual classification program,
pre-process the images and create masks for them and save images and masks as pkl
files to dataset directory.
After all dataset is created, a list of it is created in a csv file.

It is possible to add new data to existing dataset, by calling this module with
the flag '--additional_dataset'.
For new dataset from zero, use the flag '--new_dataset'.

In general , us the following command to run the script:

python data_generator.py -c ../configs/config.yml --new_dataset
"""


import _pickle as cPickle
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import yaml
import cv2




def main(is_additional_dataset, config):

    # create readable pkls
    decode_pkl_files(config)

    # create images and masks, and save them as pkl files in a target directory
    preprocess_data(is_additional_dataset, config)

    # create csv file containing a list of the images & labels
    create_csv(config)

    # for debug - display image & mask
    # display_image_from_csv(config, 172)
    print('done')


def decode_pkl_files(config):
    # the following path is to the directory that holds the data extracted from the
    # cameranetwork component
    raw_dataset_path = os.path.join('..', config['dataset']['base_path'], 'reconstructions')
    for date_dir in os.listdir(raw_dataset_path):
        # load data from this date and encode it
        path_pkl = str(raw_dataset_path) + '/' + str(date_dir) + '/export_data.pkl'
        with open(path_pkl, 'rb') as data_pkl:
            data = cPickle.load(data_pkl, encoding="bytes")

        decoded_path = os.path.join('..', config['dataset']['base_path'], 'decoded_pkls')

        # dump the data to the file export_data_v3.pkl after encoding
        path_new_pkl = decoded_path + '/' + str(date_dir) + '.pkl'
        with open(path_new_pkl, "wb") as data_new_pkl:
            cPickle.dump(data, data_new_pkl)


def preprocess_data(is_additional_dataset, config):
    csv_path = os.path.join('..', config['dataset']['base_path'], 'dataset/CloudCT_full_dataset.csv')
    decoded_pkl_path = os.path.join('..', config['dataset']['base_path'], 'decoded_pkls')
    image_path = os.path.join('..', config['dataset']['base_path'], 'dataset/images')
    mask_path = os.path.join('..', config['dataset']['base_path'], 'dataset/masks')
    image_jpg_path = os.path.join('..', config['dataset']['base_path'], 'dataset/database/images')
    mask_jpg_path = os.path.join('..', config['dataset']['base_path'], 'dataset/database/masks')

    # names of images and labels are defined by index
    if not is_additional_dataset:
        idx = 0
    else:  # if data is added to the dataset, start index from last image saved
        dataset = pd.read_csv(csv_path)
        dataset = dataset.drop('Unnamed: 0', axis=1)
        idx = dataset.shape[0]

    # for debug:
    # data_detailes = pd.DataFrame(
    #     columns=['Name', 'Image size', 'Mask size', 'Image max', 'Image min', 'Mask max', 'Mask min'])

    for file in os.listdir(decoded_pkl_path):
        with open(decoded_pkl_path + '/' + file, 'rb') as data_pkl:
            data = cPickle.load(data_pkl)

        for cam in data:
            camera = data[cam]
            RedIm, GreenIm, BlueIm = camera[b'R'], camera[b'G'], camera[b'B']
            t = 1200
            RedIm[RedIm>t] = t
            GreenIm[GreenIm>t] = t
            BlueIm[BlueIm>t] = t
            RedIm_norm = 255*RedIm/t
            GreenIm_norm = 255*GreenIm/t
            BlueIm_norm = 255*BlueIm/t
            image = np.array([RedIm_norm, GreenIm_norm, BlueIm_norm], dtype=np.uint8).transpose(1,2,0)

            # extract masks
            cloudmask = camera[b'cloud_mask']
            rect_sunmask = camera[b'rect_sun_mask']
            mask = camera[b'MASK']

            # create binary 'other' label
            binary_mask = (mask == 1)
            binary_mask = binary_mask.astype(np.int)

            # create binary 'cloud' mask
            binary_cloudmask = cloudmask > 0  # 'cloud'=1, 'sky'=0
            binary_cloudmask = binary_cloudmask.astype(np.int)

            # create 'other' label mask
            othermask = 2*(1-(binary_mask*(1-rect_sunmask)))  # 'other'=2

            cloudmask_no_sun = binary_cloudmask*(1-rect_sunmask)

            ground_truth = cloudmask_no_sun + othermask
            ground_truth[ground_truth > 2] = 1

            cam_name = str(cam).split('\'')[1]
            date = str(file).split('.')[0]
            if cam_name != 'cam_119bL':
                # optional: name sample with incremental index
                # idx = idx + 1
                # index = '{0:04d}'.format(idx)
                # image_name = image_path + '/' + str(index) + '.pkl'
                # mask_name = mask_path + '/' + str(index) + '.pkl'
                image_name = image_path + '/image_' + date + '_' + cam_name + '.pkl'
                mask_name = mask_path + '/mask_' + date + '_' + cam_name + '.pkl'
            else:
                # images from camera 119 go to different directory for now
                image_name = '../dataset/defected_images/images/image_' + date + '_' + cam_name + '.pkl'
                mask_name = '../dataset/defected_images/masks/mask_' + date + '_' + cam_name + '.pkl'

            # save image and mask to .pkl files
            with open(image_name, "wb") as data_new_pkl:
                cPickle.dump(image, data_new_pkl)
            with open(mask_name, "wb") as data_new_pkl:
                cPickle.dump(ground_truth, data_new_pkl)

            # save image and mask to .jpg files
            # image_name = image_jpg_path + '/' + date + '_' + cam_name + '.jpg'
            # cv2.imwrite(image_name, image)
            # mask_name = mask_jpg_path + '/' + date + '_' + cam_name + '.jpg'
            # cv2.imwrite(mask_name, ground_truth)

            # for debug:
            # data_detailes = data_detailes.append(
            #     other={'Name': str(idx), 'Image size': image.shape, 'Mask size': ground_truth.shape, 'Image max': str(np.max(image)),
            #            'Image min': str(np.min(image)), 'Mask max': str(np.max(ground_truth)), 'Mask min': str(np.min(ground_truth))}, ignore_index=True)

    # data_detailes.to_csv(os.path.join('..', config['dataset']['base_path'], 'dataset/dataset_detailes.csv'))


def create_csv(config):

    original_dataset = pd.DataFrame(columns=['Name', 'Image Path', 'Mask Path'])

    csv_path = os.path.join('..', config['dataset']['base_path'], 'dataset/CloudCT_full_dataset.csv')
    csv_train_path = os.path.join('..', config['dataset']['base_path'], 'dataset/CloudCT_train_dataset.csv')
    csv_test_path = os.path.join('..', config['dataset']['base_path'], 'dataset/CloudCT_test_dataset.csv')
    images_path = os.path.join(config['dataset']['base_path'], 'dataset/images')
    masks_path = os.path.join(config['dataset']['base_path'], 'dataset/masks')

    # maybe problematic images:
    problematic_ims = ['0172.pkl']

    # go over all images and add to dataframe
    for file in os.listdir(os.path.join('..', images_path)):
        if str(file) in problematic_ims:
            continue
        original_dataset = original_dataset.append(other={'Name': str(file), 'Image Path': images_path + '/' + str(file),
                                                          'Mask Path': masks_path + '/' + str(file)}, ignore_index=True)

    # write dataframe to csv
    original_dataset.to_csv(csv_path)

    # split to train & test
    train_data = original_dataset.sample(frac=0.8)
    test_data = original_dataset.drop(train_data.index)

    train_data.to_csv(csv_train_path)
    test_data.to_csv(csv_test_path)


def display_image_from_csv(config, im_idx):
    csv_path = os.path.join('../', config['dataset']['full_dataset_csv'])
    dataset = pd.read_csv(csv_path)
    dataset = dataset.drop('Unnamed: 0', axis=1)

    im_path = os.path.join('..', dataset.at[im_idx - 1, 'Image Path'])
    mask_path = os.path.join('..', dataset.at[im_idx - 1, 'Mask Path'])

    with open(im_path, 'rb') as im_pkl:
        image = cPickle.load(im_pkl)
    with open(mask_path, 'rb') as mask_pkl:
        mask = cPickle.load(mask_pkl)

    # plot separately:
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    # plot together:
    plt.figure()
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.show()
    # print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--additional_dataset', action='store_true', help='Add new images to existing data')
    parser.add_argument('--new_dataset', action='store_true', help='Create dataset from zero')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)

    args = parser.parse_args()

    if args.additional_dataset:  # dataset exist and the new data is additional
        additional_dataset = True
    elif args.new_dataset:  # dataset doesn't exist, create from zero
        additional_dataset = False

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(additional_dataset, config)
