"""
Based on the work of Giovanni Guidi:
'DeepLab V3+ Network for Semantic Segmentation' - https://github.com/giovanniguidi/deeplabV3-PyTorch

Modified by Amit Mizrahi & Amit Ben-Aroush
Project Cloud Segmentation in VISL (Vision and Image Sciences Laboratory)
Technion Institute of Technology

The Trainer class is implemented here.
In the main function, call the 'training' or 'validation' function.
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_generators.dataloader import initialize_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import DeepLab
from losses.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import torch
import cv2


class Trainer(object):
    def __init__(self, config):

        self.config = config
        self.best_pred = 0.0

        # Define Saver
        self.saver = Saver(config)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.config['training']['tensorboard']['log_dir'])
        self.writer = self.summary.create_summary()
        
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.mean, self.std = initialize_data_loader(config)
        print(f'Train set size = {len(self.train_loader.dataset)}')
        print(f'Validation set size = {len(self.val_loader.dataset)}')
        print(f'Test set size = {len(self.test_loader.dataset)}')
        print(f'Full dataset size = {len(self.train_loader.dataset) + len(self.val_loader.dataset) + len(self.test_loader.dataset)}')

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=self.config['network']['backbone'],
                        output_stride=self.config['image']['out_stride'],
                        sync_bn=self.config['network']['sync_bn'],
                        freeze_bn=self.config['network']['freeze_bn'])

        train_params = [{'params': model.get_1x_lr_params(), 'lr': self.config['training']['lr']},
                        {'params': model.get_10x_lr_params(), 'lr': self.config['training']['lr'] * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=self.config['training']['momentum'],
                                   weight_decay=self.config['training']['weight_decay'], nesterov=self.config['training']['nesterov'])

        # Define Criterion
        # whether to use class balanced weights
        if self.config['training']['use_balanced_weights']:
            classes_weights_path = os.path.join(self.config['dataset']['base_path'], self.config['dataset']['dataset_name'] + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(self.config, self.config['dataset']['dataset_name'], self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=self.config['network']['use_cuda']).build_loss(mode=self.config['training']['loss_type'])
        self.model, self.optimizer = model, optimizer

        self.model = self.model.double()


        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.config['training']['lr_scheduler'], self.config['training']['lr'],
                                            self.config['training']['epochs'], len(self.train_loader))

        # Using cuda
        if self.config['network']['use_cuda']:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if self.config['training']['weights_initialization']['use_pretrained_weights']:
            if not os.path.isfile(self.config['training']['weights_initialization']['restore_from']):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.config['training']['weights_initialization']['restore_from']))

            if self.config['network']['use_cuda']:
                checkpoint = torch.load(self.config['training']['weights_initialization']['restore_from'])
            else:
                checkpoint = torch.load(self.config['training']['weights_initialization']['restore_from'], map_location={'cuda:0': 'cpu'})

            self.config['training']['start_epoch'] = checkpoint['epoch']

            if self.config['network']['use_cuda']:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

#            if not self.config['ft']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.config['training']['weights_initialization']['restore_from'], checkpoint['epoch']))

    def training(self, epoch):
        train_loss = 0.0
        self.model = self.model.double()
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            target = torch.squeeze(target, 1)  # remove ch dimensions
            if self.config['network']['use_cuda']:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image.double())
            pred = output.data.cpu().numpy()
            target_vis = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # save predictions:
            # for j in range(self.config['training']['batch_size']):
            #     targ_i = target_vis[j]
            #     targ_save = np.zeros_like(targ_i)
            #     targ_save[:, :] = targ_i[:, :]
            #     targ_save[targ_save == 1] = 130
            #     targ_save[targ_save == 2] = 255
            #     targ_save_255 = targ_save
            #     target_path = 'dataset/visualize_res/train_epoch' + str(epoch) + '_batch' + str(i) + '_target' + str(j) + '.png'
            #     cv2.imwrite(target_path, targ_save_255)
            #     pred_i = pred[j]
            #     pred_save = np.zeros_like(pred_i)
            #     pred_save[:, :] = pred_i[:, :]
            #     pred_save[pred_save == 1] = 130
            #     pred_save[pred_save == 2] = 255
            #     pred_save_255 = pred_save
            #     pred_path = 'dataset/visualize_res/train_epoch' + str(epoch) + '_batch' + str(i) + '_pred' + str(j) + '.png'
            #     cv2.imwrite(pred_path, pred_save_255)

            loss = self.criterion(output, target)  # avg loss over images on current batch
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() # cumulative loss of batches so far
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))  # this is the avg loss over batches
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_vis, pred)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.config['dataset']['dataset_name'], image, target, output, global_step)

        train_Acc = self.evaluator.Pixel_Accuracy()
        self.writer.add_scalar('train/total_loss_epoch', train_loss/(len(self.train_loader.sampler)/self.config['training']['batch_size']), epoch)
        self.writer.add_scalar('train/Acc', train_Acc, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config['training']['batch_size'] + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss/(len(self.train_loader.sampler)/self.config['training']['batch_size'])))
        print('train_Acc:{}'.format(train_Acc))

        # save last checkpoint
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            # 'state_dict': self.model.module.state_dict(),
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best = False, filename='checkpoint_last.pth.tar')

        # if training on a subset reshuffle the data
        # if self.config['training']['train_on_subset']['enabled']:
        #     self.train_loader.dataset.shuffle_dataset()

    def validation(self, epoch):
        self.model = self.model.double()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            target = torch.squeeze(target, 1)  # remove ch dimensions
            if self.config['network']['use_cuda']:
                image, target = image.cuda(), target.cuda()
                print(f'image type = {image.dtype}')
                print(f'model type = {self.model.dtype}')
            with torch.no_grad():
                output = self.model(image.double())
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # save predictions:
            # for j in range(self.config['training']['batch_size']):
            #     targ_i = target[j]
            #     targ_save = np.zeros_like(targ_i)
            #     targ_save[:, :] = targ_i[:, :]
            #     targ_save[targ_save == 1] = 130
            #     targ_save[targ_save == 2] = 255
            #     targ_save_255 = targ_save
            #     target_path = 'dataset/visualize_res/val_epoch' + str(epoch) + '_batch' + str(i) + '_target' + str(j) + '.png'
            #     cv2.imwrite(target_path, targ_save_255)
            #     pred_i = pred[j]
            #     pred_save = np.zeros_like(pred_i)
            #     pred_save[:, :] = pred_i[:, :]
            #     pred_save[pred_save == 1] = 130
            #     pred_save[pred_save == 2] = 255
            #     pred_save_255 = pred_save
            #     pred_path = 'dataset/visualize_res/val_epoch' + str(epoch) + '_batch' + str(i) + '_pred' + str(j) + '.png'
            #     cv2.imwrite(pred_path, pred_save_255)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        #Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        #FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss/(len(self.val_loader.sampler)/self.config['training']['batch_size']), epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        #self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        #self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config['training']['batch_size'] + image.data.shape[0]))
        print("val_Acc:{},  mIoU:{}".format(Acc, mIoU))
        # print("val_Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('val_Loss: %.3f' % (test_loss/(len(self.val_loader.sampler)/self.config['training']['batch_size'])))

        new_pred = mIoU
        if new_pred > self.best_pred:
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                # 'state_dict': self.model.module.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            },  is_best = True, filename='checkpoint_best.pth.tar')