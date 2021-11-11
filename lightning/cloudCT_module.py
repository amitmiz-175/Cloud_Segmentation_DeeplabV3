import os
import numpy as np
import torchvision
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from models.deeplab import DeepLab
from losses.loss import SegmentationLosses


class CloudCTModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = SegmentationLosses(cuda=self.config['network']['use_cuda']).build_loss(mode=self.config['training']['loss_type'])
        self.accuracy = torchmetrics.Accuracy()
        self.model = DeepLab(num_classes=config['network']['num_classes'],
                        backbone=config['network']['backbone'],
                        output_stride=config['image']['out_stride'],
                        sync_bn=config['network']['sync_bn'],
                        freeze_bn=config['network']['freeze_bn'])

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y = torch.squeeze(y, 1)
        y = y.type(torch.LongTensor)  # cross enthropy requires the target to be float
        y_hat = self.model(x.double())
        loss = self.criterion(y_hat, y.cuda())
        # loss = F.cross_entropy(y_hat, y.cuda())
        # accuracy = self.accuracy(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger

        # self.log('train_loss', loss, 'train_acc_step', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    # TODO: If you need to do something with all the outputs of each training_step, override training_epoch_end yourself
    # (if we want to plot results)
    # def training_epoch_end(self, training_step_outputs):
    #     for pred in training_step_outputs:
    #         # do something

    # def training_epoch_end(self):
    #     self.log('train_acc_epoch', self.accuracy.compute())

    # TODO: same for validation - to plot results, see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y = torch.squeeze(y, 1)
        y = y.type(torch.LongTensor)  # cross enthropy requires the target to be float
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.cuda())
        # loss = F.cross_entropy(y_hat, y.cuda())
        # accuracy = self.accuracy(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_loss', loss, 'valid_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val loss': loss}

    def validation_epoch_end(self, outputs):
        for dataset_result in outputs:
            self.logger.agg_and_log_metrics(
                dataset_result, step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        # loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config['training']['lr'])


"""
super()
"""


# def accuracy(self, logits, labels):
#     _, predicted = torch.max(logits.data, 1)
#     correct = (predicted == labels).sum().item()
#     accuracy = correct / len(labels)
#     return torch.tensor(accuracy)