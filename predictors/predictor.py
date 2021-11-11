"""
Based on the work of Giovanni Guidi:
'DeepLab V3+ Network for Semantic Segmentation' - https://github.com/giovanniguidi/deeplabV3-PyTorch

Modified by Amit Mizrahi & Amit Ben-Aroush
Project Cloud Segmentation in VISL (Vision and Image Sciences Laboratory)
Technion Institute of Technology

The Predictor class is implemented here.
In the main function, call the 'inference_on_test_set' function to perform test.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from models.deeplab import *

from utils.datagen_utils import denormalize_image
#from deeplab_model.utils.plot_utils import centroid_histogram, mask_and_downsample, get_average_color, normalize_colors
from data_generators.dataloader import initialize_data_loader
from utils.metrics import Evaluator
from tqdm import tqdm
from losses.loss import SegmentationLosses


class Predictor:
    def __init__(self, config,  checkpoint_path='./experiments/checkpoint_best.pth.tar'):
        self.config = config
        self.checkpoint_path = checkpoint_path

        self.categories_dict = {"sky": 0, "clouds": 1, "other": 2}

        self.categories_dict_rev = {v: k for k, v in self.categories_dict.items()}
        
        self.model = self.load_model()
        self.model = self.model.double()
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.mean, self.std = initialize_data_loader(config)

        self.evaluator = Evaluator(self.nclass)
        self.criterion = SegmentationLosses(weight=None, cuda=self.config['network']['use_cuda']).build_loss(mode=self.config['training']['loss_type'])


    def load_model(self):
        model = DeepLab(num_classes=self.config['network']['num_classes'], backbone=self.config['network']['backbone'],
                        output_stride=self.config['image']['out_stride'], sync_bn=False, freeze_bn=True)


        if self.config['network']['use_cuda']:
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location={'cuda:0': 'cpu'})

#        print(checkpoint)
#         model = torch.nn.DataParallel(model)

        model.load_state_dict(state_dict=checkpoint['state_dict'], strict=False)

        return model.cuda()

    def inference_on_test_set(self):
        print("inference on test set")

        self.model = self.model.double()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            target = torch.squeeze(target, 1)  # remove ch dimensions
            if self.config['network']['use_cuda']:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image.double())
            loss = self.criterion(output.cuda(), target)
            test_loss += loss.item()
            test_loss_norm = test_loss / (i + 1)
            tbar.set_description('Test loss: %.3f' % test_loss_norm)
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
            #     target_path = 'dataset/visualize_res/test_target' + str(j) + '_batch' + str(
            #         i) + '.png'
            #     cv2.imwrite(target_path, targ_save_255)
            #     im_i = pred[j]  # .permute(1,2,0).cpu().numpy()
            #     im_save = np.zeros_like(im_i)
            #     im_save[:,:] = im_i[:,:]
            #     im_save[im_save == 1] = 130
            #     im_save[im_save == 2] = 255
            #     im_save_255 = im_save
            #     image_path = 'dataset/visualize_res/test_pred' + str(j) + '_batch' + str(i) + '.png'
            #     cv2.imwrite(image_path, im_save_255)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print("Accuracy:{}, mean IoU:{}".format(Acc, mIoU))
        test_loss_norm = test_loss/(len(self.test_loader.sampler)/self.config['training']['batch_size'])
        print('Loss: %.3f' % test_loss_norm)


# # if we want to predict on ONE image, change this function
#     def segment_image(self, filename):
#
# #        file_path = os.path.join(dir_path, filename)
#         img = Image.open(filename).convert('RGB')
#
#         sample = {'image': img, 'label': img}
#
#         sample = DeepFashionSegmentation.preprocess(sample, crop_size=513)
#         image, _ = sample['image'], sample['label']
#         image = image.unsqueeze(0)
#
#         with torch.no_grad():
#             prediction = self.model(image)
#
#         image = image.squeeze(0).numpy()
#         image = denormalize_image(np.transpose(image, (1, 2, 0)))
#         image *= 255.
#
#         prediction = prediction.squeeze(0).cpu().numpy()
#
# #        print(prediction[])
#
#         prediction = np.argmax(prediction, axis=0)
#
#         return image, prediction