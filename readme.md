# Cloud Segmentation using DeeplabV3+

###  About the project
This project is a final project of two students for electrical engineering in the Technion institute of technologies.  
The project is a part of VISL (Vision and Image Sciences Laboratory) research.  
The project goal is cloud identification and semantic segmentation using a neural network.  
The task is to classify each pixel in an image to one out of three classes: clouds, sky, others.

The raw data is unique and was purchased by the lab, using a set of fish-eye cameras located in different location in Haifa.  
The data was pre-processed manually using the Cameranetwork GUI created by Amit Aides. For more information about the cameras see <add link to paper>.  
The pre-processed data channels were concatenated to RGB images and masks were created accordingly. Each image has a ground truth mask composed from 3 different masks:
  1. Sun gaussian mask + sun block
  2. Sun shader binary mask
  3. Clouds probability mask
<p align="center">
  <img src="https://github.com/giovanniguidi/deeplabV3_Pytorch/blob/master/docs/deeplab.png">
  <i>Fig. 1: DeepLabV3+ model (source Chen et al. 2018)</i>
</p>
  <masks images + gound truth>
  
The final dataset - CloudCT - includs 315 RGB images and their masks of size 301x301. 
*insert image example*

The chosen model for the task is DeeplabV3+ network. <link to deeplab paper>

The project code is based on the work of Giovanni Guidi - [DeepLab V3+ Network for Semantic Segmentation](https://github.com/giovanniguidi/deeplabV3-PyTorch) and was modified to fit the project task.
Modules of data process and data loading were added.
  
### Results
The performance of the network over the dataset is:
  
_**Train set results**_  
Accuracy: 0.969  
Loss: 0.019  
  
_**Test set results**_  
Accuracy: 0.954  
Loss: 0.028  
<image of predictions>

  
### What operations can be performed?

First create a virtual environment:

    conda create --name cloudct --python==3.7.1  
    conda activate cloudct
  
#### DATA GENERATION
  
The datagenerator is compatible with raw data created by the Cameranetwork GUI.
For different dataset, a new datagenerator will need to be created. Make sure the output of the datagenerator is:
  1. Each image/mask is saves to a .pkl file as an ndarray.
  2. .csv files for train and test sets containing information about the dataset location, see example:
  <image of csv file>  
    
to operate the datagenerator of this project: 
    
    python data_generator.py -c ../configs/config.yml --new_dataset
  
#### TRAIN
    
The train parameters are configured in the config.yml file.
To perform train:
  
    python -c configs/config.yml --train

#### INFERENCE
    
To perform inference:  
    
    python -c configs/config.yml --predict_on_test_set
  
