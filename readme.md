# Cloud Segmentation using DeeplabV3+

###  About the project
This project is a final project of two students from the [Electrical and Computer Engineering faculty in the Technion Institute of Technologies](https://ece.technion.ac.il/).  
The project is a part of [_**VISL**_](https://visl.technion.ac.il/) (Vision and Image Sciences Laboratory) research, and was supervised by Adi Vainiger.  
The project goal is cloud identification and semantic segmentation using a neural network.  
The task is to classify each pixel in an image to one out of three classes: clouds, sky, others.

The raw data is unique and was purchased by the lab, using a set of fish-eye cameras located in different location in Haifa.  
The data was pre-processed manually using the **Cameranetwork** GUI created by Amit Aides. For more information about the cameras see [Distributed Sky Imaging Radiometry and Tomography](https://github.com/amitmiz-175/Cloud_Segmentation_DeeplabV3/tree/master/docs/Distributed_Sky_Imaging_Radiometry_and_Tomography.pdf).  
The pre-processed data channels were concatenated to RGB images and masks were created accordingly. Each image has a ground truth mask composed from 3 different masks:
  1. Sun gaussian mask + sun block
  2. Sun shader binary mask
  3. Clouds probability mask

![masks from gui](/docs/masks_from_gui.png)
<p align="center">
  <i>Fig. 1: From the left: Sun shader mask, Sun mask, Sun block mask, Clouds probability mask</i>
</p>
  
The final dataset - CloudCT - includs 315 RGB images and their masks of size 301x301. 

![image + mask](/docs/image_and_mask.png)
<p align="center">
  <i>Fig. 2: A sample image and its ground truth</i>
</p>

The chosen model for the task is [DeeplabV3+](https://github.com/amitmiz-175/Cloud_Segmentation_DeeplabV3/tree/master/docs/deeplab.pdf) network.  

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

![test prediction](/docs/test_predictions.png)
<p align="center">
  <i>Fig. 3: Results of inference</i>
</p>


The checkpoints to recapture the results above can be found [here](https://drive.google.com/drive/folders/1cDQdH_Gua2WY0KowADI2k6cJyXM7eoe7?usp=sharing).  
After download, put the file in the _'experiments/'_ directory.
  
### What operations can be performed?

First create a virtual environment:

    conda create --name cloudct --python==3.7.1  
    conda activate cloudct

Also, create the following empty folders in the project directory according to the specified hierarchy:

    - decoded_pkls
    - experiments
    - reconstructions
    - dataset
        - database
            - images
        - defected_images
            - images
            - masks
        - images
        - masks

You should put the data in the _'reconstructions/'_ directory.
It should be organized in subdirectories named after the dates of the data, as it is saved from the Cameranetwork GUI.

#### DATA GENERATION
  
The datagenerator is compatible with raw data created by the **Cameranetwork** GUI.
For different dataset, a new datagenerator will need to be created. Make sure the output of the datagenerator is:
  1. Each image/mask is saves to a _.pkl_ file as an _ndarray_.
  2. _.csv_ files for train and test sets containing information about the dataset location, see example:
  
![csv example](/docs/csv_example.png)
<p align="center">
  <i>Fig. 4: An example for how the .csv file should look like</i>
</p>
    
to operate the datagenerator of this project: 
    
    python data_generator.py -c ../configs/config.yml --new_dataset
  
#### TRAIN
    
The train parameters are configured in the _config.yml_ file.
To perform train:
  
    python -c configs/config.yml --train

#### INFERENCE
    
To perform inference:  
    
    python -c configs/config.yml --predict_on_test_set
  

### Contacts

* Amit Mizrahi, github: [amitmiz-175](https://github.com/amitmiz-175) , mail: amitm175@gmail.com
* Amit Ben-Aroush, github: [amit-benaroush](https://github.com/amit-benaroush) , mail: amiti0108@gmail.com
* Adi Vainiger, github: [Addalin](https://github.com/Addalin) , mail: adi.vainiger@gmail.com


This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No 810370: CloudCT).
