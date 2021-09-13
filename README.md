# Deep-Learning-Supermarket
The objective of this project is to try to understand the position of a person in a supermarket using image classification and deep learning.

The images of the supermaket are taken from a [dataset](https://iplab.dmi.unict.it/MLC2018/). These images have been taken by a camera attached to a cart that followed some routes inside the supermarket. After the acquisition they have been divided into 16 classes/routes.    

![image](img/pretrained/Routes.png)

With this dataset we will perform three different tests:
* Feature extraction using three pretrained networks:
  * AlexNet
  * ResNet18
  * VGG16  
and perform the classification using linear SVMs.
* Fine tuning using an ImageNet pretrained network
* Classification creating a new network

More info on the tests:
1. [Feature extraction using pretrained networks](https://github.com/thisispivi/Deep-Learning-Supermarket/blob/main/PRETRAINED.md)
1. [Fine-tuning](https://github.com/thisispivi/Deep-Learning-Supermarket/blob/main/FINETUNING.md)
1. [New network](https://github.com/thisispivi/Deep-Learning-Supermarket/blob/main/NEWNETWORK.md)


# Index
- [Files Structure](#files-structure)
- [Dataset Organization](#dataset-organization)
- [How to run the project](#how-to-run-the-project)
  * [Preliminary steps](#preliminary-steps)
  * [Dataset organization](#dataset-organization)
    + [Download our organized dataset](#download-our-organized-dataset)
    + [Download and manually organize the original dataset](#download-and-manually-organize-the-original-dataset)
  * [Variables configuration](#variables-configuration)
  * [Run the script](#run-the-script)

# Files Structure
```
.
|
| Folders
├── img   # Images for the readme
├── doc   # Report folder
│   └── Report.pdf   # Report
├── split  # Split folder
│   └── split.sh   # Split bash file
├── *TrainingSet*   # Folder with the images of the training set
│   ├── 00
│   ├── 01
│   ├── ...
│   ├── 14
│   └── 15
├── *TestSet*   # Folder with the images of the test set
│   ├── 00
│   ├── 01
│   ├── ...
│   ├── 14
│   └── 15
├── *ValidationSet*   # Folder with the images of the validation set
│   ├── 00
│   ├── 01
│   ├── ...
│   ├── 14
│   └── 15
|
| Markdown
├── README.md
├── PRETRAINED.md
├── FINETUNING.md
├── NEWNETWORK.md
|
| Liblinear
├── libsvmread.mexw64   # Read
├── libsvmwrite.mexw64   # Write
├── train.mexw64   # File with train function
├── predict.mexw64   # File with predict function
|
| Matlab Script
├── fine_tuning.mlx   # Script for the AlexNet fine tuning
├── new_network.mlx   # Script for new Network
└── pretrained_networks.mlx   # Script for the pretrained feature extraction
```

Folders with * are not included in the repository.

# Dataset Organization
The first step to do is to organize the images into folders. On the [site](https://iplab.dmi.unict.it/MLC2018/) it is possible to download the zip with the dataset. This file is a reduced version of another [dataset](https://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/) and is made by a folder that contains all the images and three csv files:
* **training_set**: that contains all the images to use to train the pretrained network
* **validation_set**: that contains all the images to use to verify the accuracy
* **test_set**: that contains all the images to use to run the algorithm

The first two csv files have 6 columns:
* Image name
* x coordinate
* y coordinate
* u coordinate
* z coordinate
* Class

The last csv file has just the image file names, so there are no labels.

Just the first two files have been used and all the coordinate columns have been removed, because the position in which the photo was taken it’s not necessary. So just the name of the file and the class have been used.

In both the training and the validation set, using a bash script file, all the images have been divided into folders from 00 to 15 based on their class.

The **test_set** csv file hasn’t been taken from the reduced dataset. So we downloaded the original dataset and we took the test set txt file. From this txt file we removed the coordinate columns and we used just the name of the file and the class. At the end, using a bash script file, we divided the images into folders from 00 to 15 based on their class, like we did for the validation and training set.

So there are two folders:
1. **TestSet**: in which there are all the test set images
1. **ValidationSet**: in which there are all the validation set images
1. **TrainingSet**: in which there are all the training set images

The images folder won't be in this repository because the size is too high for github.

The organized dataset that has been used can be downloaded [here](https://mega.nz/file/pQdQ0bgK#agEcbPofnVOqPUiarCcUngtgiKaYXXuK9N-_59YukXw).

# How to run the project
This section will explain how to run the project

## Preliminary steps

1. Clone the repository
```shell script
git clone https://github.com/thisispivi/Deep-Learning-Supermarket
```

2. Open one of the matlab script file:  
    1. ```pretrained_networks.mlx```: the file with the feature extraction using pretrained networks and with the classification using linear SVM
    1. ```fine_tuning.mlx```: the file with the fine tuning
    1. ```new_network.mlx```: the file with the new network


1. Install the matlab Statistic and Machine Learning Toolbox Add On: Home > Add-On > Statistic and Machine Learning Toolbox

2. Install the matlab Deep Learning Toolbox Model for AlexNet Network Add On: Home > Add-On > Deep Learning Toolbox Model for AlexNet Network

3. Install the matlab Deep Learning Toolbox Model for ResNet-18 Network Add On: Home > Add-On > Deep Learning Toolbox Model for ResNet-18 Network

4. Install the matlab Deep Learning Toolbox Model for VGG-16 Network Add On: Home > Add-On > Deep Learning Toolbox Model for VGG-16 Network

5. Install the Plot Confusion Matrix Add On: Home > Add-On > Plot Confusion Matrix by Vane Tshitoyan

## Dataset organization
Here there are two options:

### Download our organized dataset
Download the organized dataset we used from this [link](https://mega.nz/file/pQdQ0bgK#agEcbPofnVOqPUiarCcUngtgiKaYXXuK9N-_59YukXw).

### Download and manually organize the original dataset
1. Download the original [dataset](https://iplab.dmi.unict.it/MLC2018/)

1. Insert the images in the split folder

1. Run all the ```split.sh``` files

1. Move the Training, Validation and Test set folders in the root of the project

## Variables configuration
The next step is to configure the variables of the first section. 

## Run the script
The only thing left is to run the script


