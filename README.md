# Deep-Learning-Supermarket
This project tries to understand the position of a person in a supermarket using image classification and deep learning.
The images are taken from a [dataset](https://iplab.dmi.unict.it/MLC2018/). These images has been taken following some routes inside the supermarket. After that they have been divided into 16 classes.  

![image](img/ROUTE.png)

The feature extraction part will be perfomed using three pretrained networks:
* AlexNet
* ResNet18
* VGG16
The classification part will be perfomed using linear SVMs.

# Index

# Dataset Organization
The first part is to organize the images into folders. The zip file is divided into a folder that contains all the images and three csv files:
* **training_set**: that contains all the images to use to train the knn
* **validation_set**: that contains all the images to use to verify the accuracy
* **test_set**: that contains all the images to use to run the algorithm

The first two csv files have 6 columns:
* Image name
* x coordinate
* y coordinate
* u coordinate
* z coordinate
* Class

We used just the first file and we removed all the coordinates columns, because we don't need the position in which the photo was taken, we need just the name of the file and the class.

Using a bash script file we divided all the images into folders from 00 to 15 based on their class.

The images folder won't be in this repository because the dimension is too high for github.

# How the project works

## First step
The first step is to use AlexNet, ResNet and VGG16 to extract the features. In the code there is a part in which is possible to select which neural network use:
```
net = alexnet;
% net = resnet18;
% net = vgg16;
```

# How to run the project
Install the matlab Deep Learning Toolbox Add On: Home > Add-On > Deep Learning Toolbox