# Deep-Learning-Supermarket
This project tries to understand the position of a person in a supermarket using image classification and deep learning.

The images are taken from a [dataset](https://iplab.dmi.unict.it/MLC2018/). These images has been taken by a camera attached to a supermarket cart that followed some routes inside the supermarket. After the acquisition they have been divided into 16 classes.  

![image](img/ROUTE.png)

With this dataset we will perform the feature extraction using three pretrained networks:
* AlexNet
* ResNet18
* VGG16

After that we will perfom the classification using linear SVMs.

# Index
- [Pretrained Networks](#pretrained-networks)
  * [AlexNet](#alexnet)
  * [ResNet18](#resnet18)
  * [VGG16](#vgg16)
- [Dataset Organization](#dataset-organization)
- [How the project works](#how-the-project-works)
  * [Import the dataset and split the training set](#import-the-dataset-and-split-the-training-set)
  * [Pretrained network selection](#pretrained-network-selection)
  * [Image resize](#image-resize)
  * [Select the activation layer for the feature extraction and extract the features](#select-the-activation-layer-for-the-feature-extraction-and-extract-the-features)
  * [Classification](#classification)
- [How to run the project](#how-to-run-the-project)
  * [Preliminary steps](#preliminary-steps)
  * [Pretrained network selection](#pretrained-network-selection-1)
  * [Run the script](#run-the-script)
- [Test and output analysis](#test-and-output-analysis)

# Pretrained Networks
In this section we will show which are the pretrained network that we used in this project.

## AlexNet
AlexNet is a convolutional neural network that is 8 layers deep. The pretrained version has more than a milion images from the [ImageNet](http://www.image-net.org) database. This pretrained network can classify images into 100 object categories, such as keyboard, mouse, pencil and many animals. The network has an image input size of 227x227.

To see the structure of the network in matlab you have to put these lines in the command window:
<details>
<summary>Expand</summary>

```
net = alexnet;
```
After
```
net.Layers;
OR
analyzeNetwork(net)
```
</details>
<br>

## ResNet18
The ResNet-18 is a convolutional neural network that is 18 layer deep. The pretrained version has more than a milion images from the [ImageNet](http://www.image-net.org) database. This pretrained network can classify images into 100 object categories, such as keyboard, mouse, pencil and many animals. The network has an image input size of 244x244.

To see the structure of the network in matlab you have to put these lines in the command window:
<details>
<summary>Expand</summary>

```
net = resnet18;
```
After
```
net.Layers;
OR
analyzeNetwork(net)
```
</details>
<br>

## VGG16
VGG16 is a convolutional neural network that is 16 layers deep. The pretrained version has more than a milion images from the [ImageNet](http://www.image-net.org) database. This pretrained network can classify images into 100 object categories, such as keyboard, mouse, pencil and many animals. The network has an image input size of 224x224.

To see the structure of the network in matlab you have to put these lines in the command window:
<details>
<summary>Expand</summary>

```
net = vgg16;
```
After
```
net.Layers;
OR
analyzeNetwork(net)
```
</details>
<br>

# Dataset Organization
The first step to do is to organize the images into folders. The zip file is divided into a folder that contains all the images and three csv files:
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
In this section we will explain how the project works.

## Import the dataset and split the training set
In the first part of the code we will import all the images using ```imageDataStore``` a function that automatically labels all the images based on the folder names. The images will be stored into an ```ImageDataStore``` object. After that we split each label into **training** and into **validation** set. We chose to split into 70% training and 30% test.

## Pretrained network selection
The next step is to choose which pretrained network use to extract the features In the code there is a part in which is possible to select which neural network use:
```
network = "alexnet";
% network = "resnet";
% network = "vgg"
```

## Image resize
The networks require different input size, in this section the image will be resized to fit the first input layer. To automatically resize the training and test images before they are input to the network, create augmented image datastores, specify the desired image size, and use these datastores as input arguments to activations.

## Select the activation layer for the feature extraction and extract the features
The network constructs a hierarchical representation of input images. Deeper layers contain higher-level features, constructed using the lower-level features of earlier layers. To get the feature representations of the training and test images, we will use activations on different layers depending on the network used. 

In our case for **alexnet** is **fc7**, for **resnet18** is **pool5** and for **vgg16** is **fc7**. This parameter can be changed. Basically we are extracting the feature from the layer before the layer that actually classify the things.

## Classification

# How to run the project
In this section we will explain how to run the project

## Preliminary steps
1. Install the matlab Deep Learning Toolbox Add On: Home > Add-On > Deep Learning Toolbox

2. Install the matlab Deep Learning Toolbox Model for AlexNet Network Add On: Home > Add-On > Deep Learning Toolbox Model for AlexNet Network

3. Install the matlab Deep Learning Toolbox Model for ResNet-18 Network Add On: Home > Add-On > Deep Learning Toolbox Model for ResNet-18 Network

4. Install the matlab Deep Learning Toolbox Model for VGG-16 Network Add On: Home > Add-On > Deep Learning Toolbox Model for VGG-16 Network

## Pretrained network selection
The next step is to choose which pretrained network use to extract the features In the code there is a part in which is possible to select which neural network use:
```
network = "alexnet";
% network = "resnet";
% network = "vgg"
```

## Run the script

# Test and output analysis
