# Deep-Learning-Models
# Comparative study of Different Deep Learning Models on CIFAR10 Dataset
This repository presents a comparative study of four deep learning architectures - LeNet, ResNet , VGG16 and Transformer - applied to CIFAR10 dataset for image classification task.

## Model Implemented

- LeNet : LeNet is one of the earliest convolutional neural network (CNN) architectures Objective of LeNet model is to evaluate and understand the performance of a classical convolutional neural network architecture on relatively complex, multi-class image classification task.
- ResNet : Its major innovation is the use of residual connections that allow signals to bypass one or more layers.
- VGG16 : The VGG-16 model is a convolutional neural network (CNN) architecture Its structured design increases depth progressively, enabling the network to capture complex patterns. It stacks multiple convolutional layers with small 3x3 filters each followed by ReLU activation.
- Transformer: A Transformer model is a deep learning architecture that leverages self-attention mechanisms to process and understand sequential data, excelling at tasks like natural language processing and computer vision. 


## Evaluation Metrics

- Accuracy

-Precision

-Recall
 

  ## Results Summary



| Model | Accuracy(%) | Precision(%) |  Recall(%)       |
| :----------- | :------- | :--------- | :------------------ |
| LeNet | 61.86 | 61.61 | 61.86 |
| ResNet| 83.28 | 83.25 | 83.28 |
| VGG16 | 62.46 | 62.66 | 61.46 |
| Transformer | 60.65 | 61.24 | 60.65 |


## Dataset

- CIFAR10 Dataset
- 60,000 images (32x32 pixels)
- Training set - 50,000
- Test set - 10,000
- Classes - 10 

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Libraries : Numpy,Matplotlib,Scikit-learn
------

Platform - Google Colab
-----
Submitted by - Shampy
