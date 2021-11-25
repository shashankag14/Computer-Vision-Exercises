# Computer-Vision-Exercises

Exercises on High Level Computer Vision course as a part of master's degree program at Universität des Saarlandes, Germany. 

## Exercise 1 
It is based on familiarisation with basic image filtering routines. In the second part, a simple image querying system has been developed which accepts a query image as input and then finds a set of similar images in the database. In order to compare image some simple histogram based distance functions has been developed and evaluated their performance in combination with different image representations.

## Exercise 2
In this exercise, a first hands-on with neural network training has been experienced. Many frameworks (e.g. PyTorch, Tensorow, Cafe) allow easy usage of deep neural networks without precise knowledge on the inner workings of backpropagation and gradient descent algorithms. While these are very useful tools, it is important
to get a good understanding of how to implement basic network training from scratch, before using this libraries to speed up the process. For this purpose a simple two-layer neural network has been developed and its training algorithm based on back-propagation using only basic matrix operations. Further, PyTorch library has been used to do the same and understand the advantanges offered in using such tools.

As a benchmark to test the models, an image classiffcation task has been considered using the widely used CIFAR-10 dataset. The task is to code and train a
parametrised model for classifying those images.

## Exercise 3
A PyTorch implementation of a CNN to perform image classification and explore methods to improve the training performance and generalization of these networks.
CIFAR-10 dataset has been used as a benchmark for the networks similar to the previous exercise. Experiments have been done related to improving the results like Early Stopping, Dropout, Transfer Learning etc. 
