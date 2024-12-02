# TensorFlow-and-Autoencoder-Deep-Learning-Project

This is the TensorFlow/Keras Assignment as part of the Masters in Artifical Intelligence Course. The assignment is specifically focused on building a model using TensorFlow’s low level API and directly using automatic differentiation.

Technologies and tools used: Python, ipynb file(Google Colab), tensorflow, matplotlib, Numpy

The Dataset 
The Fashion-MNIST dataset is a basic image dataset consisting of a training set of 60,000 examples 
and a test set of 10,000 examples. Each example is a 28x28 grayscale image (784 pixel values in 
total), associated with a label from 10 different classes.  
The following are the set of classes in this classification problem (the associated integer class label is 
listed in brackets).  
• T-shirt/top (0) 
• Trouser (1) 
• Pullover (2) 
• Dress  (3) 
• Coat (4) 
• Sandal (5) 
• Shirt (6) 
• Sneaker (7) 
• Bag (8) 
• Ankle boot (9) 

The objective is to build a vectorized implementation of a basic autoencoder neural network that will attempt to remove noise from images from the Fashion MNIST dataset (we will be artificially adding noise to the Fashion MNIST images). While there are more advanced autoencoder network architectures that can achieve better performance, we are going to focus on implementing the following basic network architecture. The model will take as input a noisy image from the Fashion MNIST dataset and will learn to output a denoised version of the image. As mentioned above, each image in Fashion MNIST is a 28x28 grayscale image (784 pixel values in total). 


The network architecture consists of the following densely connected layers: 


-> Layer 1: 128 Neurons with ReLu activations 


-> Layer 2: 64 Neurons with ReLu activations 


->Layer 3: 32 Neurons with ReLu activations 


-> Layer 4: 64 Neurons with ReLu activations 


->Layer 5: 128 Neurons with ReLu activations 


->Layer 6: 784 Neurons with Sigmoid activations


Autoencoder Implementation Requirements and Constraints
The project provides a low-level implementation of the above autoencoder using vectorization and TensorFlows gradient tape (the use of iterative for loops should be minimized). When training the model the code should first perform a forward pass by pushing the noisy feature training matrix (containing all noisy training feature data) through the layers of neurons and the final Sigmoid layer (using matrix multiplication, etc). You should use the Adam optimizer to update the trainable parameters. Where possible, your code should use Tensorflow functions and unless otherwise stated you should only use lower level TF functions (such as tf.matmul, tf.reduce_sum, etc). 
