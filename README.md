## CelebA dataset training using DCGAN, or Deep Convolutional GAN, a generative adversarial network architecture in Pytorch

## Overview

This project implements a Generative Adversarial Network (GAN) using PyTorch. The GAN is trained on the CelebA dataset to generate realistic images of human faces. The model consists of two neural networks: a Generator that generates fake images, and a Discriminator that tries to distinguish between real and fake images. Both networks are trained together in a competitive setting where the Generator tries to fool the Discriminator, and the Discriminator tries to correctly classify real and fake images.

## Requirements

This script requires the following libraries:

* Python 3.6+
* PyTorch
* Torchvision
* Numpy
* Matplotlib
* Google Colab (if running on Colab)
* A GPU is highly recommended for faster training.

## Setup

* Download CelebA Dataset: The script downloads the CelebA dataset, which consists of 202,599 images of celebrity faces.
* Preprocessing: The images are resized to 64x64 pixels, normalized to a range of (-1, 1), and loaded into a PyTorch DataLoader for batching.

## Key Hyperparameters

* batch_size: Number of images per batch (default is 128).
* image_size: The resolution of images used for training (default is 64x64 pixels).
* nz: Dimensionality of the latent space (default is 100).
* ngf: Feature map size in the generator.
* ndf: Feature map size in the discriminator.
* num_epochs: Number of epochs to train the models (default is 5).
* lr: Learning rate for the Adam optimizer (default is 0.0002).
* beta1: Hyperparameter for Adam optimizer (default is 0.5).

## Model Architecture

* Generator
The Generator takes in random noise (latent vector) and generates a 64x64 pixel image. It consists of several layers of transposed convolution and batch normalization, followed by ReLU activation. The final layer uses the Tanh activation function to generate pixel values between -1 and 1.

* Discriminator
The Discriminator is a CNN that takes a 64x64 image as input and outputs a probability score, indicating whether the image is real or fake. The model consists of convolutional layers, batch normalization, and LeakyReLU activation, ending with a Sigmoid activation function to output a probability score.

## Training Loop

The training follows the typical GAN approach:

1. Train the Discriminator:

    * The Discriminator is trained to maximize the log-likelihood of correctly identifying real images and minimize the likelihood of classifying fake images as real.
    * Real and fake images are passed through the Discriminator, and the loss is calculated using binary cross-entropy.
    * The Discriminator's gradients are updated after processing both real and fake images.

2. Train the Generator:

    * The Generator is trained to maximize the Discriminator's likelihood of classifying fake images as real.
    * Fake images are generated from random noise, passed through the Discriminator, and the Generator's loss is calculated.
    * The Generator's gradients are updated based on this feedback.

3. Loss Tracking:

    * The losses for both Generator and Discriminator are stored and plotted at the end of training for visualization.

4. Image Generation:

    * The model generates new images from the latent space at regular intervals to track the progress of the Generator.

## Results

* The script generates plots showing the loss for both the Generator and Discriminator during training.
* A side-by-side comparison of real and fake images is shown to visualize the Generator’s performance.
## How to Run

1. Run in Colab:

    * The script includes commands to mount Google Drive and copy the CelebA dataset into your working directory.
    * Execute the entire script in Google Colab for training the GAN.

2. Run Locally:

    * Download the CelebA dataset and place it in the directory specified by dataroot.
    * Make sure to have all the required libraries installed.
Run the script and ensure that a GPU is available for faster training.

## Visualization

* The script uses Matplotlib to plot the loss of the Generator and Discriminator over time.
* A Matplotlib animation is created to visualize how the generated images evolve during training.

## Conclusion

This project demonstrates how to build and train a GAN to generate images. It uses PyTorch for model building and training and requires a GPU for efficient training. The script is highly customizable and can be adapted to other datasets or different GAN architectures.