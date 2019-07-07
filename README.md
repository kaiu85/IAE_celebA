# IAE_celebA

Implementation of an Implicit Autoencoder following Makhzani (2018, https://arxiv.org/abs/1805.09804).

Build using Python 3, PyTorch. Tested using python 3.6.8, pytorch 1.0.1, torchvision 0.2.1, numpy 1.16.2, matplotlib 3.0.2, imageio 2.3.0.

CelebA Dataset was downloaded from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Images were preprocessed and rescaled to 96x96 pixels following: https://mlnotebook.github.io/post/GAN2/

Architecture of the encoder and generator network, plotting and training functions follow this implementation of DCGANs: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN

To start training the model just run 

python IAE_release.py

A log-file showing the relevant discriminator, regularizer, encoder and generator losses will be created, as well as .png files showing samples, reconstructions and interpolations from the current model. 
