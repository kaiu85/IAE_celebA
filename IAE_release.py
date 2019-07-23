import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy
from numpy.linalg import norm

######
###### NOTES
######
###### CelebA Dataset was downloaded from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
######
###### Images were preprocessed and rescaled to 96x96 pixels following: https://mlnotebook.github.io/post/GAN2/
######
###### Architecture of the network, plotting and training functions follow this awesome implementation
###### of DCGANS: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
######


######
###### Hyper-Parameters
######

# data parameters

img_size = 96 # image dimensions (quadratic images assumed)
data_dir = '/home/kai/bernstein/Deep_Learning/celebA/celebA/' # path to the preprocessed celebA dataset (c.f. ::::)
personal_dir = '/home/kai/bernstein/Deep_Learning/celebA/aligned_personal/'          # this path depends on your computer
n_save_images = 100 # save samples everyth n-th update step

# training parameters

batch_size = 128 # size of mini-batches
lr_disc = 2e-4 # learning rate for parameters of discriminator and regularizer
lr_gen = 2e-4 # learning rate for parameters of encoder and decoder
train_epoch = 200 # number of epochs to train
k_update = 10 # number of update steps for discriminator and regularizer per update step for encoder and decoder
regularize_R = False # Should the gradients of the regularizer be regularized to stabilize convergence (c.f. https://arxiv.org/abs/1801.04406)
reg_param_R = 0.1 # corresponding regularization weight
regularize_D = False # Should the gradients of the discriminator be regularized to stabilize convergence (c.f. https://arxiv.org/abs/1801.04406)
reg_param_D = 0.1 # corresponding regularization weight

upper_bound_D = 1.0 # discriminator network will be updated iteratively until the discriminator loss is below this
                    # threshold or k_update updates are reached
upper_bound_R = 0.5 # regularizer network will be updated iteratively until the discriminator loss is below this
                    # threshold or k_update updates are reached

clip_grad = True # should gradients be clipped
clip_norm = 0.25 # maximum value of the norm of gradients

# Model Parameters

n_global = 128 # dimensionality of latent space
n_hidden_regularizer = 2000 # dimensionality of hidden layer of regularizer
n_hidden_discriminator = 1000 # dimensionality of discriminator
model_dim = 128 # scale parameter used to control the number of channels used in the convolutional architecture
continue_training = False # False: Training starts from scratch and old checkpoints will be overwritten.
                          # True: the last checkpoint is loaded and training progresses from there. 
    
#####
##### Network-Architecture
#####


# Generator: Generates image samples conditioned on latent variable
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(n_global, d*8, 4, 1, 0)
        self.ln1 = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 0)
        self.ln2 = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 0)
        self.ln3 = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 0)
        self.ln4 = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d, 4, 2, 0)
        
        self.deconv6 = nn.ConvTranspose2d(d, 3, 3, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z):
        
        x = F.relu(self.ln1(self.deconv1(z)))
        x = F.relu(self.ln2(self.deconv2(x)))
        x = F.relu(self.ln3(self.deconv3(x)))
        x = F.relu(self.ln4(self.deconv4(x)))
        x = F.relu(self.deconv5(x))
        x = torch.tanh(self.deconv6(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        #self.ln1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.ln2 = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 0)
        self.ln3 = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 0)
        self.ln4 = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        
        self.fc1 = nn.Linear(n_global, d)
        
        #self.fc2 = nn.Linear(n_global, n_hidden_discriminator)
        #self.fc3 = nn.Linear(n_hidden_discriminator, 48*48)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x, z):
        z = z.view(-1, n_global)
        x = F.leaky_relu(self.conv1(x) + self.fc1(0.1*z).view(-1, model_dim, 1, 1), 0.2)
        #print(x.data.shape)
        #y = F.leaky_relu(self.fc2(z))
        #print(y.data.shape)
        #y = self.fc3(y).view(-1, 1, 48, 48)
        #print(y.data.shape)
        #x = x + y
        #print('puh')
        x = F.leaky_relu(self.ln2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.ln3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.ln4(self.conv4(x)), 0.2)
        x = self.conv5(x)
        
        #print(x.data.shape)

        return x
    
class encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.ln1 = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.ln2 = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 0)
        self.ln3 = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 0)
        
        self.conv5 = nn.Conv2d(d*8, n_global, 4, 1, 0)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x
    
class regularizer(nn.Module):
    # initializers
    def __init__(self):
        super(regularizer, self).__init__()
        self.fc1 = nn.Linear(n_global, n_hidden_regularizer)
        self.fc2 = nn.Linear(n_hidden_regularizer, n_hidden_regularizer)
        self.fc3 = nn.Linear(n_hidden_regularizer, 1)
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z):
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = self.fc3(z)
        
        return z

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
#####
##### Plotting Functions
#####
        
fixed_z_ = torch.randn((5 * 5, n_global)).view(-1, n_global, 1, 1).cuda()    # fixed latent state

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    
    G.train()
    
    if isFix:
        # Generate Samples from fixed latent vector
        test_images = G(fixed_z_)
    else:
        # Generate Samples from random latent vector
        z_ = torch.randn((5*5, n_global)).view(-1, n_global, 1, 1).cuda()
        test_images = G(z_)

    # Plot results
    fig, ax = plt.subplots(5, 5, figsize=(5, 5))
    for i, j in itertools.product(range(5), range(5)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path, dpi = 300)

    if show:
        plt.show()
    else:
        plt.close()

def show_encoding_and_interpolation(num_epoch, personal, train, show = False, save = False, path = 'result.png', isFix=False):
    
    # Get two out-of-training-sample images
    personal_iterator = iter(personal)
    xp, _ = next(personal_iterator)  
    xp = xp.cuda()
    
    # Get a minibatch from the training data (for Batchnorm)
    train_iterator = iter(train)
    x, _ = next(train_iterator)  
    x = x.cuda()
    
    # Set the first two images to the out-of-training-sample ones
    x[0:2] = xp
    
    # Encode
    E.train()
    z = E(x)
    E.train()
  
    z_n = z.data
    
    # Decode
    G.train()
    G_result = G(z)
    G.train()
    
    G_n = G_result.data
    
    # Plot original out-of-sample images and reconstruction
    fig, ax = plt.subplots(2, 2, figsize=(2, 2))
    for i, j in itertools.product(range(2), range(2)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    ax[0, 0].cla()
    ax[0, 0].imshow((x[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    
    ax[0, 1].cla()
    ax[0, 1].imshow((G_result[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    
    ax[1, 0].cla()
    ax[1, 0].imshow((x[1].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    
    ax[1, 1].cla()
    ax[1, 1].imshow((G_result[1].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig('just_encoding_' + path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()
    
    # Placeholder
    z_ = torch.randn((5*5, n_global)).view(-1, n_global)
    
    # Latent code for the two out-of-training-sample images
    z0 = z[0,:,0,0].detach()
    z1 = z[1,:,0,0].detach()
    
    # Interpolate between the two latent codes (c.f. https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/)
    for i in range(25):
        z_[i,:] = z0*numpy.sqrt(1.0 - float(i)/24.0) + z1*numpy.sqrt(float(i)/24.0) 
        
    z_ = z_.cuda().view(-1, n_global, 1, 1)
    
    # Generate samples from interpolated latent codes
    G.train()
    test_images = G(z_)
    G.train()
    
    # Plot results
    fig, ax = plt.subplots(5, 5, figsize=(5, 5))
    for i, j in itertools.product(range(5), range(5)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig('interp_' + path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()
        
    return z


#####
##### Load Datasets
#####

# Create Loaders for Image Datasets
# Converts input images to tensors and rescales color channels
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Loader for celebA training data
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)

# Check for correct image dimensions
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)
    
# Directory containing test images to be encoded and reconstructed    
dset = datasets.ImageFolder(personal_dir, transform)
personal_loader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=False)


#####
##### Initialize Networks
#####

G = generator(model_dim)
D = discriminator(model_dim)
E = encoder(model_dim)
R = regularizer()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
E.weight_init(mean=0.0, std=0.02)
R.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
E.cuda()
R.cuda()

#####
##### Load previous checkpoints
#####

if continue_training:
    G.load_state_dict(torch.load("generator_param_IAE_Refactored.pkl"))
    D.load_state_dict(torch.load("discriminator_param_IAE_Refactored.pkl"))
    R.load_state_dict(torch.load("regularizer_param_IAE_Refactored.pkl"))
    E.load_state_dict(torch.load("encoder_param_IAE_Refactored.pkl"))

# Binary Cross Entropy loss
BCE_loss = nn.BCEWithLogitsLoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr_gen, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr_disc, betas=(0.5, 0.999))
E_optimizer = optim.Adam(E.parameters(), lr=lr_gen, betas=(0.5, 0.999))
R_optimizer = optim.Adam(R.parameters(), lr=lr_disc, betas=(0.5, 0.999))

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    
    epoch_start_time = time.time()

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        E_optimizer.param_groups[0]['lr'] /= 10
        R_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        E_optimizer.param_groups[0]['lr'] /= 10
        R_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    
    num_iter = 0
    
    for x_, _ in train_loader:
        
        D.train()
        G.train()
        R.train()
        E.train()
        
        #####
        ##### train discriminator D
        #####
        
        D_threshold = upper_bound_D + 1.0
        k_D = 0
        
        # Iterate until Discriminator Loss is below upper_bound_D or k_update steps have been made
        while D_threshold > upper_bound_D and k_D < k_update:
            
            k_D = k_D + 1
            
            D.zero_grad()

            mini_batch = x_.size()[0]

            # Indices for real data samples (zeros)
            y_real_ = torch.zeros(mini_batch)
            # Indices for fake data samples (ones)
            y_fake_ = torch.ones(mini_batch)

            x_, y_real_, y_fake_ = x_.cuda(), y_real_.cuda(), y_fake_.cuda()

            # Encode real data
            z_ = E(x_)
            
            # Generate fake data from encoded latent variables
            G_result = G(z_)

            # Train the discriminator to recognise real image + latent pairs
            D_result = D(x_,z_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            # Train the discriminator to recognise fake image + latent pairs
            D_result = D(G_result,z_).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            
            # If required, perform gradient regularization
            if regularize_D == True:
                D_grad = torch.autograd.grad(D_real_loss + D_fake_loss, D.parameters(), create_graph=True)
                D_reg_loss = 0.0
                for gr in D_grad:
                    D_reg_loss += (gr**2).sum().cuda()
                D_train_loss = D_real_loss + D_fake_loss + reg_param_D*D_reg_loss
            else:
                D_train_loss = D_real_loss + D_fake_loss

            # Backpropagation
            D_train_loss.backward()
            
            # If required, clip gradients
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(D.parameters(), clip_norm)
            
            # Update step
            D_optimizer.step()

            # Check if Discriminator loss is below threshold
            D_threshold = D_train_loss.data.item()
            
        
        
        #####
        ##### train regularizer R
        #####
        
        R_threshold = upper_bound_R + 1.0
        k_R = 0
        
        # Iterate until Regularizer Loss is below upper_bound_D or k_update steps have been made
        while R_threshold > upper_bound_R and k_R < k_update:
            
            k_R = k_R + 1
        
            R.zero_grad()

            # Train the Regularizer to recognize samples from the true prior distribution on the latent space
            z_ = torch.randn((mini_batch, n_global)).cuda()
            R_result = R(z_).squeeze()
            R_real_loss = BCE_loss(R_result, y_real_)

            # Train the Regularizes to recognise samples from the conditional posterior distribution on the latent space
            z_ = E(x_).squeeze()
            R_result = R(z_).squeeze()
            R_fake_loss = BCE_loss(R_result, y_fake_)
            
            # If required, perform gradient regularization
            if regularize_D == True:
                R_grad = torch.autograd.grad(R_real_loss + R_fake_loss, R.parameters(), create_graph=True)
                R_reg_loss = 0.0
                for gr in R_grad:
                    R_reg_loss += (gr**2).sum().cuda()
                R_train_loss = R_real_loss + R_fake_loss + reg_param_R*R_reg_loss
            else:
                R_train_loss = R_real_loss + R_fake_loss
            
            # Backpropagation
            R_train_loss.backward()
            
            # If required, clip gradients
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(R.parameters(), clip_norm)
            
            # Perform update step
            R_optimizer.step()
            
            # Check if regularizer loss is still above threshold
            R_threshold = R_train_loss.data.item()
        
        
        
        #####
        ##### train generator G
        #####
        
        G.zero_grad()

        # Encode real data
        z_ = E(x_)
        
        # Generate fake data from encoded latent variables
        G_result = G(z_)
        
        # Evaluate discriminator loss
        D_result = D(G_result,z_).squeeze()        
        G_train_loss = torch.mean(D_result).cuda()
        
        # Backpropagation
        G_train_loss.backward(retain_graph = True)
        
        # If required, clip gradients
        if clip_grad:
                torch.nn.utils.clip_grad_norm_(G.parameters(), clip_norm)
        
        # Update step
        G_optimizer.step()
        
        

        #####
        ##### train encoder E
        #####
        
        E.zero_grad()
        
        # Encode real data
        z_ = E(x_).squeeze()
        
        # Evaluate regularizer loss
        R_result = R(z_).squeeze()
        
        # Encoder loss = sum of regularizer and discriminator loss
        E_train_loss = torch.mean(R_result).cuda() + G_train_loss
        
        # Backpropagation
        E_train_loss.backward()
        
        # If required, clip gradients
        if clip_grad:
                torch.nn.utils.clip_grad_norm_(E.parameters(), clip_norm)
        
        # Update step
        E_optimizer.step()
        
        
        # Create and write to log-file
        if num_iter == 0 and epoch == 0:
            with open('log_IAE_Refactored.txt', "w") as myfile:
                myfile.write("%f %f %f %f %f %f" % (D_train_loss.data.item(),G_train_loss.data.item(), R_train_loss.data.item(), E_train_loss.data.item(), z_.data.mean(), z_.data.std()) )
                if regularize_D:
                    myfile.write(" %f" % D_reg_loss.data.item())
                if regularize_R:
                    myfile.write(" %f" % R_reg_loss.data.item())
                myfile.write("\n")
        else:
            with open('log_IAE_Refactored.txt', "a") as myfile:
                myfile.write("%f %f %f %f %f %f" % (D_train_loss.data.item(),G_train_loss.data.item(), R_train_loss.data.item(), E_train_loss.data.item(), z_.data.mean(), z_.data.std()) )
                if regularize_D:
                    myfile.write(" %f" % D_reg_loss.data.item())
                if regularize_R:
                    myfile.write(" %f" % R_reg_loss.data.item())
                myfile.write("\n")
        
        # Count interations
        num_iter += 1
        
        # Save samples every n_th update step
        if num_iter % n_save_images == 0:
            p = 'CelebA_IAE_Refactored_iter_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_IAE_Refactored_fixed_iter_' + str(epoch + 1) + '.png'
            encoding_p = 'encoding_CelebA_IAE_Refactored_fixed_iter_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            z_enc = show_encoding_and_interpolation((epoch+1), personal_loader, train_loader, save=True, path=encoding_p, isFix=True)            

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - time: %.2f' % ((epoch + 1), train_epoch, per_epoch_ptime) )
    
    # Save samples after every epoch
    p = 'CelebA_IAE_Refactored_' + str(epoch + 1) + '.png'
    fixed_p = 'CelebA_IAE_Refactored_fixed_' + str(epoch + 1) + '.png'
    encoding_p = 'encoding_CelebA_IAE_Refactored_fixed_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    z_enc = show_encoding_and_interpolation((epoch+1), personal_loader, train_loader, save=True, path=encoding_p, isFix=True)
    
    # Save network parameters after every epoch
    torch.save(G.state_dict(), "generator_param_IAE_Refactored.pkl")
    torch.save(D.state_dict(), "discriminator_param_IAE_Refactored.pkl")
    torch.save(R.state_dict(), "regularizer_param_IAE_Refactored.pkl")
    torch.save(E.state_dict(), "encoder_param_IAE_Refactored.pkl")

end_time = time.time()
total_ptime = end_time - start_time

print("Training finished!")