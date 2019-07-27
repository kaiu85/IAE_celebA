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
from torch.autograd import Variable
import numpy
from numpy.linalg import norm

# training parameters
batch_size = 128 # size of mini-batches
lr_gen = 3e-4 # learning rate for parameters of encoder and decoder
train_epoch = 200 # number of epochs to train
clip_grad = True # should gradients be clipped
clip_norm = 0.25 # maximum value of the norm of gradients



# Path to CelebA (preprocessed to 96x96 pixels)
data_dir = '/home/kai/bernstein/Deep_Learning/celebA/celebA/'          # this path depends on your computer
# Path to two face images (preprocessed to 96x96 pixels), which will be encoded, reconstructed and interpolated
personal_dir = '/home/kai/bernstein/Deep_Learning/celebA/aligned_personal/'          # this path depends on your computer

# Model Parameters
n_global = 128 # dimensionality of latent space
n_hidden_regularizer = 2000 # dimensionality of hidden layer of regularizer
n_hidden_discriminator = 1000 # dimensionality of discriminator
model_dim = 128 # scale parameter used to control the number of channels used in the convolutional architecture
continue_training = False # False: Training starts from scratch and old checkpoints will be overwritten.
                          # True: the last checkpoint is loaded and training progresses from there. 
sig_min = 1.0 # Minimum value of standard-deviation of Gaussian densities on latent and image space

torch.cuda.set_device(0)

# G(z)
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
        #self.ln5 = nn.BatchNorm2d(d)
        self.deconv_mean = nn.ConvTranspose2d(d, 3, 3, 1, 0)
        self.deconv_std = nn.ConvTranspose2d(d, 3, 3, 1, 0)

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
        x_mean = F.tanh(self.deconv_mean(x))
        x_std = F.softplus(self.deconv_std(x)) + sig_min
        
        #print(x.data.shape)

        return x_mean, x_std
    
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
        #self.ln4 = nn.BatchNorm2d(d*8)
        self.conv_mean = nn.Conv2d(d*8, n_global, 4, 1, 0)
        self.conv_std = nn.Conv2d(d*8, n_global, 4, 1, 0)
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        
        x = F.relu(self.ln1(self.conv1(x))) # + self.fc1(eps).view(-1,self.d,1,1) ))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x_mean = self.conv_mean(x)
        x_std = F.softplus(self.conv_std(x))

        return x_mean, x_std

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
fixed_z_ = torch.randn((5 * 5, n_global)).view(-1, n_global, 1, 1)    # fixed noise
fixed_z_ = fixed_z_.cuda()

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False, show_mean = False):
    z_ = torch.randn((5*5, n_global)).view(-1, n_global, 1, 1)
    z_ = z_.cuda()
    

    G.train()
    if isFix:
        test_images_mean, test_images_std = G(fixed_z_)
    else:
        test_images_mean, test_images_std = G(z_)
        
    if show_mean == False:
        eps = torch.randn(test_images_std.shape).cuda()
        test_images = test_images_mean + eps*test_images_std
    else:
        test_images = test_images_mean

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
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

def show_encoding_and_interpolation(num_epoch, personal, train, show = False, save = False, path = 'result.png', isFix=False, show_mean = False):
    
    personal_iterator = iter(personal)
    xp, _ = next(personal_iterator)  
    xp = xp.cuda()
    
    train_iterator = iter(train)
    x, _ = next(train_iterator)  
    x = x.cuda()
    
    x[0:3] = xp
    
    E.train()
    z_mean, z_std = E(x)
    E.train()
    
    eps = torch.randn(z_std.data.shape).cuda()
    z = z_mean + eps * z_std
    
    G.train()
    G_result_mean, G_result_std = G(z)
    G.train()
    
    if show_mean == False:
        eps = torch.randn(G_result_std.data.shape).cuda()
        G_result = G_result_mean + eps*G_result_std
    else:
        G_result = G_result_mean
    
    G_n = G_result.data
    
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
    
    z_ = torch.randn((5*5, n_global)).view(-1, n_global)
    
    z0 = z[0,:,0,0].detach()
    z1 = z[1,:,0,0].detach()
    
    for i in range(25):
        z_[i,:] = z0*numpy.sqrt(1.0 - float(i)/24.0) + z1*numpy.sqrt(float(i)/24.0) 
        
    z_ = z_.cuda().view(-1, n_global, 1, 1)
    
    G.train()
    test_images_mean, test_images_std = G(z_)
    G.train()
    
    if show_mean == False:
        eps = torch.randn(test_images_std.data.shape).cuda()
        test_images = test_images_mean + eps*test_images_std
    else:
        test_images = test_images_mean
    
    #Show original images as end points
    #test_images[0,:,:,:] = x[0,:,:,:]
    #test_images[24,:,:,:] = x[1,:,:,:]
    
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()
        
    return
        

# data_loader
img_size = 96
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)
    
dset = datasets.ImageFolder(personal_dir, transform)
personal_loader = torch.utils.data.DataLoader(dset, batch_size=3, shuffle=False)

# network
G = generator(model_dim)
E = encoder(model_dim)
G.weight_init(mean=0.0, std=0.02)
E.weight_init(mean=0.0, std=0.02)
G.cuda()
E.cuda()

if continue_training:
    G.load_state_dict(torch.load("generator_param_VAE_Refactoring.pkl"))
    E.load_state_dict(torch.load("encoder_param_VAE_Refactoring.pkl"))

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr_gen, betas=(0.5, 0.999))
E_optimizer = optim.Adam(E.parameters(), lr=lr_gen, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('CelebA_DCGAN_results'):
    os.mkdir('CelebA_DCGAN_results')
if not os.path.isdir('CelebA_DCGAN_results/Random_results'):
    os.mkdir('CelebA_DCGAN_results/Random_results')
if not os.path.isdir('CelebA_DCGAN_results/Fixed_results'):
    os.mkdir('CelebA_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    
    epoch = epoch
    
    G.train()
    E.train()
    
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        E_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        E_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    
    num_iter = 0

    epoch_start_time = time.time()
    for x_, _ in train_loader:
        
        G.train()
        E.train()
        
        mini_batch = x_.data.shape[0]
        
        x_ = x_.cuda()
      
        # train generator G
        G.zero_grad()
        
        z_mean, z_std = E(x_)
        
        eps = torch.randn(z_std.data.shape).cuda()
        
        z_ = z_mean + eps * z_std
        
        G_result_mean, G_result_std = G(z_)
        
        G_train_loss = torch.sum( torch.mean( (x_ - G_result_mean)**2 / ( 2*(G_result_std**2) ) + torch.log(G_result_std) , dim = 0 ) )
        
        G_train_loss.backward(retain_graph = True)
        
        if clip_grad:
                torch.nn.utils.clip_grad_norm_(G.parameters(), clip_norm)
        
        G_optimizer.step()

        G_losses.append(G_train_loss.data.item())
        
        # train encoder E
        E.zero_grad()
        
        KL_loss = torch.sum( torch.mean( -torch.log(z_std) + (z_mean**2)/2 + (z_std)**2/2, dim = 0) )
        
        E_train_loss = G_train_loss + KL_loss
        
        E_train_loss.backward()
        
        if clip_grad:
                torch.nn.utils.clip_grad_norm_(E.parameters(), clip_norm)
        
        E_optimizer.step()
        
        
        if num_iter == 0 and epoch == 0:
            with open('log_VAE_Refactoring.txt', "w") as myfile:
                myfile.write("%f %f %f %f" % (G_train_loss.data.item(), E_train_loss.data.item(), z_.data.mean(), z_.data.std()) )
                myfile.write("\n")
        else:
            with open('log_VAE_Refactoring.txt', "a") as myfile:
                myfile.write("%f %f %f %f" % (G_train_loss.data.item(), E_train_loss.data.item(), z_.data.mean(), z_.data.std()) )
                myfile.write("\n")
        

        num_iter += 1
        
        if num_iter % 100 == 0:
            p = 'CelebA_VAE_Refactoring_iter_' + str(epoch + 1) + '.png'
            fixed_p = 'CelebA_VAE_Refactoring_fixed_iter_' + str(epoch + 1) + '.png'
            encoding_p = 'encoding_CelebA_VAE_Refactoring_fixed_iter_' + str(epoch + 1) + '.png'
            show_result((epoch+1), save=True, path=p, isFix=False)
            show_result((epoch+1), save=True, path=fixed_p, isFix=True)
            show_encoding_and_interpolation((epoch+1), personal_loader, train_loader, save=True, path=encoding_p, isFix=True)            
            show_result((epoch+1), save=True, path='mean_' + p, isFix=False, show_mean = True)
            show_result((epoch+1), save=True, path='mean_' + fixed_p, isFix=True, show_mean = True)
            show_encoding_and_interpolation((epoch+1), personal_loader, train_loader, save=True, path='mean_' + encoding_p, isFix=True, show_mean = True)            

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = 'CelebA_VAE_Refactoring_' + str(epoch + 1) + '.png'
    fixed_p = 'CelebA_VAE_Refactoring_fixed_' + str(epoch + 1) + '.png'
    encoding_p = 'encoding_CelebA_VAE_Refactoring_fixed_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    show_encoding_and_interpolation((epoch+1), personal_loader, train_loader, save=True, path=encoding_p, isFix=True)
    show_result((epoch+1), save=True, path='mean_' + p, isFix=False, show_mean = True)
    show_result((epoch+1), save=True, path='mean_' + fixed_p, isFix=True, show_mean = True)
    show_encoding_and_interpolation((epoch+1), personal_loader, train_loader, save=True, path='mean_' + encoding_p, isFix=True, show_mean = True)            

    torch.save(G.state_dict(), "generator_param_VAE_Refactoring.pkl")
    torch.save(E.state_dict(), "encoder_param_VAE_Refactoring.pkl")
    
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "generator_param.pkl")
torch.save(E.state_dict(), "encoder_param.pkl")
#with open('CelebA_DCGAN_results/train_hist.pkl', 'wb') as f:
#    pickle.dump(train_hist, f)

#show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/CelebA_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'CelebA_VAE_Refactoring_fixed_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('generation_animation.gif', images, fps=5)

