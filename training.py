import os
import sys

import time
import datetime

import numpy as np

import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable

from torchvision.utils import save_image

from model import ZAugGenerator, ZAugDiscriminator


def train_zaugnet(train_loader, beta1=0.5, beta2=0.999, lambda_adv=0.001, lambda_gp=10, last_kernel_size=64, loss='L1', lr=0.0001, n_critic=5, n_epochs=2):
    """
    Training for the ZAugGAN architecture.

        Args:
            train_loader: (DataLoader) dataloader wrapping the training dataset
            loss: (str) loss function for the generator during training
            n_epochs: (int) number of epochs during training
            lr: (float) learning rate for Adam optimizer
            beta1: (float) beta1 coefficient for Adam optimizer
            beta2: (float) beta1 coefficient for Adam optimizer
            lambda_gp: (int) loss weight for gradient penalty
            n_critic: (int) number of training steps for discriminator per iteration
            lambda_adv: (float) weight for generator adversarial loss
            last_kernel_size: (float) last kernel size to initialize the architecture

        Returns:
            zaugnet: (nn.Module) the trained ZAugGAN model
    """

    os.makedirs('./training-images', exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    print(f"using cuda device: {cuda}")

    if torch.cuda.device_count() > 1:
        print("using", torch.cuda.device_count(), "GPUs for training")

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if loss == 'MSE' or loss == 'L2':
        criterion_pixelwise = torch.nn.MSELoss()
    elif loss == 'L1':
        criterion_pixelwise = torch.nn.L1Loss()
    else:
        print('Error: loss not found.\n')
        exit()

    generator = ZAugGenerator(last_kernel_size=last_kernel_size)
    discriminator = ZAugDiscriminator()

    if cuda:
        generator = nn.DataParallel(generator).cuda()  # parallelized on all available GPUs
        discriminator = nn.DataParallel(discriminator).cuda()  # parallelized on all available GPUs
        criterion_pixelwise.cuda()
    else:
        pass

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    def compute_gradient_penalty(D, real_samples, fake_samples):

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        d_interpolates = D(interpolates)

        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    # ----------
    #  Training
    # ----------

    for epoch in range(n_epochs):

        d_loss_list = []
        g_loss_list = []

        for i, batch in enumerate(train_loader):

            # input
            frame_1 = batch["f1"].type(Tensor)
            frame_2 = batch["f2"].type(Tensor)
            frame_gt = batch["gt"].type(Tensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Generate a batch of images
            fake_imgs = generator(frame_1, frame_2)

            # Real images
            real_validity = discriminator(frame_gt)

            # Fake images
            fake_validity = discriminator(fake_imgs)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, frame_gt, fake_imgs)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()

            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(frame_1, frame_2)

                # Train on fake images
                fake_validity = discriminator(fake_imgs)
 
                g_loss = criterion_pixelwise(fake_imgs, frame_gt) + lambda_adv * (-torch.mean(fake_validity))

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )

            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

        # d_loss and g_loss for each batch of each epoch
        with open('./losses.txt', 'a') as f:
            if epoch == 0 and i == 0:
                f.write('# Epoch D_loss G_loss \n')
                f.write("%d %f %f \n" % (epoch, np.mean(d_loss_list), np.mean(g_loss_list)))
            else:
                f.write("%d %f %f \n" % (epoch, np.mean(d_loss_list), np.mean(g_loss_list)))
        
        save_image((torch.cat((frame_1, fake_imgs, frame_2, frame_gt), 2)).data[0], "./training-images/epoch-%d.png" % epoch, normalize=True)

    return generator
