import numpy as np

import torch
import torch.cuda
from torch.autograd import Variable

from config import config


def compute_gradient_penalty(D, real_samples, fake_samples, DPM, device):
    """
    Computes the gradient penalty for Wasserstein GAN with Gradient Penalty (WGAN-GP).
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, DPM) 
    fake = Variable(torch.ones(real_samples.shape[0], 1).to(device), requires_grad=False)

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


def train_zaugnet(train_loader, generator, discriminator, optimizer_G, optimizer_D, lap_loss, lambda_adv=0.001, lambda_gp=10, n_critic=5):
    """
    Trains the ZAugGAN model.

    Args:
        train_loader: DataLoader wrapping the training dataset.
        generator, discriminator: Models to be trained.
        optimizer_G, optimizer_D: Optimizers for generator and discriminator.
        lap_loss: Loss functions.
        lambda_adv, lambda_gp: Hyperparameters for adversarial loss and gradient penalty.
        n_critic: Number of discriminator training steps per generator step.

    Returns:
        Trained generator, discriminator, and average losses for generator and discriminator.
    """

    opt = config()
    device = torch.device(f'cuda:{opt.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    d_loss_list, g_loss_list = [], [] 

    for i, batch in enumerate(train_loader):
        frame_1 = batch["f1"].to(device, non_blocking = True)
        frame_2 = batch["f2"].to(device, non_blocking = True)
        frame_gt = batch["gt"].to(device, non_blocking = True)
        DPM = batch["DPM"].to(device, non_blocking = True)

        # -----------------
        #  Train Discriminator
        # -----------------
        """https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py"""
        optimizer_D.zero_grad()

        # Generate a batch of images
        _, _, merged, _, _, _ = generator(torch.cat((frame_1, frame_2, frame_gt), 1), DPM=DPM)
        fake_imgs = merged[2]

        # Real images
        real_validity = discriminator(frame_gt, None)

        # Fake images
        fake_validity = discriminator(fake_imgs, None)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, frame_gt, fake_imgs, None, device)

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward(retain_graph=True)

        optimizer_D.step()        

        optimizer_G.zero_grad()

        if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            # Generate a batch of images
            _, _, merged, _, merged_teacher, loss_distill = generator(torch.cat((frame_1, frame_2, frame_gt), 1), DPM=DPM)
            fake_imgs = merged[2]

            fake_validity = discriminator(fake_imgs, None)
    
            l1_loss = (lap_loss(fake_imgs, frame_gt)).mean()
            tea_loss = (lap_loss(merged_teacher, frame_gt)).mean()

            g_loss = l1_loss + tea_loss + loss_distill * 0.01 - lambda_adv * torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()
    
        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item()) 

    return generator, discriminator, np.mean(g_loss_list), np.mean(d_loss_list)



def validate_zaugnet(val_loader, generator, discriminator, lap_loss, lambda_adv=0.001):
    """
    Validates the ZAugGAN model.

    Args:
        val_loader: DataLoader wrapping the validation dataset.
        generator, discriminator: Models to be validated.
        lap_loss: Loss functions.
        lambda_adv: Hyperparameter for adversarial loss.

    Returns:
        Validated generator, discriminator, and average losses for generator and discriminator.
    """
    opt = config()
    device = torch.device(f'cuda:{opt.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    d_loss_list, g_loss_list = [], []

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            frame_1 = batch["f1"].to(device, non_blocking=True)
            frame_2 = batch["f2"].to(device, non_blocking=True)
            frame_gt = batch["gt"].to(device, non_blocking=True)
            DPM = batch["DPM"].to(device, non_blocking = True)

            
            _, _, merged, _, merged_teacher, loss_distill = generator(torch.cat((frame_1, frame_2, frame_gt), 1), DPM=DPM)
            fake_imgs = merged[2]

            real_validity = discriminator(frame_gt, None)
            # Fake images
            fake_validity = discriminator(fake_imgs, None)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

            l1_loss = (lap_loss(merged[2], frame_gt)).mean()
            tea_loss = (lap_loss(merged_teacher, frame_gt)).mean()
            g_loss = l1_loss + tea_loss + loss_distill * 0.01 - lambda_adv * torch.mean(fake_validity)

            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

    return generator, discriminator, np.mean(g_loss_list), np.mean(d_loss_list)