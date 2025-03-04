import numpy as np

import torch
import torch.cuda
from torch.autograd import Variable

from tqdm import tqdm


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


def train_zaugnet(train_loader, generator, discriminator, optimizer_G, optimizer_D, lap_loss, cfg):
    """
    Trains the ZAugGAN model using the given data, models, and loss functions.

    Args:
        train_loader: DataLoader for the training dataset.
        generator, discriminator: Generator and discriminator models.
        optimizer_G, optimizer_D: Optimizers for the generator and discriminator.
        lap_loss: Loss function(s) for the training process.
        cfg: Configuration options, including hyperparameters such as lambda_adv, lambda_gp, and n_critic.

    Returns:
        Updated generator and discriminator models, along with average losses for both.
    """

    lambda_adv = cfg.lambda_adv
    lambda_gp = cfg.lambda_gp
    n_critic = cfg.n_critic
    device = torch.device(f'cuda:{cfg.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    d_loss_list, g_loss_list = [], [] 

    for i, batch in enumerate(tqdm(train_loader, desc='Training loop')):
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


def validate_zaugnet(val_loader, generator, discriminator, lap_loss, cfg):
    """
    Validates the ZAugGAN model on the validation dataset.

    Args:
        val_loader: DataLoader for the validation dataset.
        generator, discriminator: Generator and discriminator models to validate.
        lap_loss: Loss function(s) used for validation.
        cfg: Configuration options, including hyperparameters like lambda_adv.

    Returns:
        Validation losses for the generator and discriminator.
    """
    lambda_adv = cfg.lambda_adv
    device = torch.device(f'cuda:{cfg.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    d_loss_list, g_loss_list = [], []

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for _, batch in enumerate(tqdm(val_loader, desc='Validation loop')):
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
