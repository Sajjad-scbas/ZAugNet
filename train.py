import os
import time
from tqdm import tqdm 
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from dataset import ZDataset
from training import train_zaugnet, validate_zaugnet
from model import ZAugGenerator, ZAugDiscriminator
from laplacian import LapLoss
from utils import create_zdataset

from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    root = '.'

    opt = config()

    # Initialize device and logging
    device = torch.device(f'cuda:{opt.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f"./runs/ZAugNet")
    writer.flush()

    # Loading data and preparing dataset
    cfg_nb = opt.cfg_nb

    #Set up the training and validation datasets and their loaders.
    list_of_files = os.listdir(root + f'/data/train{cfg_nb}/')
    create_zdataset(list_of_files, dataset_size=opt.dataset_size, model_name=opt.model_name, p_val=opt.p_val)

    dataset_train = ZDataset(root + f'/dataset/train{cfg_nb}_{opt.model_name}/', augmentations=opt.augmentations)
    dataset_val = ZDataset(root + f'/dataset/val{cfg_nb}_{opt.model_name}/', augmentations=False)

    train_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)

   
    #Initialize models and their optimizers.
    generator = ZAugGenerator().to(device)
    discriminator = ZAugDiscriminator(model_name=opt.model_name)
    
    if torch.cuda.is_available():
        generator.set_multiple_gpus()
        discriminator = nn.DataParallel(discriminator, device_ids=opt.device_ids).cuda()
        discriminator.to(f"cuda:{opt.device_ids[0]}") 

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # Loss function
    lap_loss = LapLoss()

    print('training loop')
    for epoch in tqdm(range(opt.n_epochs)):
        # Train model
        generator, discriminator, g_loss_train, d_loss_train = train_zaugnet(
            train_loader, generator, discriminator, optimizer_G, optimizer_D,
            lap_loss, opt.lambda_adv, opt.lambda_gp, opt.n_critic
        )

        # Validate model 
        generator, discriminator, g_loss_val, d_loss_val = validate_zaugnet(
            val_loader, generator, discriminator, lap_loss, opt.lambda_adv
        )
        
        # Log training and validation losses
        print(
            "Training : [Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, d_loss_train, g_loss_train)
        )
        print(
            "Validation : [Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, d_loss_val, g_loss_val)
        )

        # Writing log in tensorboard
        writer.add_scalar('training_d_loss', d_loss_train, epoch)
        writer.add_scalar('training_g_loss', g_loss_train, epoch)
        writer.add_scalar('validation_d_loss', d_loss_val, epoch)
        writer.add_scalar('validation_g_loss', g_loss_val, epoch)
        # writer.add_scalar('lr_g', optimizer_G.param_groups[0]['lr'], epoch)
        # writer.add_scalar('lr_d', optimizer_D.param_groups[0]['lr'], epoch) 
        
    # Save the model's state dictionary.
    name_state_dict = time.strftime(f"./{opt.model_name}_RIFE-state-dict" + "-%Y%m%d-%H%M%S" + "epoch" + str(epoch) +
                            "-beta1" + str(opt.beta1) + "-beta2" + str(opt.beta2) +
                            "-lambda_adv" + str(opt.lambda_adv) + "-lambda_gp" + str(opt.lambda_gp) +
                            "-lr" + str(opt.lr) + "-last_kernel_size" + str(opt.last_kernel_size) +
                            "-loss" + str(opt.loss) + "-n_critic" + str(opt.n_critic) +
                            "-n_epochs" + str(opt.n_epochs) + "-distance_triplets" + str(opt.distance_triplets) + "_" + str(cfg_nb)+ ".pt")

    torch.save(generator.state_dict(), name_state_dict)
