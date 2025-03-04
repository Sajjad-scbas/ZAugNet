import os
import time
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ZDataset
from training import train_zaugnet, validate_zaugnet
from model import ZAugGenerator, ZAugDiscriminator
from laplacian import LapLoss
from utils import create_zdataset

from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)


def train(cfg) : 
    root = '.'

    # Initialize device and logging
    device = torch.device(f'cuda:{cfg.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f"./runs/ZAugNet")
    writer.flush()

    #Set up the training and validation datasets and their loaders.
    list_of_files = os.listdir(f'{root}/data/train/')
    create_zdataset(list_of_files, cfg=cfg)

    dataset_train = ZDataset(f'{root}/dataset/train_{cfg.model_name}/', cfg, augmentations=cfg.augmentations)
    dataset_val = ZDataset(f'{root}/dataset/val_{cfg.model_name}/', cfg, augmentations=False)

    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True)

   
    #Initialize models and their optimizers.
    generator = ZAugGenerator(cfg=cfg).to(device)
    discriminator = ZAugDiscriminator()
    
    if torch.cuda.is_available():
        generator.set_multiple_gpus()
        discriminator = nn.DataParallel(discriminator, device_ids=cfg.device_ids).cuda()
        discriminator.to(f"cuda:{cfg.device_ids[0]}") 

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    # Loss function
    lap_loss = LapLoss()

    print('training loop')
    for epoch in tqdm(range(cfg.n_epochs), desc='Epochs'):
        # Train model
        generator, discriminator, g_loss_train, d_loss_train = train_zaugnet(
            train_loader, generator, discriminator, optimizer_G, 
            optimizer_D, lap_loss, cfg
        )

        # Validate model 
        generator, discriminator, g_loss_val, d_loss_val = validate_zaugnet(
            val_loader, generator, discriminator, lap_loss, cfg
        )
        
        # Log training and validation losses
        print(
            "Training : [Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, cfg.n_epochs, d_loss_train, g_loss_train)
        )
        print(
            "Validation : [Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, cfg.n_epochs, d_loss_val, g_loss_val)
        )

        # Writing log in tensorboard
        writer.add_scalar('training_d_loss', d_loss_train, epoch)
        writer.add_scalar('training_g_loss', g_loss_train, epoch)
        writer.add_scalar('validation_d_loss', d_loss_val, epoch)
        writer.add_scalar('validation_g_loss', g_loss_val, epoch)
        
    # Save the model's state dictionary.
    root = f'./results/{cfg.dataset}/{cfg.model_name}'
    name_state_dict = time.strftime(f"{root}/{cfg.model_name}-state-dict" + "-%Y%m%d-%H%M%S" +
                            "-beta1" + str(cfg.beta1) + "-beta2" + str(cfg.beta2) +
                            "-lambda_adv" + str(cfg.lambda_adv) + "-lambda_gp" + str(cfg.lambda_gp) +
                            "-lr" + str(cfg.lr)  + "-n_critic" + str(cfg.n_critic) +
                            "-n_epochs" + str(cfg.n_epochs) + "-distance_triplets" + str(cfg.distance_triplets) + ".pt")
    
    if not os.path.exists(root) :
        os.makedirs(root)
    torch.save(generator.state_dict(), name_state_dict)

if __name__ == "__main__":
    from config import config
    import ast
    cfg = config()
    cfg.device_ids = ast.literal_eval(cfg.device_ids)
    train(cfg)
    print('Done')
