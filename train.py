import os
import time
import zipfile

import tifffile as tiff

from utils import read_settings, crop_3d, create_zdataset, str_to_bool

import torch
from torch.utils.data import DataLoader

from dataset import ZDataset
from training import train_zaugnet

if __name__ == '__main__':

    root = '.'
    root_contents = os.listdir(root)

    # Reading settings

    settings = read_settings()

    augmentations = str_to_bool(settings['augmentations'])
    print('augmentations=\t' + str(augmentations))
    batch_size = int(settings['batch_size'])
    print('batch size=\t' + str(batch_size))
    beta1 = float(settings['beta1'])
    print('beta 1=\t' + str(beta1))
    beta2 = float(settings['beta2'])
    print('beta 2=\t' + str(beta2))
    lambda_adv = float(settings['lambda_adv'])
    print('lambda adversarial=\t' + str(lambda_adv))
    lambda_gp = int(settings['lambda_gp'])
    print('lambda gp=\t' + str(lambda_gp))
    last_kernel_size = int(settings['last_kernel_size'])
    print('last kernel size=\t' + str(last_kernel_size))
    loss = str(settings['loss'])
    print('loss pixelwise=\t' + str(loss))
    lr = float(settings['lr'])
    print('learning rate=\t' + str(lr))
    n_critic = int(settings['n_critic'])
    print('n critic=\t' + str(n_critic))
    n_epochs = int(settings['n_epochs'])
    print('n epochs=\t' + str(n_epochs))
    normalization = str_to_bool(settings['normalization'])
    print('normalization=\t' + str(normalization))
    patch_size = int(settings['patch_size'])
    print('patch size=\t' + str(patch_size))
    zscore = str_to_bool(settings['zscore'])
    print('z score=\t' + str(zscore))


    # Loading data and preparing dataset

    list_of_files = os.listdir(root + '/data/')

    if 'datasets' not in os.listdir(root):

        for num, file in enumerate(list_of_files):

            filename_fill = str(file)
            path_fill = str(root) + '/data/' + filename_fill
            print('reading ' + filename_fill[:-4] + ' ...')
            image_fill = tiff.imread(path_fill).astype('uint8')

            len_shape = len(image_fill.shape)
            if len_shape <= 2:
                print('Error: images has to be 3d at least. ')
                exit()
            elif len_shape == 3:
                image_fill = image_fill[None, :, None, :, :]
            elif len_shape == 4:
                if num == 0:
                    format = input('Are data time-lapses? [y/n]  ')
                    old_format = format
                else:
                    format = old_format
                if format == 'y':
                    image_fill = image_fill[:, :, None, :, :]
                elif format == 'n':
                    image_fill = image_fill[None, :, :, :, :]
                else:
                    print('Error: data format not recognized.')
                    exit()
            elif len_shape == 5:
                pass
            else:
                print('Error: images has to be 5d maximum.')
                exit()

            cropped_image_fill = crop_3d(image_fill, image_fill.shape[1], patch_size, patch_size)

            t, z, c, h, w = cropped_image_fill.shape

            print('creating dataset from ' + filename_fill[:-4] + ' ...')
            for ch in range(c):
                create_zdataset(filename_fill, cropped_image_fill, ch, normalization=normalization, zscore=zscore)

        else:
            pass

    current_folder_contents = os.listdir(root)

    # Training

    for ch in range(len(os.listdir('./datasets'))):

        dataset = ZDataset(str(root) + '/datasets/dataset-ch' + str(ch) + '/', augmentations=augmentations)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print('training...')
        zaugnet = train_zaugnet(train_loader, beta1=beta1, beta2=beta2, lambda_adv=lambda_adv, lambda_gp=lambda_gp,
                                last_kernel_size=last_kernel_size, loss=loss, lr=lr, n_critic=n_critic, n_epochs=n_epochs)

        # Saving model

        name_state_dict = time.strftime("./zaugnet-state-dict" + "-%Y%m%d-%H%M%S" + "-ch" + str(ch) +
                                        "-beta1" + str(beta1) + "-beta2" + str(beta2) +
                                        "-lambda_adv" + str(lambda_adv) + "-lambda_gp" + str(lambda_gp) +
                                        "-lr" + str(lr) + "-last_kernel_size" + str(last_kernel_size) +
                                        "-loss" + str(loss) + "-n_critic" + str(n_critic) +
                                        "-n_epochs" + str(n_epochs) + ".pt")

        torch.save(zaugnet.state_dict(), name_state_dict)

        # Creating the ZIP file
        with zipfile.ZipFile('model.zip', 'w') as zipMe:
            for file in [name_state_dict, './settings.txt']:
                zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
