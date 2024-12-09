import os
import itertools
import tifffile as tiff

import numpy as np

import torch

from config import config

EPS =  1e-12


def normalization(image0, image1, image2, opt):

    if opt.normalization == 'min_max':            
        min_, max_ = opt.min_max
        image0 = (image0 - min_)/(max_ - min_)
        image1 = (image1 - min_)/(max_ - min_)
        image2 = (image2 - min_)/(max_ - min_)
    

    elif opt.normalization == 'zscore':
        image0 = (image0 - image0.float().mean()) / max(image0.float().std(), EPS)
        image1 = (image1 - image1.float().mean()) / max(image1.float().std(), EPS)
        image2 = (image2 - image2.float().mean()) / max(image2.float().std(), EPS)

    return image0, image1, image2

def create_zdataset(list_of_files: str,
                    dataset_size, 
                    model_name : str, 
                    p_val: float = 0.1):
    """
    It creates a new folder called 'datasets' with one dataset per channel made of pytorch tensors containing three
    consecutive 2D images from the 5d image_fill in input.

    Args:
        filename: (str) filename image_fill
        image_fill: (np.ndarray) image to split
        channel: (int) image channel
        normalization: (bool) normalization range [0,1]
        zscore: (bool) zscore mean 0 std 1
    Return:
        (void)
    """
    opt = config()

    cfg_nb = opt.cfg_nb
    os.makedirs(f'./dataset/train{cfg_nb}_{opt.model_name}/', exist_ok=True)
    os.makedirs(f'./dataset/val{cfg_nb}_{opt.model_name}', exist_ok=True)

    triplets = np.empty((0, 5), dtype=int)
    
    root = '.'
    for i, file in enumerate(list_of_files):

        path = f"{root}/data/train{cfg_nb}/{file}"
        image = tiff.imread(path)  # .astype('uint8')

        len_shape = len(image.shape)
        if len_shape <= 2:
            print('Error: images has to be 3d at least. ')
            exit()
        elif len_shape == 3:
            image = image[None, :, None, :, :]
        elif len_shape == 4:
            if i == 0:
                #format = input('are data time-lapses? [y/n]  ')
                format = 'y'
                old_format = format
            else:
                format = old_format
            if format == 'y':
                image = image[:, :, None, :, :]
            elif format == 'n':
                image = image[None, :, :, :, :]
            else:
                print('Error: data format not recognized.')
                exit()
        else:
            print('Error: images has to be 5d maximum.')
            exit()

        t, z, c, h, w = image.shape

        if model_name == 'zaugnet+':
            all_triplets = np.array(list(itertools.combinations(range(z), 3)))
            selected_all_triplets = []
            for idx in range(len(all_triplets)):
                if (all_triplets[idx][2] - all_triplets[idx][0]) <= opt.distance_triplets : 
                    selected_all_triplets.append(all_triplets[idx])
            all_triplets = np.array(selected_all_triplets)

            for dt in range(t):
                rand_ = np.random.randint(0, len(all_triplets), dataset_size)
                            
                selected_triplets = np.concatenate((np.repeat([[i,dt]], dataset_size, axis=0), all_triplets[rand_]), axis=1)
                triplets = np.concatenate((triplets, selected_triplets))            


        else :
            for dt, dz in itertools.product(range(t), range(z - 2)):
                triplets = np.concatenate((triplets, np.array([(i, dt, dz, dz+1, dz+2)])))


    np.random.shuffle(triplets)
    triplets_train = triplets[:int((1-p_val)*len(triplets))]
    triplets_val = triplets[int((1-p_val)*len(triplets)):]
    
    if opt.save_dataset :
        save_data(root + f'/data/train{cfg_nb}/', root + f'/dataset/train{cfg_nb}_{opt.model_name}/', triplets_train) 
        save_data(root + f'/data/train{cfg_nb}/', root + f'/dataset/val{cfg_nb}_{opt.model_name}/', triplets_val) 
 
  
def save_data(data_path, dataset_path, triplets):  
    files = os.listdir(data_path)
    for file_name in set(list(triplets[:,0])):
        selected_triplets = triplets[np.where(triplets[:,0] == file_name)]
        image = tiff.imread(f"{data_path}{files[file_name]}")

        i = 0
        len_shape = len(image.shape)
        if len_shape <= 2:
            print('Error: images has to be 3d at least. ')
            exit()
        elif len_shape == 3:
            image = image[None, :, None, :, :]
        elif len_shape == 4:
            if i == 0:
                #format = input('are data time-lapses? [y/n]  ')
                format = 'y'
                old_format = format
            else:
                format = old_format
            if format == 'y':
                image = image[:, :, None, :, :]
            elif format == 'n':
                image = image[None, :, :, :, :]
            else:
                print('Error: data format not recognized.')
                exit()
        else:
            print('Error: images has to be 5d maximum.')
            exit()

        image = torch.from_numpy(image)  #.to(torch.uint16)
        image = image.to(torch.int32) # for torch.uint16 problem nuclei
        for tri in selected_triplets:
            torch.save(image[tri[1], tri[2:]], f"{dataset_path}{tri[0]}_{tri[1]}_{tri[2]}_{tri[3]}_{tri[4]}.pt")
 