import os
import shutil
import itertools
import tifffile as tiff

import numpy as np

import torch

EPS =  1e-12


def normalization(image0, image1, image2, cfg):
    if cfg.normalization == 'min_max':            
        min_, max_ = cfg.min_max
        image0 = (image0 - min_)/(max_ - min_)
        image1 = (image1 - min_)/(max_ - min_)
        image2 = (image2 - min_)/(max_ - min_)
    

    elif cfg.normalization == 'zscore':
        image0 = (image0 - image0.float().mean()) / max(image0.float().std(), EPS)
        image1 = (image1 - image1.float().mean()) / max(image1.float().std(), EPS)
        image2 = (image2 - image2.float().mean()) / max(image2.float().std(), EPS)

    return image0, image1, image2


def create_zdataset(list_of_files:str, cfg:dict):
    """
    Generates a 'datasets' folder containing one dataset per channel. Each dataset comprises PyTorch tensors, 
    with each tensor representing three consecutive 2D slices extracted from the input 5D image array.

    Args:
        list_of_files (str): File paths of the input images.
        cfg (dict): Configuration options including dataset size, model name, and validation split ratio.

    Returns:
        None
    """
    dataset_size = cfg.dataset_size
    model_name = cfg.model_name
    p_val = cfg.p_val

    triplets = np.empty((0, 5), dtype=int)
    
    root = '.'
    for i, file in enumerate(list_of_files):

        path = f"{root}/data/train/{file}"
        image = tiff.imread(path) 

        format = 'y'
        len_shape = len(image.shape)
        if len_shape <= 2:
            raise ValueError('Error: images must be at least 3D.')
        elif len_shape == 3:
            image = image[None, :, None, :, :]
        elif len_shape == 4:
            if format == 'y':
                image = image[:, :, None, :, :]
            elif format == 'n':
                image = image[None, :, :, :, :]
            else:
                raise ValueError('Error: data file-format not recognized.')
        else:
            raise ValueError('Error: images must be at most 5D.')

        t, z = image.shape[0:2]

        if model_name == 'zaugnet+':
            all_triplets = np.array(list(itertools.combinations(range(z), 3)))
            selected_all_triplets = []
            for idx in range(len(all_triplets)):
                if (all_triplets[idx][2] - all_triplets[idx][0]) <= cfg.distance_triplets : 
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
    
    if cfg.save_dataset :
        if os.path.exists('./dataset/'):
            shutil.rmtree('./dataset/')  
        save_data(f'{root}/data/train/', f'{root}/dataset/train_{cfg.model_name}/', triplets_train) 
        save_data(f'{root}/data/train/', f'{root}/dataset/val_{cfg.model_name}/', triplets_val) 
 
  
def save_data(data_path, dataset_path, triplets):
    os.makedirs(dataset_path, exist_ok=True)

    files = os.listdir(data_path)
    for file_name in set(list(triplets[:,0])):
        selected_triplets = triplets[np.where(triplets[:,0] == file_name)]
        image = tiff.imread(f"{data_path}{files[file_name]}")

        format = 'y'
        len_shape = len(image.shape)
        if len_shape <= 2:
            raise ValueError('Error: images must be at least 3D.')
        elif len_shape == 3:
            image = image[None, :, None, :, :]
        elif len_shape == 4:
            if format == 'y':
                image = image[:, :, None, :, :]
            elif format == 'n':
                image = image[None, :, :, :, :]
            else:
                raise ValueError('Error: data file-format not recognized.')
        else:
            raise ValueError('Error: images must be at most 5D.')

        image = torch.from_numpy(image)  #.to(torch.uint16)
        image = image.to(torch.int32) # for torch.uint16 problem nuclei
        for tri in selected_triplets:
            torch.save(image[tri[1], tri[2:]], f"{dataset_path}{tri[0]}_{tri[1]}_{tri[2]}_{tri[3]}_{tri[4]}.pt")
 