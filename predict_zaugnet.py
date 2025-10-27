import os
import time
from glob import glob
import tifffile as tiff

import torch
from torchvision.transforms import v2, InterpolationMode

from model import ZAugGenerator

from tqdm import tqdm 

EPS = 1e-12 # Small epsilon value to avoid division by zero


def load_model(cfg, dataset, model_name):
    """Load the pre-trained model based on the model name."""
    print(f"./results/{dataset}/{model_name}/*.pt")
    path_model = glob(f"./results/{dataset}/{model_name}/*.pt")[0]
    print(f"The model used for this prediction : {path_model}")

    zaug = ZAugGenerator(cfg)
    zaug.set_multiple_gpus()
    zaug.load_state_dict(torch.load(path_model,  map_location=f"cuda:{cfg.device_ids[0]}"))
    zaug.eval()

    return zaug


def preprocess_image(image, cfg, device, format='y'):
    """Preprocess images by adding necessary dimensions and applying resizing/cropping."""
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
    
    # Convert to torch tensor and move to device
    image = torch.from_numpy(image).float().to(device)
    # Apply resizing or cropping based on configuration options
    image = v2.CenterCrop(cfg.patch_size) if cfg.resize_par == 'crop' else v2.Resize((cfg.patch_size, cfg.patch_size), interpolation=InterpolationMode.BICUBIC)(image)

    return image


def normalize_image(image0, image2, cfg, device):
    """Normalize the images based on the chosen method (min_max or zscore)."""
    if cfg.normalization == 'min_max':
        min_, max_ = cfg.min_max
        image0 = ((image0 - min_)/(max_ - min_))
        image2 = ((image2 - min_)/(max_ - min_))
        norm_values = (min_, max_)
    elif cfg.normalization == 'zscore':
        image02 = torch.cat((image0, image2), dim=0)
        image02_mean = torch.mean(image02)
        image02_std = torch.max(torch.std(image02), torch.Tensor([EPS]).to(device))
        image0 = ((image0 - image02_mean) / image02_std)
        image2 = ((image2 - image02_mean) / image02_std)
        norm_values = (image02_mean, image02_std)
    else:
        raise ValueError('Error: Unsupported normalization type.')
    return image0, image2, norm_values


def denormalize_image(image0, image2, pred, cfg, norm_values):
    """Denormalize the images and predictions back to original scale."""
    if cfg.normalization == 'min_max':
        min_, max_ = norm_values
        image0 = image0 * (max_ - min_) + min_
        image2 = image2 * (max_ - min_) + min_
        pred = pred * (max_ - min_) + min_
    elif cfg.normalization == 'zscore':
        mean_, std_ = norm_values  
        image0 = image0 * std_ + mean_
        image2 = image2 * std_ + mean_
        pred = pred * std_ + mean_ 
    return image0, image2, pred


def predict(cfg, dataset):
    start_time = time.time()
    
    # Create output directory for predictions
    if not os.path.exists('./predictions'):
        os.makedirs('./predictions')

    list_of_files = os.listdir('./data/test/')
    list_of_files.sort()

    # Check if CUDA (GPU support) is available
    cuda = True if torch.cuda.is_available() else False
    print(f"Using CUDA device: {cuda}")

    device = torch.device(f'cuda:{cfg.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model_name = 'zaugnet'
    zaug = load_model(cfg, dataset, model_name)

    # Iterate over the files in the test data folder
    for filename in tqdm(list_of_files, desc='Files'):
        path = './data/test/' + filename
        image = tiff.imread(path)

        image = preprocess_image(image, cfg, device)    
    
        t, z, _, h, w = image.shape
        prediction = torch.zeros((t, int(z*2)-1, 1, h, w), dtype=torch.uint8, device=device)
    
        for dt in tqdm(range(t), desc='Time Range'):
            for dz in range(z-1):
                with torch.no_grad():
                    image0 = image[dt, dz][None, ].to(device, non_blocking = True)
                    image2 = image[dt, dz+1][None, ].to(device, non_blocking = True)

                    # Normalize the images
                    image0, image2, norm_values = normalize_image(image0, image2, cfg, device)

                    # Make predictions
                    pred = zaug(torch.cat((image0, image2), dim=1), gt_bool = False)[2]
                    pred = pred[2]

                    # Denormalize the images and prediction
                    image0, image2, pred = denormalize_image(image0, image2, pred, cfg, norm_values)

                    prediction[dt, 2 * dz + 0] = torch.clip(image0[0, 0, None, :, :], cfg.min_max[0], cfg.min_max[1]).type(prediction.dtype)
                    prediction[dt, 2 * dz + 2] = torch.clip(image2[0, 0, None, :, :], cfg.min_max[0], cfg.min_max[1]).type(prediction.dtype)
                    prediction[dt, 2 * dz + 1] = torch.clip(pred[0, 0, None, :, :], cfg.min_max[0], cfg.min_max[1]).type(prediction.dtype)

        # Convert the prediction tensor to a NumPy array and save it
        prediction = prediction.detach().cpu().numpy()
        tif_path = './predictions/' + str(filename[:-4]) +'-z' + str(int(z*2)-1) + '_' + model_name +'.tif'
        tiff.imwrite(tif_path, prediction, imagej=True)

    # Measure and print the total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
