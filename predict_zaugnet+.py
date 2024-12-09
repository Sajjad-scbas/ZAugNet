import os
import time
from glob import glob
import tifffile as tiff

import torch
from torchvision.transforms import v2, InterpolationMode

from model import ZAugGenerator
from config import config

from tqdm import tqdm 


EPS = 1e-12 # Small epsilon value to avoid division by zero


def load_model(model_name):
    """Load the pre-trained model based on the model name."""
    path_model = glob(f"./results/human_embryo/{model_name}/*.pt")[0]
    print(f"The model used for this prediction : {path_model}")

    zaug = ZAugGenerator()
    zaug.set_multiple_gpus()
    zaug.load_state_dict(torch.load(path_model))
    zaug.eval()

    return zaug


def preprocess_image(image, opt, device, format='y'):
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
    image = v2.CenterCrop(opt.patch_size) if opt.resize_par == 'crop' else v2.Resize((opt.patch_size, opt.patch_size), interpolation=InterpolationMode.BICUBIC)(image)

    return image


def normalize_image(image0, image2, opt, device):
    """Normalize the images based on the chosen method (min_max or zscore)."""
    if opt.normalization == 'min_max':
        min_, max_ = opt.min_max
        image0 = ((image0 - min_)/(max_ - min_))
        image2 = ((image2 - min_)/(max_ - min_))
        norm_values = (min_, max_)
    elif opt.normalization == 'zscore':
        image02 = torch.cat((image0, image2), dim=0)
        image02_mean = torch.mean(image02)
        image02_std = torch.max(torch.std(image02), torch.Tensor([EPS]).to(device))
        image0 = ((image0 - image02_mean) / image02_std)
        image2 = ((image2 - image02_mean) / image02_std)
        norm_values = (image02_mean, image02_std)
    else:
        raise ValueError('Error: Unsupported normalization type.')
    return image0, image02, norm_values


def denormalize_image(image0, image2, pred, opt, norm_values):
    """Denormalize the images and predictions back to original scale."""
    if opt.normalization == 'min_max':
        min_, max_ = norm_values
        image0 = image0 * (max_ - min_) + min_
        image2 = image2 * (max_ - min_) + min_
        pred = pred * (max_ - min_) + min_
    elif opt.normalization == 'zscore':
        mean_, std_ = norm_values  
        image0 = image0 * std_ + mean_
        image2 = image2 * std_ + mean_
        pred = pred * std_ + mean_ 
    return image0, image2, pred


def main(factor):
    start_time = time.time()
    
    # Create output directory for predictions
    if not os.path.exists('./predictions'):
        os.makedirs('./predictions')

    # Get configuration options
    opt = config()

    list_of_files = os.listdir('./data/test/')
    list_of_files.sort()

    # Check if CUDA (GPU support) is available
    cuda = True if torch.cuda.is_available() else False
    print(f"Using CUDA device: {cuda}")

    device = torch.device(f'cuda:{opt.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model_name = 'human_embryo'
    zaug = load_model(model_name)

    # Iterate over the files in the test data folder
    for i, filename in enumerate(list_of_files):
        path = './data/test/' + filename
        image = tiff.imread(path)

        image = preprocess_image(image, opt, device)    
    
        t, z, c, h, w = image.shape
        prediction = torch.zeros((t, int(z*2)-1, 1, h, w), dtype=torch.uint8, device=device)
        
        DPMS = torch.linspace(0,1,factor+1)[1:-1].to(device)

        for dt in tqdm(range(t)):
            for dz in range(z-1):
                with torch.no_grad():
                    image0 = image[dt, dz][None, ].to(device, non_blocking = True)
                    image2 = image[dt, dz+1][None, ].to(device, non_blocking = True)

                    # Normalize the images
                    image0, image2, norm_values = normalize_image(image0, image2, opt, device)

                     # Make predictions
                    pred = zaug(torch.cat((image0, image2), dim=1).expand(factor-1, -1, -1, -1), DPM = DPMS.reshape(-1, 1,1,1), gt_bool = False)[2]
                    pred = pred[2]

                    # Denormalize the images and prediction
                    image0, image2, pred = denormalize_image(image0, image2, pred, opt, norm_values)

                    prediction[dt, factor * dz + 0] = torch.clip(image0[0, 0, None, :, :], opt.min_max[0], opt.min_max[1]).type(prediction.dtype)
                    prediction[dt, factor*(dz + 1)] = torch.clip(image2[0, 0, None, :, :], opt.min_max[0], opt.min_max[1]).type(prediction.dtype)
                    prediction[dt, factor*dz+1 : factor*(dz + 1)] = torch.clip(pred, opt.min_max[0], opt.min_max[1]).type(prediction.dtype)








    
        # Convert the prediction tensor to a NumPy array and save it
        prediction = prediction.detach().cpu().numpy()
        tif_path = './predictions/' + str(filename[:-4]) +'-z' + str(int(z*2)-1) + '_' + model_name +'.tif'
        tiff.imwrite(tif_path, prediction, imagej=True)

    # Measure and print the total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main(factor)