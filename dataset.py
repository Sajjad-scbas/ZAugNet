import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

from utils import normalization

class ZDataset(Dataset):
    """
    Dataset class for ZAugNet.
    
    Args:
        dataset_path (str): Folder path containing the image data.
        augmentations (bool): Whether to apply data augmentations.
    """
    
    def __init__(self, dataset_path, cfg, augmentations):
        self.path = str(dataset_path)
        self.data = os.listdir(dataset_path)
        self.augmentations = augmentations
        
        # Load configurations
        self.cfg = cfg

        if self.augmentations:
            self.transforms = v2.Compose([
                v2.CenterCrop(self.cfg.patch_size) if self.cfg.resize_par == 'crop' else v2.Resize((self.cfg.patch_size, self.cfg.patch_size), interpolation=InterpolationMode.BICUBIC),
                v2.RandomApply([v2.ColorJitter(contrast=[0.5, 1.5])], p = 0.3),
                v2.RandomApply([v2.ColorJitter(brightness=[0.4, 1.6])], p = 0.3),
                v2.RandomRotation((0, 270)), 
                v2.RandomHorizontalFlip(p=0.5),
                #v2.Lambda(lambda img: img[[2,1,0]] if torch.rand(1).item() > 0.5 else img)
            ])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Contains `f1`, `f2`, `gt` (normalized tensors) and `DPM` (displacement).
        """
        # Load the image and parse frame indices
        image = torch.load(self.path + self.data[idx])
        f1_idx, gt_idx, f2_idx = map(int, self.data[idx].split('.')[0].split('_')[2:5])
        
        if self.augmentations : 
            image = self.transforms(image)
        
        # adding DPM
        DPM = (gt_idx - f1_idx) * 1.0 / (f2_idx - f1_idx + 1e-6)

        if torch.rand(1).item() > 0.5:
            image = image[[2,1,0]]
            # f1_idx, f2_idx = f2_idx, f1_idx
            DPM = 1 - DPM

        f1 = image[0]
        f2 = image[2]
        gt = image[1]
        
        # Apply normalization
        f1, gt, f2 = normalization(f1, gt, f2, self.cfg)

        return {"f1": f1, "f2": f2, "gt": gt, "DPM":torch.tensor(DPM).reshape(1,1,1)}
