import os

import torch
from torch.utils.data import Dataset

from utils import rotateXY90, flipX, flipZ


class ZDataset(Dataset):
    """
    ZAugGAN Dataset class.

    Args:
        data: (str) folder path containing the images
    """
    
    def __init__(self, dataset_path, augmentations):

        self.path = str(dataset_path)
        self.augmentations = augmentations
        self.data = os.listdir(str(dataset_path))
        self.data_len = len(self.data)

        if augmentations:
            self.len = 16 * len(self.data)
        else:
            self.len = len(self.data)

    def __getitem__(self, idx):

        data_idx = idx % self.data_len
        path = self.path + self.data[data_idx]

        if self.augmentations:

            # Augmentations
            # (90deg-xy-plane-rotations and x-flips and z-flips: 16 variants from original the image)
            # (y-flips not needed to avoid repetitions)

            if int(idx / self.data_len) == 0:
                image = torch.from_numpy((rotateXY90(torch.load(path).cpu().detach().numpy(), 0)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 1:
                image = torch.from_numpy((rotateXY90(torch.load(path).cpu().detach().numpy(), 1)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 2:
                image = torch.from_numpy((rotateXY90(torch.load(path).cpu().detach().numpy(), 2)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 3:
                image = torch.from_numpy((rotateXY90(torch.load(path).cpu().detach().numpy(), 3)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 4:
                image = torch.from_numpy((rotateXY90(flipZ(torch.load(path).cpu().detach().numpy()), 0)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 5:
                image = torch.from_numpy((rotateXY90(flipZ(torch.load(path).cpu().detach().numpy()), 1)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 6:
                image = torch.from_numpy((rotateXY90(flipZ(torch.load(path).cpu().detach().numpy()), 2)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 7:
                image = torch.from_numpy((rotateXY90(flipZ(torch.load(path).cpu().detach().numpy()), 3)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 8:
                image = torch.from_numpy((rotateXY90(flipX(torch.load(path).cpu().detach().numpy()), 0)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 9:
                image = torch.from_numpy((rotateXY90(flipX(torch.load(path).cpu().detach().numpy()), 1)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 10:
                image = torch.from_numpy((rotateXY90(flipX(torch.load(path).cpu().detach().numpy()), 2)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 11:
                image = torch.from_numpy((rotateXY90(flipX(torch.load(path).cpu().detach().numpy()), 3)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 12:
                image = torch.from_numpy((rotateXY90(flipX(flipZ(torch.load(path).cpu().detach().numpy())), 0)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 13:
                image = torch.from_numpy((rotateXY90(flipX(flipZ(torch.load(path).cpu().detach().numpy())), 1)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 14:
                image = torch.from_numpy((rotateXY90(flipX(flipZ(torch.load(path).cpu().detach().numpy())), 2)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

            elif int(idx / self.data_len) == 15:
                image = torch.from_numpy((rotateXY90(flipX(flipZ(torch.load(path).cpu().detach().numpy())), 3)).copy())
                f1 = image[0, 0, 0, :, :][None, :, :]
                f2 = image[0, 2, 0, :, :][None, :, :]
                gt = image[0, 1, 0, :, :][None, :, :]

        else:

            image = torch.from_numpy((rotateXY90(torch.load(path).cpu().detach().numpy(), 0)).copy())
            f1 = image[0, 0, 0, :, :][None, :, :]
            f2 = image[0, 2, 0, :, :][None, :, :]
            gt = image[0, 1, 0, :, :][None, :, :]

        return {"f1": f1, "f2": f2, "gt": gt}

    def __len__(self):

        return self.len
