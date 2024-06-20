import os

import numpy as np
import random

import torch


def str_to_bool(string: str):
    """
    It converts the string in input into a boolean.

    Args:
        string: (str) string to convert
    Return:
        boolean: (bool) boolean value for the string
    """

    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        print('Error: typo in one of the boolean settings. ')
        exit()


def read_settings():
    """
    It reads the settings from the 'settings.txt' file in the root folder.

    Args:
        (void)
    Return:
        settings: (dict) settings for training
    """

    with open('./settings.txt') as file:
        lines = [line.rstrip() for line in file]
        par = [i.split('\t', 1)[1] for i in lines]

    settings = {'augmentations': par[0],
                'batch_size': par[1],
                'beta1': par[2],
                'beta2': par[3],
                'lambda_adv': par[4],
                'lambda_gp': par[5],
                'last_kernel_size': par[6],
                'loss': par[7],
                'lr': par[8],
                'n_critic': par[9],
                'n_epochs': par[10],
                'normalization': par[11],
                'patch_size': par[12],
                'zscore': par[13]}

    return settings


def crop_3d(image: np.ndarray,
           z_size: int = 0,
           y_size: int = 0,
           x_size: int = 0):
    """
    It crops the 3 spatial dimensions x, y, z of the 5d image in input starting from the central pixel.

    Args:
        image: (np.ndarray) image to crop
        z_size: (int) z final size
        y_size: (int) y final size
        x_size: (int) x final size

    Return:
        image: (np.ndarray) cropped image
    """

    if x_size == 0 or y_size == 0 or z_size == 0:
        print("Error: zero sizes not allowed.")
        exit()

    else:
        t, z, c, h, w = image.shape

        half_h = np.abs(h - y_size) / 2
        if (half_h - int(half_h)) == 0:
            down = int(half_h)
            up = int(half_h)
        else:
            down = int(half_h + 0.5)
            up = int(half_h - 0.5)

        half_w = np.abs(w - x_size) / 2
        if (half_w - int(half_w)) == 0.:
            right = int(half_w)
            left = int(half_w)
        else:
            right = int(half_w + 0.5)
            left = int(half_w - 0.5)

        half_z = np.abs(z - z_size) / 2
        if (half_z - int(half_z)) == 0.:
            bottom = int(half_z)
            top = int(half_z)
        else:
            bottom = int(half_z + 0.5)
            top = int(half_z - 0.5)

        return image[:, bottom:z - top, :, down:h - up, left:w - right]


def create_zdataset(filename: str,
                    image_fill: np.ndarray,
                    channel: int,
                    normalization: bool = True,
                    zscore: bool = False):
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

    t, z, c, h, w = image_fill.shape

    folder = 'dataset-ch' + str(channel)
    os.makedirs('./datasets/' + str(folder), exist_ok=True)

    for i in range(t):

        pt_path = './datasets/' + str(folder)
        three_consecutive_images = np.zeros((1, 3, 1, h, w))

        for j in range(z - 2):

            image0 = image_fill[i, j, channel, :, :]
            image1 = image_fill[i, j + 1, channel, :, :]
            image2 = image_fill[i, j + 2, channel, :, :]

            if np.std(image0) != 0. and np.std(image1) != 0. and np.std(image2) != 0.:

                if normalization:
                    # range [0, 1]
                    image0 = np.true_divide(image0, 255, dtype=np.float32)
                    image1 = np.true_divide(image1, 255, dtype=np.float32)
                    image2 = np.true_divide(image2, 255, dtype=np.float32)
                else:
                    pass

                if zscore:
                    # mean 0 std 1
                    image012 = np.concatenate((image0, image1, image2), axis=0)
                    image012_mean = np.mean(image012)
                    image012_std = np.std(image012)
                    image0 = (image0 - image012_mean) / image012_std
                    image1 = (image1 - image012_mean) / image012_std
                    image2 = (image2 - image012_mean) / image012_std
                else:
                    pass

                if normalization and zscore:
                    # range [0, 1]
                    image0 = np.true_divide(image0, 255, dtype=np.float32)
                    image1 = np.true_divide(image1, 255, dtype=np.float32)
                    image2 = np.true_divide(image2, 255, dtype=np.float32)
                    # mean 0 std 1
                    image012 = np.concatenate((image0, image1, image2), axis=0)
                    image012_mean = np.mean(image012)
                    image012_std = np.std(image012)
                    image0 = (image0 - image012_mean) / image012_std
                    image1 = (image1 - image012_mean) / image012_std
                    image2 = (image2 - image012_mean) / image012_std
                else:
                    pass

                three_consecutive_images[0, 0, 0, :, :] = image0
                three_consecutive_images[0, 1, 0, :, :] = image1
                three_consecutive_images[0, 2, 0, :, :] = image2

                torch.save(torch.from_numpy(three_consecutive_images.copy()), str(pt_path) + '/' + filename[:-4] + '-t' + str(i) + '-z' + str(j) + '.pt')

            else:
                pass


def rotateXY90(image: np.ndarray,
               k: int):
    """
    It rotates the the 5d image by k*90 degrees over the XY plane.

    Args:
        image: (np.ndarray) image to rotate
        k: (int) number of rotations
    Returns:
        rotated_image: (np.ndarray) rotated image
    """

    rotated_image = np.rot90(image, k=k, axes=(3, 4))
    return rotated_image


def flipX(image: np.ndarray):
    """
    It flips the 5d image over the X dimension (horizontally).

    Args:
        image: (np.ndarray) image to flip
    Returns:
        flipped_image: (np.ndarray) flipped image
    """

    flipped_image = np.flip(image, axis=4)
    return flipped_image


def flipY(image: np.ndarray):
    """
    It flips the 5d image over the Y dimension (vertically).

    Args:
        image: (np.ndarray) image to flip
    Returns:
        flipped_image: (np.ndarray) flipped image
    """

    flipped_image = np.flip(image, axis=3)
    return flipped_image


def flipZ(image: np.ndarray):
    """
    It flips the 5D image over the Z dimension.

    Args:
        image: (np.ndarray) image to flip
    Returns:
        flipped_image: (np.ndarray) flipped image
    """

    flipped_image = np.flip(image, axis=1)
    return flipped_image


def mean_absolute_error(fake_frame: np.array,
                        true_frame: np.array):
    """
    It computes the mean absolute error between the predicted 3d frame and the 3d ground truth.

    Args:
        fake_frame: (np.array) predicted frame
        true_frame: (np.array) ground truth

    Returns:
        mae: (float) mean absolute error
    """

    return np.mean(np.abs(np.subtract(fake_frame, true_frame)))


def mean_squared_error(fake_frame: np.array,
                       true_frame: np.array):
    """
    It computes the mean squared error between the 3d predicted frame and the 3d ground truth.

    Args:
        fake_frame: (np.array) predicted frame
        true_frame: (np.array) ground truth

    Returns:
        mse: (float) mean squared error
    """

    return np.mean(np.square(np.subtract(fake_frame, true_frame)))


def peak_signal_to_noise_ratio(fake_frame: np.array,
                               true_frame: np.array):
    """
    It computes the peak signal-to-noise ratio between the 3d predicted frame and the 3d ground truth.

    Args:
        fake_frame: (np.array) predicted frame
        true_frame: (np.array) ground truth

    Returns:
        psnr: (float) peak signal-to-noise ratio
    """

    mse = np.mean(np.square(np.subtract(fake_frame, true_frame)))
    return 10 * np.log10(255 / mse)


def structural_similarity_index(fake_frame: np.array,
                                true_frame: np.array,
                                C1=0.01,
                                C2=0.03):
    """
    It computes the structural similarity index between the 3d predicted frame and the 3d ground truth.

    Args:
        fake_frame: (np.array) predicted frame
        true_frame: (np.array) ground truth
        C1: (float) variable to stabilize the denominator
        C2: (float) variable to stabilize the denominator

    Returns:
        ssim: (float) structural similarity index
    """

    mean_true = true_frame.mean()
    mean_generated = fake_frame.mean()
    std_true = true_frame.std()
    std_generated = fake_frame.std()
    covariance = ((true_frame - mean_true) * (fake_frame - mean_generated)).mean()

    numerator = (2 * mean_true * mean_generated + C1) * (2 * covariance + C2)
    denominator = ((mean_true ** 2 + mean_generated ** 2 + C1) * (std_true ** 2 + std_generated ** 2 + C2))

    return numerator / denominator





















###############################
##### OTHER FUNCTIONS ########
###############################



def circularCrop2D(image: np.ndarray,
                   center: tuple = None,
                   radius: float = None):
    """
    It applies a circular mask to the the 2d image in input, with a circle defined by center and radius in input.
    If center=None means middle of the image. If radius=None means smallest distance between the center and image walls.

    Args:
        image: (np.ndarray) image to crop
        center: (tuple) center circle
        radius: (float) radius circle

    Return:
        image: (np.ndarray) circular cropped image
    """

    h, w = image.shape

    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    image[~mask] = 0

    return image


def normalize01_8bit(image: np.ndarray):
    """
    It normalizes the 8bit 2d-image in input in the range [0,1].

    Args:
        image: (np.ndarray) image to normalize

    Return:
        image: (np.ndarray) normalized image
    """

    return np.true_divide(image, 255, dtype=np.float32)