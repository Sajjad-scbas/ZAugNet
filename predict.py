import os

import numpy as np
import tifffile as tiff

import torch.cuda
import torch.nn as nn

from utils import crop_3d

from model import ZAugGenerator

if not os.path.exists('./output'):
    os.makedirs('./output')

if not os.path.exists('./output/false'):
    os.makedirs('./output/false')

for file in os.listdir('.'):
    if file.endswith(".pt"):
        model = str(file)
        break
    else:
        pass

patch_sz = 256
last_kern_sz = 32

list_of_files = os.listdir('./test')

for num, filename in enumerate(list_of_files):

    path = './test/' + filename

    path_model = './' + str(model)

    channel = 0

    image = tiff.imread(path).astype('uint8')

    len_shape = len(image.shape)
    if len_shape <= 2:
        print('Error: images has to be 3d at least. ')
        exit()
    elif len_shape == 3:
        image = image[None, :, None, :, :]
    elif len_shape == 4:
        if num == 0:
            format = input('are data time-lapses? [y/n]  ')
            old_format = format
        else:
            format = old_format
        if format == 'y':
            image = image[:, :, None, :, :]
        elif format == 'n':
            image = image[None, :, :, :, :]
        else:
            print('Error: data file-format not recognized.')
            exit()
    elif len_shape == 5:
        pass
    else:
        print('Error: images has to be 5d at maximum.')
        exit()

    cropped_image = crop_3d(image, image.shape[1], patch_sz, patch_sz)
    t, z, c, h, w = cropped_image.shape

    pt_cropped_image = torch.from_numpy(cropped_image.copy())

    cuda = True if torch.cuda.is_available() else False
    print(f"using cuda device: {cuda}")

    if torch.cuda.device_count() > 1:
        print("using", torch.cuda.device_count(), "GPUs for predicting")

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    zaug = ZAugGenerator(last_kernel_size=last_kern_sz)

    if cuda:
        zaug = nn.DataParallel(zaug).cuda()  # parallelized on all available GPUs
        zaug.load_state_dict(torch.load(path_model))
        zaug.eval()
    else:
        zaug.load_state_dict(torch.load(path_model))
        zaug.eval()

    fake_timelapse = np.zeros((t, int(z*2)-1, 1, h, w))

    for dt in range(t):
        for dz in range(z-1):
            with torch.no_grad():
                # print(dz)
                frame_1 = (np.true_divide(pt_cropped_image[dt, dz, 0, :, :], 255, dtype=np.float32)).type(Tensor)[None, None, :, :]
                frame_2 = (np.true_divide(pt_cropped_image[dt, dz+1, 0, :, :], 255, dtype=np.float32)).type(Tensor)[None, None, :, :]

                # frames_1_2 = torch.cat((frame_1, frame_2), 2)
                #
                # frames_1_2_mean = torch.mean(frames_1_2)
                # frames_1_2_std = torch.std(frames_1_2)
                #
                # frame_1 = (frame_1 - frames_1_2_mean) / frames_1_2_std
                # frame_2 = (frame_2 - frames_1_2_mean) / frames_1_2_std

                fake_frame = zaug(frame_1, frame_2)

                fake_frame = fake_frame * 255
                frame_1 = frame_1 * 255
                frame_2 = frame_2 * 255

                # fake_frame = ((fake_frame * frames_1_2_std) + frames_1_2_mean) * 255
                # frame_1 = ((frame_1 * frames_1_2_std) + frames_1_2_mean) * 255
                # frame_2 = ((frame_2 * frames_1_2_std) + frames_1_2_mean) * 255

            frame_1_image = frame_1.detach().cpu().numpy()
            frame_2_image = frame_2.detach().cpu().numpy()
            fake_frame_image = fake_frame.detach().cpu().numpy()

            # tiff.imwrite('./output/false/' + str(filename[:-4]) + '-dt'+str(dt)+ '-dz' + str(2 * dz + 1) + '.tif', np.clip(fake_frame_image[0, 0, None, :, :], 0, 255).astype('uint8'), imagej=True)

            fake_timelapse[dt, 2 * dz + 0, :, :, :] = np.clip(frame_1_image[0, 0, None, :, :], 0, 255).astype('uint8')
            fake_timelapse[dt, 2 * dz + 2, :, :, :] = np.clip(frame_2_image[0, 0, None, :, :], 0, 255).astype('uint8')
            fake_timelapse[dt, 2 * dz + 1, :, :, :] = np.clip(fake_frame_image[0, 0, None, :, :], 0, 255).astype('uint8')

    tif_path = './output/' + str(filename[:-4]) +'-z' + str(int(z*2)-1) + '.tif'
    tiff.imwrite(tif_path, fake_timelapse.astype('uint8'), imagej=True)
