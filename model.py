import math

import torch
import torch.nn as nn

import sepconv  # Custom separable convolution layer written in CUDA by Niklaus et al. (2017)


class ZAugGenerator(nn.Module):
    """
    ZAugNet Generator class.
    """

    def __init__(self, last_kernel_size):
        super(ZAugGenerator, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=last_kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=last_kernel_size, out_channels=last_kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=last_kernel_size, out_channels=last_kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=last_kernel_size, out_channels=last_kernel_size, kernel_size=3, stride=1, padding=1)
            )

        self.netConv1 = Basic(2, 32)
        self.netConv2 = Basic(32, 64)
        self.netConv3 = Basic(64, 128)
        self.netConv4 = Basic(128, 256)
        self.netConv5 = Basic(256, 512)

        self.netDeconv5 = Basic(512, 512)
        self.netDeconv4 = Basic(512, 256)
        self.netDeconv3 = Basic(256, 128)
        self.netDeconv2 = Basic(128, 64)

        self.netUpsample5 = Upsample(512, 512)
        self.netUpsample4 = Upsample(256, 256)
        self.netUpsample3 = Upsample(128, 128)
        self.netUpsample2 = Upsample(64, 64)

        self.netVertical1 = Subnet()
        self.netVertical2 = Subnet()
        self.netHorizontal1 = Subnet()
        self.netHorizontal2 = Subnet()

        self.last_kernel_size = last_kernel_size

    def forward(self, tenOne, tenTwo):

        tenConv1 = self.netConv1(torch.cat([tenOne, tenTwo], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

        tenCombine = tenDeconv2 + tenConv2

        tenOne = torch.nn.functional.pad(input=tenOne, pad=[int(math.floor(self.last_kernel_size / 2.0)), int(math.floor(self.last_kernel_size / 2.0)), int(math.floor(self.last_kernel_size / 2.0)), int(math.floor(self.last_kernel_size / 2.0))], mode='replicate')
        tenTwo = torch.nn.functional.pad(input=tenTwo, pad=[int(math.floor(self.last_kernel_size / 2.0)), int(math.floor(self.last_kernel_size / 2.0)), int(math.floor(self.last_kernel_size / 2.0)), int(math.floor(self.last_kernel_size / 2.0))], mode='replicate')

        tenDot1 = sepconv.sepconv_func.apply(tenOne, self.netVertical1(tenCombine), self.netHorizontal1(tenCombine))
        tenDot2 = sepconv.sepconv_func.apply(tenTwo, self.netVertical2(tenCombine), self.netHorizontal2(tenCombine))

        return tenDot1 + tenDot2


class ZAugDiscriminator(nn.Module):
    """
    ZAugNet Discriminator class.
    """

    def __init__(self):
        super(ZAugDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            # input is 256 x 256
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 128
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 4
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, image):
        image = self.layers(image)
        return image.view(image.shape[0], -1)
