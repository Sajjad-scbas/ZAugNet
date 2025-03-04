import torch
import torch.nn as nn
import torch.nn.functional as F

backwarp_tenGrid = {}


def warp(tenInput, tenFlow, device):
    k = (str(tenFlow.device), str(tenFlow.size()))
    # creating a normalized coordinate grid (from -1 to 1)
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(1, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow, x.device)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow, x.device) 
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow, x.device) 
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow, x.device) 
        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(9, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 1, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)
    

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class ZAugGenerator(nn.Module):
    def __init__(self, cfg):
        super(ZAugGenerator, self).__init__()
        self.cfg = cfg
        if self.cfg.model_name == 'zaugnet+':
            self.block0 = IFBlock(2+1, c=240)
            self.block1 = IFBlock(5+4+1, c=150)
            self.block2 = IFBlock(5+4+1, c=90)
            self.block_tea = IFBlock(6+4+1, c=90)
        else : 
            self.block0 = IFBlock(2, c=240)
            self.block1 = IFBlock(5+4, c=150)
            self.block2 = IFBlock(5+4, c=90)
            self.block_tea = IFBlock(6+4, c=90)

        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4,2,1], DPM=0.5, gt_bool = True):
        if self.cfg.model_name == 'zaugnet+':
            DPM = (x[:,:1].clone() * 0 + 1)*DPM
        img0 = x[:, 0][:, None]
        img1 = x[:, 1][:, None]
        if gt_bool :
            gt = x[:, 2][:, None] # In inference time, gt is None
        else : 
            gt = None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                if self.cfg.model_name == 'zaugnet+':
                    flow_d, mask_d = stu[i](x=torch.cat((img0, img1, DPM, warped_img0, warped_img1, mask), 1), flow=flow, scale=scale[i])
                else :
                    flow_d, mask_d = stu[i](x=torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow=flow, scale=scale[i])

                flow = flow + flow_d
                mask = mask + mask_d
            else:
                if self.cfg.model_name == 'zaugnet+':
                    flow, mask = stu[i](x=torch.cat((img0, img1, DPM), 1), flow=None, scale=scale[i])
                else :
                    flow, mask = stu[i](x=torch.cat((img0, img1), 1), flow=None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], img0.device)
            warped_img1 = warp(img1, flow[:, 2:4], img1.device)
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt is not None:
            if self.cfg.model_name == 'zaugnet+':
                flow_d, mask_d = self.block_tea(x=torch.cat((img0, img1, DPM, warped_img0, warped_img1, mask, gt), 1), flow=flow, scale=1)
            else:
                flow_d, mask_d = self.block_tea(x=torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow=flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2], img0.device)
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4], img1.device)
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt_bool:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                # loss_mask = 1
                loss_distill = loss_distill + (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return [flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill]
    
    def set_multiple_gpus(self):
        self.block0 = nn.DataParallel(self.block0, device_ids=self.cfg.device_ids).cuda()
        self.block0.to(f"cuda:{self.cfg.device_ids[0]}")
        self.block1 = nn.DataParallel(self.block1, device_ids=self.cfg.device_ids).cuda()
        self.block1.to(f"cuda:{self.cfg.device_ids[0]}")
        self.block2 = nn.DataParallel(self.block2, device_ids=self.cfg.device_ids).cuda()
        self.block2.to(f"cuda:{self.cfg.device_ids[0]}")
        self.block_tea = nn.DataParallel(self.block_tea, device_ids=self.cfg.device_ids).cuda()
        self.block_tea.to(f"cuda:{self.cfg.device_ids[0]}")
        self.contextnet = nn.DataParallel(self.contextnet, device_ids=self.cfg.device_ids).cuda()
        self.contextnet.to(f"cuda:{self.cfg.device_ids[0]}")
        self.unet = nn.DataParallel(self.unet, device_ids=self.cfg.device_ids).cuda()
        self.unet.to(f"cuda:{self.cfg.device_ids[0]}")


class ZAugDiscriminator(nn.Module):
    """
    ZAugNet Discriminator class.
    """
    def __init__(self):
        super(ZAugDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            # input is 256 x 256
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 128 x 128
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 64 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 16 x 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 8 x 8
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 4 x 4
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, image, DPM=None):
        if DPM != None:
            image = torch.cat([image, DPM], dim=1)
        image = self.layers(image)
        image = image.view(image.shape[0], -1)
        return image
