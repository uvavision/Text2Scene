#!/usr/bin/env python

import math, cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from modules.separable_convolution import same_padding_size
# from modules.bilinear_downsample import BilinearDownsample
import torch.nn.functional as F


sp = 256
in_channels = 24

class CRNDecoder(nn.Module):
    def __init__(self, config):
        super(CRNDecoder, self).__init__()
        self.cfg = config

        self.block6 = nn.Sequential(self.make_layers(512 + in_channels, [512, 512], True, [sp//64, sp//32]))
        self.block5 = nn.Sequential(self.make_layers(512 + 512 + in_channels, [512, 512], True, [sp//32, sp//16]))
        self.block4 = nn.Sequential(self.make_layers(512 + 512 + in_channels, [512, 512], True, [sp//16, sp//8]))
        self.block3 = nn.Sequential(self.make_layers(512 + 256 + in_channels, [512, 512], True, [sp//8, sp//4]))
        self.block2 = nn.Sequential(self.make_layers(512 + 128 + in_channels, [512, 512], True, [sp//4, sp//2]))
        self.block1 = nn.Sequential(self.make_layers(512 + 64 + in_channels, [256, 256], True, [sp//2, sp]))
        self.block0 = nn.Sequential(self.make_layers(256 + in_channels, [256, 256], True, [sp, sp*2]))
        # self.block_up = nn.Sequential(self.make_layers(256 + in_channels, [128, 128], True, [sp, sp*2]))

        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        self.imaging  = conv2d(256, 3, 1)
        self.labeling = conv2d(256, in_channels-4, 1)

    def make_layers(self, inplace, places, use_layer_norm=False, resolution=None):
        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        layers = []
        in_channels = inplace

        for v in places:
            if use_layer_norm:
                current_conv2d = conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                current_lnorm = nn.LayerNorm([v, resolution[0], resolution[1]])
                layers.extend([current_conv2d, current_lnorm])
            else:
                current_conv2d = conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                layers.append(current_conv2d)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, inputs):
        xx, x1, x2, x3, x4, x5, x6 = inputs

        xx_d6 = F.interpolate(xx, size=[sp//64, sp//32], mode='bilinear', align_corners=True) #self.downsample6(x0)
        x = torch.cat([xx_d6, x6], dim=1)
        x = self.block6(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d5 = F.interpolate(xx, size=[sp//32, sp//16], mode='bilinear', align_corners=True)#self.downsample5(x0)
        x = torch.cat([xx_d5, x5, x], dim=1)
        x = self.block5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d4 = F.interpolate(xx, size=[sp//16, sp//8], mode='bilinear', align_corners=True)#self.downsample4(x0)
        x = torch.cat([xx_d4, x4, x], dim=1)
        x = self.block4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d3 = F.interpolate(xx, size=[sp//8, sp//4], mode='bilinear', align_corners=True)#self.downsample3(x0)
        x = torch.cat([xx_d3, x3, x], dim=1)
        x = self.block3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d2 = F.interpolate(xx, size=[sp//4, sp//2], mode='bilinear', align_corners=True)#self.downsample2(x0)
        x = torch.cat([xx_d2, x2, x], dim=1)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d1 = F.interpolate(xx, size=[sp//2, sp], mode='bilinear', align_corners=True)#self.downsample1(x0)
        x = torch.cat([xx_d1, x1, x], dim=1)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # x = torch.cat([x0, x], dim=1)
        # x = self.block0(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d0 = F.interpolate(xx, size=[sp, 2*sp], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d0, x], dim=1)
        x = self.block0(x)

        image = self.imaging(x)
        label = self.labeling(x)

        image = (image+1.0)/2.0

        return image, label
