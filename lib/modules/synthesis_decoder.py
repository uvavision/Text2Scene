#!/usr/bin/env python

import math, cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from modules.separable_convolution import same_padding_size, separable_conv2d
import torch.nn.functional as F


class SynthesisDecoder(nn.Module):
    def __init__(self, config):
        super(SynthesisDecoder, self).__init__()
        self.cfg = config

        h, w = config.output_image_size
        # if config.use_color_volume:
        #     in_channels = 3 * config.output_vocab_size 
        # else:
        #     in_channels = config.output_vocab_size + 4
        in_channels = config.output_vocab_size + 4

        self.block6 = nn.Sequential(self.make_layers(512 + in_channels,       [512, 512], config.use_normalization, [h//64, w//64]))
        self.block5 = nn.Sequential(self.make_layers(512 + 512 + in_channels, [512, 512], config.use_normalization, [h//32, w//32]))
        self.block4 = nn.Sequential(self.make_layers(512 + 512 + in_channels, [512, 512], config.use_normalization, [h//16, w//16]))
        self.block3 = nn.Sequential(self.make_layers(512 + 256 + in_channels, [512, 512], config.use_normalization, [h//8,  w//8]))
        self.block2 = nn.Sequential(self.make_layers(512 + 256 + in_channels, [512, 512], config.use_normalization, [h//4,  w//4]))
        self.block1 = nn.Sequential(self.make_layers(512 + 256 + in_channels, [256, 256], config.use_normalization, [h//2,  w//2]))
        self.block0 = nn.Sequential(self.make_layers(256 + in_channels,       [256, 256], config.use_normalization, [h,     w]))
        # self.block_up = nn.Sequential(self.make_layers(256 + in_channels, [128, 128], config.use_normalization, [2*h, 2*w]))

        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        self.imaging  = conv2d(256, 3, 1)
        self.labeling = conv2d(256, config.output_vocab_size, 1)

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
        h, w = self.cfg.output_image_size

        xx, x1, x2, x3, x4, x5, x6 = inputs

        xx_d6 = F.interpolate(xx, size=[h//64, w//64], mode='bilinear', align_corners=True) 
        x = torch.cat([xx_d6, x6], dim=1)
        x = self.block6(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d5 = F.interpolate(xx, size=[h//32, w//32], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d5, x5, x], dim=1)
        x = self.block5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d4 = F.interpolate(xx, size=[h//16, w//16], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d4, x4, x], dim=1)
        x = self.block4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d3 = F.interpolate(xx, size=[h//8, w//8], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d3, x3, x], dim=1)
        x = self.block3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d2 = F.interpolate(xx, size=[h//4, w//4], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d2, x2, x], dim=1)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d1 = F.interpolate(xx, size=[h//2, w//2], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d1, x1, x], dim=1)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # x = torch.cat([x0, x], dim=1)
        # x = self.block0(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        xx_d0 = xx #F.interpolate(xx, size=[h, w], mode='bilinear', align_corners=True)
        x = torch.cat([xx_d0, x], dim=1)
        x = self.block0(x)

        image = self.imaging(x)
        label = self.labeling(x)

        image = 255.0*(image+1.0)/2.0

        return image, label
