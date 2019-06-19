#!/usr/bin/env python

import _init_paths
import cv2,torch
import torch.nn.functional as F
from modules.bilinear_downsample import BilinearDownsample
from config import get_config

config, unparsed = get_config()

# Create fake image
image = torch.zeros(1, 3, 224, 224)
image[0, :, 10:110, 10:110] = 1.

downsampler = BilinearDownsample(config, [100, 50])
image_small = downsampler(image)


# # Create grid
# out_size = 112
# x = torch.linspace(-1, 1, out_size).view(-1, 1).repeat(1, out_size)
# y = torch.linspace(-1, 1, out_size).repeat(out_size, 1)
# print('x.size()', x.size())
# print('y.size()', y.size())
# grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
# print('grid.size()', grid.size())
# grid.unsqueeze_(0)
# print('grid.size()', grid.size())
#
# image_small = F.grid_sample(image, grid, mode='bilinear', padding_mode='border')

cv2.imshow('image1',image[0].permute(1, 2, 0).numpy())
cv2.imshow('image2',image_small[0].permute(1, 2, 0).numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()
