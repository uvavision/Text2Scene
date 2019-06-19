#!/usr/bin/env python

import torch
import torch.nn as nn

def same_padding_size(kernel_size, dilation_rate=1):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end


class separable_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
            kernel_size, stride=1,
            padding=0, dilation=1,
            groups=1, bias=True):
        super(separable_conv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
