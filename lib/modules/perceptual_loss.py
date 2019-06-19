#!/usr/bin/env python

import math, cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from modules.separable_convolution import same_padding_size
from collections import namedtuple


LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"])


class VGG19LossNetwork(torch.nn.Module):
    def __init__(self, config):
        super(VGG19LossNetwork, self).__init__()
        self.cfg = config
        original_model = models.vgg19(pretrained=True)
        self.vgg_layers = original_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }


    def preprocess(self, x):
        if not (hasattr(self, 'mean') and hasattr(self, 'std')):
            self.mean = x.new_ones((1, 3))
            self.std = x.new_ones((1, 3))
            self.mean[0, 0] = 0.485; self.mean[0, 1] = 0.456; self.mean[0, 2] = 0.406
            self.std[0, 0] = 0.229; self.std[0, 1] = 0.224; self.std[0, 2] = 0.225
            self.mean = self.mean.view(1,3,1,1)
            self.std = self.std.view(1,3,1,1)

        return (x/255.0 - self.mean)/self.std


    def forward(self, inputs):
        x = self.preprocess(inputs)

        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # output[self.layer_name_mapping[name]] = x
                output.append(x.clone())
        # return LossOutput(**output)
        return output
