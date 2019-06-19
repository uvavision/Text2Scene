#!/usr/bin/env python

import cv2, random, json
import numpy as np

import torch
import torch.nn as nn

from modules.encoder import ImageAndLayoutEncoder
from modules.crn_decoder import CRNDecoder
from modules.perceptual_loss import VGG19LossNetwork


class ImageSynthesisModel(nn.Module):
    def __init__(self, config):
        super(ImageSynthesisModel, self).__init__()
        self.cfg = config
        self.encoder = ImageAndLayoutEncoder(self.cfg)
        self.decoder = CRNDecoder(self.cfg)
        self.lossnet = VGG19LossNetwork(self.cfg).eval()

    def forward(self, proposal_inputs, compute_perceptual_features=False, gt_images=None):
        # x0, x1, x2, x3, x4, x5, x6 = self.encoder(proposal_inputs)
        # y = (x0, x1, x2, x3, x4, x5, x6)
        y = self.encoder(proposal_inputs)
        synthesized_images, synthesized_labels = self.decoder(y)
        synthesized_features, gt_features = None, None
        if compute_perceptual_features:
            synthesized_features = self.lossnet(synthesized_images)
            if gt_images is not None:
                with torch.no_grad():
                    gt_features = self.lossnet(gt_images.detach())

        return synthesized_images, synthesized_labels, synthesized_features, gt_features
