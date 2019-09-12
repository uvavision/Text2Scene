#!/usr/bin/env python

import cv2, random, json
import numpy as np

import torch
import torch.nn as nn

from modules.composites_encoder import SynthesisEncoder
from modules.perceptual_loss import VGG19LossNetwork
from modules.synthesis_decoder import SynthesisDecoder


class SynthesisModel(nn.Module):
    def __init__(self, config):
        super(SynthesisModel, self).__init__()
        self.cfg = config
        self.encoder = SynthesisEncoder(self.cfg)
        self.decoder = SynthesisDecoder(self.cfg)
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
                z = gt_images.clone()
                # z = z.permute(0,3,1,2)
                with torch.no_grad():
                    gt_features = self.lossnet(z)

        return synthesized_images, synthesized_labels, synthesized_features, gt_features