#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from nntable import AllCategoriesTables

from composites_utils import *
from composites_config import get_config
from datasets.composites_coco import composites_coco
from datasets.composites_loader import sequence_loader, synthesis_loader
from modules.synthesis_model import SynthesisModel
from modules.composites_encoder import SynthesisEncoder
from modules.perceptual_loss import VGG19LossNetwork
from modules.synthesis_decoder import SynthesisDecoder
from modules.synthesis_model import SynthesisModel


import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def test_perceptual_loss_network(config):
    img_encoder = VGG19LossNetwork(config).eval()
    print(get_n_params(img_encoder))

    db = composites_coco(config, 'train', '2017')
    syn_loader = synthesis_loader(db)
    loader = DataLoader(syn_loader, batch_size=1, 
        shuffle=False, num_workers=config.num_workers)

    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['gt_image'].float()
        y = img_encoder(x.permute(0,3,1,2))
        for z in y:
            print(z.size())
        break


def test_syn_encoder(config):
    img_encoder = SynthesisEncoder(config)
    print(get_n_params(img_encoder))

    db = composites_coco(config, 'train', '2017')
    syn_loader = synthesis_loader(db)
    loader = DataLoader(syn_loader, batch_size=1, 
        shuffle=False, num_workers=config.num_workers)
    
    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['input_vol'].float()
        y = img_encoder(x)
        for z in y:
            print(z.size())
        break


def test_syn_decoder(config):
    img_encoder = SynthesisEncoder(config)
    img_decoder = SynthesisDecoder(config)
    print(get_n_params(img_encoder))
    print(get_n_params(img_decoder))

    db = composites_coco(config, 'train', '2017')
    syn_loader = synthesis_loader(db)
    loader = DataLoader(syn_loader, batch_size=1, 
        shuffle=False, num_workers=config.num_workers)

    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['input_vol'].float()
        x0, x1, x2, x3, x4, x5, x6 = img_encoder(x)
        inputs = (x0, x1, x2, x3, x4, x5, x6)
        image, label = img_decoder(inputs)
        print(image.size(), label.size())
        break


def test_syn_model(config):
    synthesizer = SynthesisModel(config)
    print(get_n_params(synthesizer))

    db = composites_coco(config, 'train', '2017')
    syn_loader = synthesis_loader(db)
    loader = DataLoader(syn_loader, batch_size=1, 
        shuffle=False, num_workers=config.num_workers)

    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['input_vol'].float()
        y = batched['gt_image'].float()
        z = batched['gt_label'].long()
        y = y.permute(0,3,1,2)
        image, label, syn_feats, gt_feats = synthesizer(x, True, y)
        print(image.size(), label.size())
        for v in syn_feats:
            print(v.size())
        print('------------')
        for v in gt_feats:
            print(v.size())
        break


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # test_perceptual_loss_network(config)
    # test_syn_encoder(config)
    # test_syn_decoder(config)
    test_syn_model(config)
