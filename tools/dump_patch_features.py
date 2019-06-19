#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from config import get_config
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from utils import *

from datasets.coco import coco
from datasets.coco_loader import patch_loader
from modules.encoder import FeatureExtractor
import torch, torchtext
from torch.utils.data import Dataset, DataLoader


def batch_dump_patch_features(config, split, year):
    db = coco(config, split, year)
    data_dir = osp.join(config.data_dir, 'coco')

    if config.use_patch_background:
        if (split == 'test') or (split == 'aux'):
            main_dir = osp.join(data_dir, 'patch_resnet152_with_bg', 'train' + year)
        else:
            main_dir = osp.join(data_dir, 'patch_resnet152_with_bg', split + year)
    else:
        if (split == 'test') or (split == 'aux'):
            main_dir = osp.join(data_dir, 'patch_resnet152_without_bg', 'train' + year)
        else:
            main_dir = osp.join(data_dir, 'patch_resnet152_without_bg', split + year)
    maybe_create(main_dir)

    patch_loader_db = patch_loader(db, 'patch_color_large', 'jpg', config.use_patch_background, True)
    loader = DataLoader(patch_loader_db, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True)

    net = FeatureExtractor(config)
    if config.cuda:
        net = net.cuda()
    
    for cnt, batched in enumerate(loader):
        start = time()
        patch_inds = batched['patch_ind'].long()
        patch_data = batched['patch_data'].float()
        if config.cuda:
            patch_data = patch_data.cuda(non_blocking=True)
        patch_features = net(patch_data)
        patch_features = patch_features.cpu().data.numpy()
        for i in range(patch_data.size(0)):
            patch = db.patchdb[patch_inds[i]]
            image_index = patch['image_index']
            instance_ind = patch['instance_ind']
            patch_feature_path = db.patch_path_from_indices(image_index, instance_ind, 'patch_resnet152', 'pkl', config.use_patch_background)
            patch_feature_dir = osp.dirname(patch_feature_path)
            maybe_create(patch_feature_dir)
            features = patch_features[i].flatten()
            with open(patch_feature_path, 'wb') as fid:
                pickle.dump(features, fid, pickle.HIGHEST_PROTOCOL)
        print('current_ind: %d, time consumed: %f'%(cnt, time() - start))


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    batch_dump_patch_features(config, 'train', '2017')
    batch_dump_patch_features(config, 'val',   '2017')
    batch_dump_patch_features(config, 'test',  '2017')
    batch_dump_patch_features(config, 'aux',   '2017')
