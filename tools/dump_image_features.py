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
from datasets.coco_loader import image_loader
from modules.encoder import FeatureExtractor
import torch, torchtext
from torch.utils.data import Dataset, DataLoader


def batch_dump_image_features(config, split, year):
    db = coco(config, split, year)
    data_dir = osp.join(config.data_dir, 'coco')

    if (split == 'test') or (split == 'aux'):
        main_dir = osp.join(data_dir, 'image_resnet152', 'train' + year)
    else:
        main_dir = osp.join(data_dir, 'image_resnet152', split + year)
    maybe_create(main_dir)

    image_loader_db = image_loader(db, 'images', 'jpg', True)
    loader = DataLoader(image_loader_db, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True)

    net = FeatureExtractor(config)
    if config.cuda:
        net = net.cuda()
    
    for cnt, batched in enumerate(loader):
        start = time()
        image_inds = batched['image_ind'].long()
        image_data = batched['image_data'].float()
        if config.cuda:
            image_data = image_data.cuda(non_blocking=True)
        image_features = net(image_data)
        image_features = image_features.cpu().data.numpy()
        for i in range(image_data.size(0)):
            scene = db.scenedb[image_inds[i]]
            image_index = scene['image_index']
            image_feature_path = db.image_path_from_index(image_index, 'image_resnet152', 'pkl')
            image_feature_dir = osp.dirname(image_feature_path)
            maybe_create(image_feature_dir)
            features = image_features[i].flatten()
            with open(image_feature_path, 'wb') as fid:
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

    batch_dump_image_features(config, 'train', '2017')
    batch_dump_image_features(config, 'val',   '2017')
    batch_dump_image_features(config, 'test',  '2017')
    batch_dump_image_features(config, 'aux',   '2017')
