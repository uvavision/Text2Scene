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
import torch, torchtext
from torch.utils.data import Dataset


def test_coco_discretization(config):
    db = coco(config, 'train')
    output_dir = osp.join(config.model_dir, 'test_coco_discretization')
    maybe_create(output_dir)

    location_map = db.location_map
    scale_aratio_map = db.scale_aratio_map

    pred_inds, gt_inds, coords = [], [], []
    for i in range(100):
        x = np.random.random()
        y = np.random.random()
        coord = np.array([x, y])
        coords.append(coord)
        pred_idx = location_map.coord2index(coord)
        pred_inds.append(pred_idx)

        diffs = location_map.coords - coord.reshape((1,2))
        dists = diffs[:,0] * diffs[:,0] + diffs[:,1] * diffs[:,1]
        gt_idx = np.argmin(dists)
        gt_inds.append(gt_idx)
    assert pred_inds == gt_inds
    # batch process
    batched_inds = location_map.coords2indices(np.array(coords))
    assert pred_inds == batched_inds.tolist()

    
    whs = np.random.random((100, 2))
    inds_1  = scale_aratio_map.whs2indices(whs)
    new_whs = scale_aratio_map.indices2whs(inds_1)
    inds_2  = scale_aratio_map.whs2indices(new_whs)

    assert inds_1.tolist() == inds_2.tolist()

    # x = np.concatenate((whs, new_whs), axis=-1)
    # y = np.absolute(whs - new_whs)
    # z = np.concatenate((x, y), axis=-1)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_coco_discretization(config)
