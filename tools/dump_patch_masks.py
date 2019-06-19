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
from pycocotools import mask as COCOmask


def dump_patch_masks(config, split, year):
    db = coco(config, split, year)
    data_dir = osp.join(config.data_dir, 'coco')

    if (split == 'test') or (split == 'aux'):
        main_dir = osp.join(data_dir, 'patch_mask', 'train'+year)
    else:
        main_dir = osp.join(data_dir, 'patch_mask', split+year)
    maybe_create(main_dir)


    indices = range(len(db.scenedb))
    start_ind = 0
    end_ind = len(db.scenedb)

    for i in indices[start_ind:end_ind]:
        entry = db.scenedb[i]
        image_index = entry['image_index']
        output_dir = osp.join(main_dir, str(image_index).zfill(12))
        maybe_create(output_dir)
        
        boxes = entry['boxes']
        masks = entry['masks']
        instance_indices = entry['instance_inds']

        for j in range(len(boxes)):
            xywh = boxes[j]
            rle_rep = masks[j]
            mask = COCOmask.decode(rle_rep)
            mask = clamp_array(mask*255, 0, 255)
            output_image = crop_with_padding(mask[..., None], xywh, [64, 64], [32, 32], np.array([0])).squeeze()
            output_image = cv2.resize(output_image, (64, 64), interpolation = cv2.INTER_NEAREST)
            output_name = str(image_index).zfill(12) + '_' + str(instance_indices[j]).zfill(12)
            output_path = osp.join(output_dir, output_name+'.png')
            cv2.imwrite(output_path, output_image)
            
            print(i, j)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    dump_patch_masks(config, 'train', '2017')
    dump_patch_masks(config, 'val',   '2017')
    dump_patch_masks(config, 'test',  '2017')
    dump_patch_masks(config, 'aux',   '2017')
