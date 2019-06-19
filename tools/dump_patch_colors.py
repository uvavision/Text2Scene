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


def dump_patch_colors(config, split, year):
    db = coco(config, split, year)
    data_dir = osp.join(config.data_dir, 'coco')

    if config.use_patch_background:
        if (split == 'test') or (split == 'aux'):
            small_dir = osp.join(data_dir, 'patch_color_with_bg', 'train'+year)
        else:
            small_dir = osp.join(data_dir, 'patch_color_with_bg', split+year)
    else:
        if (split == 'test') or (split == 'aux'):
            small_dir = osp.join(data_dir, 'patch_color_without_bg', 'train'+year)
        else:
            small_dir = osp.join(data_dir, 'patch_color_without_bg', split+year)
    maybe_create(small_dir)

    if config.use_patch_background:
        if (split == 'test') or (split == 'aux'):
            large_dir = osp.join(data_dir, 'patch_color_large_with_bg', 'train'+year)
        else:
            large_dir = osp.join(data_dir, 'patch_color_large_with_bg', split+year)
    else:
        if (split == 'test') or (split == 'aux'):
            large_dir = osp.join(data_dir, 'patch_color_large_without_bg', 'train'+year)
        else:
            large_dir = osp.join(data_dir, 'patch_color_large_without_bg', split+year)
    maybe_create(large_dir)

    indices = range(len(db.scenedb))
    large_pad_value = np.array([103.53, 116.28, 123.675]).reshape((1,1,3))
    small_pad_value = np.array([0, 0, 0]).reshape((1,1,3))
    start_ind = 0
    end_ind = len(db.scenedb)

    for i in indices[start_ind:end_ind]:
        entry = db.scenedb[i]
        image_index = entry['image_index']
        small_output_dir = osp.join(small_dir, str(image_index).zfill(12))
        maybe_create(small_output_dir)
        large_output_dir = osp.join(large_dir, str(image_index).zfill(12))
        maybe_create(large_output_dir)

        image = cv2.imread(db.color_path_from_index(image_index), cv2.IMREAD_COLOR)
        boxes = entry['boxes']
        masks = entry['masks']
        instance_indices = entry['instance_inds']

        for j in range(len(boxes)):
            xywh = boxes[j]
            if config.use_patch_background:
                composed_image = image.copy()
            else:
                rle_rep = masks[j]
                mask = COCOmask.decode(rle_rep)
                background_image = np.ones_like(image) * small_pad_value.reshape(1,1,3)
                composed_image = compose(background_image, image, mask)

            # xyxy = xywh_to_xyxy(xywh, entry['width'], entry['height'])
            # output_image = composed_image[xyxy[1]:(xyxy[3]+1), xyxy[0]:(xyxy[2]+1), :]
            large_image = crop_with_padding(composed_image, xywh, [224, 224], [112, 112], large_pad_value)
            small_image = crop_with_padding(composed_image, xywh, [64, 64], [32, 32], small_pad_value)
            large_image = cv2.resize(large_image, (224, 224), interpolation = cv2.INTER_CUBIC)
            small_image = cv2.resize(small_image, (64, 64), interpolation = cv2.INTER_CUBIC)
            output_name = str(image_index).zfill(12) + '_' + str(instance_indices[j]).zfill(12)
            small_output_path = osp.join(small_output_dir, output_name+'.jpg')
            large_output_path = osp.join(large_output_dir, output_name+'.jpg')
            cv2.imwrite(small_output_path, small_image)
            cv2.imwrite(large_output_path, large_image)
            print(i, j)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    dump_patch_colors(config, 'train', '2017')
    dump_patch_colors(config, 'val',   '2017')
    dump_patch_colors(config, 'test',  '2017')
    dump_patch_colors(config, 'aux',   '2017')
