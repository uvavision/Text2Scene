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


def dump_patch_labels(config, split, year):
    db = coco(config, split, year)
    data_dir = osp.join(config.data_dir, 'coco')
    if (split == 'test') or (split == 'aux'):
        main_dir = osp.join(data_dir, 'patch_label', 'train'+year)
    else:
        main_dir = osp.join(data_dir, 'patch_label', split+year)
    maybe_create(main_dir)
    # small_pad_value = np.array([0, 144, 255]).reshape((1,1,3))

    start_ind = 0
    end_ind = len(db.scenedb)
    for i in range(start_ind, end_ind):
        entry = db.scenedb[i]
        image_index = entry['image_index']
        output_dir = osp.join(main_dir, str(image_index).zfill(12))
        maybe_create(output_dir)

        semantic_path = db.image_path_from_index(image_index, 'image_label', 'png')
        semantic = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
        boxes = entry['boxes']
        instance_indices = entry['instance_inds']

        for j in range(len(boxes)):
            xywh = boxes[j]
            # xyxy = xywh_to_xyxy(xywh, entry['width'], entry['height'])
            # output_image = semantic[xyxy[1]:(xyxy[3]+1), xyxy[0]:(xyxy[2]+1), :]
            output_image = crop_with_padding(semantic[..., None], xywh, [64, 64], [32, 32], np.array([0])).squeeze()
            output_image = cv2.resize(output_image, (64, 64), interpolation = cv2.INTER_NEAREST)
            # output_label = db.encode_semantic_map(output_image).astype(np.uint8)
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

    dump_patch_labels(config, 'train', '2017')
    dump_patch_labels(config, 'val',   '2017')
    dump_patch_labels(config, 'test',  '2017')
    dump_patch_labels(config, 'aux',   '2017')
