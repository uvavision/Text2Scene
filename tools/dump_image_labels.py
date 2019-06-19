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


def dump_image_labels(config, split, year):
    db = coco(config, split, year)
    data_dir = osp.join(config.data_dir, 'coco')
    if (split == 'test') or (split == 'aux'):
        output_dir = osp.join(data_dir, 'image_label', 'train'+year)
    else:
        output_dir = osp.join(data_dir, 'image_label', split+year)
    maybe_create(output_dir)
    start_ind = 0
    end_ind = len(db.scenedb)
    for i in range(start_ind, end_ind):
        entry = db.scenedb[i]
        image_index = entry['image_index']
        output_name = str(image_index).zfill(12)
        output_path = osp.join(output_dir, output_name+'.png')
        if osp.exists(output_path):
            continue
        label = db.render_label(entry, False)
        cv2.imwrite(output_path, label)
        print(i, image_index)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    dump_image_labels(config, 'train', '2017')
    dump_image_labels(config, 'val',   '2017')
    dump_image_labels(config, 'test',  '2017')
    dump_image_labels(config, 'aux',   '2017')
