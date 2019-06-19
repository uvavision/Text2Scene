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


def test_per_category_stat(config):
    db = coco(config, 'train', '2017')
    output_dir = osp.join(config.model_dir, 'test_per_category_stat')
    maybe_create(output_dir)

    index_to_class = db.classes
    class_to_index = db.class_to_ind

    indices = range(len(index_to_class))
    patches_per_class = dict(zip(indices, [[] for i in indices]))

    num_object = 0
    num_stuff = 0
    for i in range(len(db.patchdb)):
        x = db.patchdb[i]
        patches_per_class[x['class']].append(x['instance_ind'])
        if x['class'] < 83:
            num_object += 1
        else:
            num_stuff += 1

    class_to_freq = {}
    for key, value in patches_per_class.items():
        class_to_freq[len(value)] = index_to_class[key]
        # class_to_freq[key] = [index_to_class[key], str(len(value))]

    with open(osp.join(output_dir, 'class_to_freq_train.json'), 'w') as fp:
        json.dump(class_to_freq, fp, indent=4, sort_keys=True)

    print('num_object', num_object)
    print('num_stuff', num_stuff)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_per_category_stat(config)
