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
from tsne_wrapper import *

from datasets.coco import coco
import torch, torchtext
from torch.utils.data import Dataset
from pycocotools import mask as COCOmask


def test_feature_tsne(config):
    db = coco(config, 'train')
    output_dir = osp.join(config.model_dir, 'test_feature_tsne')
    maybe_create(output_dir)

    index_to_class = db.classes
    class_to_index = db.class_to_ind
    # print(class_to_index)
    
    indices = range(len(index_to_class))
    patches_per_class = dict(zip(indices, [[] for i in indices]))

    for i in range(len(db.patchdb)):
        x = db.patchdb[i]
        category_id = x['class']
        patches_per_class[category_id].append(x)
    
    for key, value in patches_per_class.items():
        if key < 3:
            continue
        inds = np.random.permutation(range(len(value)))
        inds = inds[:200]
        value = [value[i] for i in inds]
        patches_per_class[key] = value 

    pad_value = np.array([[[103.939, 116.779, 123.68]]])
    all_features, all_colors, all_patches = [], [], []
    for key, value in patches_per_class.items():
        if key == 3 or key == 4:
            for i in range(len(value)):
                x = value[i]
                category_id = x['class']
                color = db.colormap[category_id]
                with open(x['features_path'], 'rb') as fid:
                    features = pickle.load(fid)

                xywh = x['box']
                image = cv2.imread(x['image_path'], cv2.IMREAD_COLOR)
                output_image = crop_and_resize(image, xywh, [64, 64], [64, 64], pad_value)
                cv2.rectangle(output_image, (0, 0), (63, 63), color*255, 5)

                all_features.append(features)
                all_colors.append(color)
                all_patches.append(output_image)

    all_features = np.array(all_features)
    all_colors = clamp_array(np.array(all_colors) * 255, 0, 255)

    X_tsne = tsne_embedding(all_features)
    # embedding_image = tsne_colors(all_colors, X_tsne, [5000, 5000])
    embedding_image = tsne_images(all_patches, X_tsne, [5000, 5000], [64, 64])
    cv2.imwrite(osp.join(output_dir, 'tsne.png'), embedding_image)
    




if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_feature_tsne(config)