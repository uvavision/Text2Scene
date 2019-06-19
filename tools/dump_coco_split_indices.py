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

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

import torch, torchtext


this_dir = osp.dirname(__file__)


def get_ann_file(prefix, split, year):
	# Example annotation path for prefix=captions:
	#   annotations/captions_train2017.json
	root_dir = osp.join(this_dir, '..', 'data', 'coco')
	ann_path = osp.join(root_dir, 'annotations', prefix + '_' + split + year + '.json')
	assert osp.exists(ann_path), 'Path does not exist: {}'.format(ann_path)
	return ann_path


def dump_coco_indices(split, year):
	cocoAPI = COCO(get_ann_file('instances', split, year))
	image_indices = sorted(cocoAPI.getImgIds())
	output_path = osp.join(this_dir, '..', 'data', 'caches', 'coco_official_image_indices_%s_%s.txt'%(split, year))
	np.savetxt(output_path, image_indices, fmt='%d')



def create_my_splits():
	train2017_path = osp.join(this_dir, '..', 'data', 'caches', 'coco_official_image_indices_%s_%s.txt'%('train', '2017'))
	train2014_path = osp.join(this_dir, '..', 'data', 'caches', 'coco_official_image_indices_%s_%s.txt'%('train', '2014'))
	val2017_path = osp.join(this_dir, '..', 'data', 'caches', 'coco_official_image_indices_%s_%s.txt'%('val', '2017'))
	val2014_path = osp.join(this_dir, '..', 'data', 'caches', 'coco_official_image_indices_%s_%s.txt'%('val', '2014'))

	train2017_inds = np.loadtxt(train2017_path, dtype=np.int32)
	val2017_inds = np.loadtxt(val2017_path, dtype=np.int32)
	train2014_inds = np.loadtxt(train2014_path, dtype=np.int32)
	val2014_inds = np.loadtxt(val2014_path, dtype=np.int32)

	# print(len(train2017_inds))
	# print(len(val2017_inds))
	# print(len(train2014_inds))
	# print(len(val2014_inds))

	num_train2017 = len(train2017_inds)
	# train2017 - train2014
	indices2014, indices2017 = [], []
	for i in range(num_train2017):
		index = train2017_inds[i]
		if index in val2014_inds:
			indices2014.append(i)
		else:
			indices2017.append(i)

	aux_path = osp.join(this_dir, '..', 'data', 'caches', 'aux_split.txt')
	np.savetxt(aux_path, indices2014, fmt='%d')

	randomized_inds = np.random.permutation(indices2017)
	train_inds = randomized_inds[:-5000]
	test_inds  = randomized_inds[-5000:]
	train_inds = sorted(train_inds)
	test_inds  = sorted(test_inds)

	train_path = osp.join(this_dir, '..', 'data', 'caches', 'train_split.txt')
	np.savetxt(train_path, train_inds, fmt='%d')

	test_path = osp.join(this_dir, '..', 'data', 'caches', 'test_split.txt')
	np.savetxt(test_path, test_inds, fmt='%d')



if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    # prepare_directories(config)

    # dump_coco_indices('train', '2017')
    # dump_coco_indices('val', '2017')
    # dump_coco_indices('train', '2014')
    # dump_coco_indices('val', '2014')

    create_my_splits()