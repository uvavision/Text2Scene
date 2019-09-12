#!/usr/bin/env python

import _init_paths
import os, random
import numpy as np
import os.path as osp
from time import time
from composites_config import get_config
from pycocotools.coco import COCO


this_dir = osp.dirname(__file__)


def get_ann_file(prefix, split, year):
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

	print('#train17', len(train2017_inds))
	print('#val17', len(val2017_inds))
	print('#train14', len(train2014_inds))
	print('#val14', len(val2014_inds))

	# train2017 - val2014
	val17_14, train14 = [], []
	for i in range(len(train2017_inds)):
		index = train2017_inds[i]
		if index in val2014_inds:
			val17_14.append(index)
		else:
			train14.append(index)

	train_path = osp.join(this_dir, '..', 'data', 'caches', 'composites_train_split.txt')
	val_path   = osp.join(this_dir, '..', 'data', 'caches', 'composites_val_split.txt')
	test_path  = osp.join(this_dir, '..', 'data', 'caches', 'composites_test_split.txt')
	aux_path   = osp.join(this_dir, '..', 'data', 'caches', 'composites_aux_split.txt')

	randomized_inds = np.random.permutation(train14)
	train_inds = randomized_inds[:-5000]
	val_inds   = randomized_inds[-5000:]

	np.savetxt(train_path, sorted(train_inds), fmt='%d')
	np.savetxt(val_path,   sorted(val_inds), fmt='%d')
	np.savetxt(test_path,  sorted(val2017_inds), fmt='%d')
	np.savetxt(aux_path,   sorted(val17_14), fmt='%d')
	

if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    dump_coco_indices('train', '2017')
    dump_coco_indices('val', '2017')
    dump_coco_indices('train', '2014')
    dump_coco_indices('val', '2014')
    create_my_splits()