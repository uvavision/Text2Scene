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

from nntable import AllCategoriesTables


def find_closest_patch(srcdb, dstdb, query_image_index, query_instance_ind, candidate_patches):
	query_mask_path = srcdb.patch_path_from_indices(query_image_index, query_instance_ind, 'patch_mask', 'png', None)
	query_mask = cv2.imread(query_mask_path, cv2.IMREAD_GRAYSCALE)
	query_mask = query_mask.astype(np.float32)/255.0

	ious = []
	for i in range(len(candidate_patches)):
		candidate_patch = candidate_patches[i]
		image_index = candidate_patch['image_index']
		instance_ind = candidate_patch['instance_ind']
		mask_path = dstdb.patch_path_from_indices(image_index, instance_ind, 'patch_mask', 'png', None)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		mask = mask.astype(np.float32)/255
		iou = np.sum(np.minimum(query_mask, mask))/(np.sum(np.maximum(query_mask, mask)) + 1e-12)
		ious.append(iou)
	ious = np.array(ious)
	ind = np.argmax(ious)
	return candidate_patches[ind]


def generate_simulated_scenes(config, split, year):
	db = coco(config, split, year)
	data_dir = osp.join(config.data_dir, 'coco')
	if (split == 'test') or (split == 'aux'):
		images_dir = osp.join(data_dir, 'crn_images', 'train'+year)
		noices_dir = osp.join(data_dir, 'crn_noices', 'train'+year)
		labels_dir = osp.join(data_dir, 'crn_labels', 'train'+year)
		masks_dir  = osp.join(data_dir, 'crn_masks', 'train'+year)
	else:
		images_dir = osp.join(data_dir, 'crn_images', split+year)
		noices_dir = osp.join(data_dir, 'crn_noices', split+year)
		labels_dir = osp.join(data_dir, 'crn_labels', split+year)
		masks_dir  = osp.join(data_dir, 'crn_masks',  split+year)
	maybe_create(images_dir)
	maybe_create(noices_dir)
	maybe_create(labels_dir)
	maybe_create(masks_dir)

	traindb = coco(config, 'train', '2017')
	nn_tables = AllCategoriesTables(traindb)
	nn_tables.build_nntables_for_all_categories(True)

	# start_ind = 0
	# end_ind = len(db.scenedb)
	start_ind = 25000 + 14000 * config.seed
	end_ind = 25000 + 14000 * (config.seed + 1)
	patches_per_class = traindb.patches_per_class
	color_transfer_threshold = 0.8

	for i in range(start_ind, end_ind):
		entry = db.scenedb[i]
		width = entry['width']
		height = entry['height']
		xywhs = entry['boxes']
		masks = entry['masks']
		clses = entry['clses']
		image_index = entry['image_index']
		instance_inds = entry['instance_inds']

		full_mask  = np.zeros((height, width), dtype=np.float32)
		full_label = np.zeros((height, width), dtype=np.float32)
		full_image = np.zeros((height, width, 3), dtype=np.float32)
		full_noice = np.zeros((height, width, 3), dtype=np.float32)

		original_image = cv2.imread(db.color_path_from_index(image_index), cv2.IMREAD_COLOR)

		for j in range(len(masks)):
			src_img = original_image.astype(np.float32).copy()
			xywh = xywhs[j]
			mask = masks[j]
			cls_idx = clses[j]
			instance_ind = instance_inds[j]
			embed_path = db.patch_path_from_indices(image_index, instance_ind, 'patch_feature', 'pkl', config.use_patch_background)
			with open(embed_path, 'rb') as fid:
				query_vector = pickle.load(fid)
			n_samples = min(100, len(patches_per_class[cls_idx]))#min(config.n_nntable_trees, len(patches_per_class[cls_idx]))
			candidate_patches = nn_tables.retrieve(cls_idx, query_vector, n_samples)
			candidate_patches = [x for x in candidate_patches if x['instance_ind'] != instance_ind]
			assert(len(candidate_patches) > 1)

			# candidate_instance_ind = instance_ind
			# candidate_patch = None
			# while (candidate_instance_ind == instance_ind):
			# 	cid = np.random.randint(0, len(candidate_patches))
			# 	candidate_patch = candidate_patches[cid]
			# 	candidate_instance_ind = candidate_patch['instance_ind']
			candidate_patch = find_closest_patch(db, traindb, image_index, instance_ind, candidate_patches)

			# stenciling
			src_mask = COCOmask.decode(mask)
			dst_mask = COCOmask.decode(candidate_patch['mask'])
			src_xyxy = xywh_to_xyxy(xywh, width, height)
			dst_xyxy = xywh_to_xyxy(candidate_patch['box'], candidate_patch['width'], candidate_patch['height'])
			dst_mask = dst_mask[dst_xyxy[1]:(dst_xyxy[3]+1), dst_xyxy[0]:(dst_xyxy[2]+1)]
			dst_mask = cv2.resize(dst_mask, 
				(src_xyxy[2] - src_xyxy[0] + 1, src_xyxy[3] - src_xyxy[1] + 1), 
				interpolation = cv2.INTER_NEAREST)
			src_mask[src_xyxy[1]:(src_xyxy[3]+1), src_xyxy[0]:(src_xyxy[2]+1)] = \
				np.minimum(dst_mask, src_mask[src_xyxy[1]:(src_xyxy[3]+1), src_xyxy[0]:(src_xyxy[2]+1)])
			# color transfer
			if random.random() > color_transfer_threshold:
				candidate_index = candidate_patch['image_index']
				candidate_image = cv2.imread(traindb.color_path_from_index(candidate_index), cv2.IMREAD_COLOR).astype(np.float32)
				candidate_cropped = candidate_image[dst_xyxy[1]:(dst_xyxy[3]+1), dst_xyxy[0]:(dst_xyxy[2]+1)]
				candidate_cropped = cv2.resize(candidate_cropped, (src_xyxy[2] - src_xyxy[0] + 1, src_xyxy[3] - src_xyxy[1] + 1), interpolation = cv2.INTER_CUBIC)
				original_cropped = src_img[src_xyxy[1]:(src_xyxy[3]+1), src_xyxy[0]:(src_xyxy[2]+1)]
				transfer_cropped = Monge_Kantorovitch_color_transfer(original_cropped, candidate_cropped)
				src_img[src_xyxy[1]:(src_xyxy[3]+1), src_xyxy[0]:(src_xyxy[2]+1)] = transfer_cropped

			# im1 = cv2.resize(full_image, (128, 128))
			# im2 = cv2.resize(src_img[src_xyxy[1]:(src_xyxy[3]+1), src_xyxy[0]:(src_xyxy[2]+1), :], (128, 128))
			# # im2 = cv2.resize(np.repeat(255*src_mask[...,None], 3, -1), (128, 128))
			# im3 = cv2.resize(candidate_image, (128, 128))
			# im4 = cv2.resize(candidate_cropped, (128, 128))
			# im = np.concatenate((im1, im2, im3, im4), 1)
			# cv2.imwrite("%03d_%03d.png"%(i, j), im)

			full_image = compose(full_image, src_img, src_mask)

			# boundary elision
			radius = int(0.05 * min(width, height))
			if np.amin(src_mask) > 0:
				src_mask[0, :] = 0
				src_mask[-1,:] = 0
				src_mask[:, 0] = 0
				src_mask[:, -1] = 0
			sobelx = cv2.Sobel(src_mask, cv2.CV_64F, 1, 0, ksize=3)
			sobely = cv2.Sobel(src_mask, cv2.CV_64F, 0, 1, ksize=3)
			sobel = np.abs(sobelx) + np.abs(sobely)
			edge = np.zeros_like(sobel)
			edge[sobel>0.9] = 1.0
			morp_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius, radius))
			edge = cv2.dilate(edge, morp_kernel, iterations = 1)
			row, col = np.where(edge > 0)
			n_edge_pixels = len(row)
			pixel_indices = np.random.permutation(range(n_edge_pixels))
			pixel_indices = pixel_indices[:(n_edge_pixels//2)]
			row = row[pixel_indices]
			col = col[pixel_indices]
			src_img[row, col, :] = 255


			full_mask = np.maximum(full_mask, src_mask)
			full_label[src_mask>0] = cls_idx
			full_noice = compose(full_noice, src_img, src_mask)

			# im1 = cv2.resize(full_image, (128, 128))
			# im2 = cv2.resize(src_img[src_xyxy[1]:(src_xyxy[3]+1), src_xyxy[0]:(src_xyxy[2]+1), :], (128, 128))
			# im3 = cv2.resize(candidate_image, (128, 128))
			# im4 = cv2.resize(candidate_cropped, (128, 128))
			# im = np.concatenate((im1, im2, im3, im4), 1)
			# cv2.imwrite("%03d_%03d.png"%(i, j), im)

		output_name = str(image_index).zfill(12)
		output_path = osp.join(images_dir, output_name+'.jpg')
		cv2.imwrite(output_path, clamp_array(full_image, 0, 255).astype(np.uint8))
		output_path = osp.join(noices_dir, output_name+'.jpg')
		cv2.imwrite(output_path, clamp_array(full_noice, 0, 255).astype(np.uint8))
		output_path = osp.join(masks_dir, output_name+'.png')
		cv2.imwrite(output_path, clamp_array(255*full_mask, 0, 255).astype(np.uint8))
		output_path = osp.join(labels_dir, output_name+'.png')
		cv2.imwrite(output_path, full_label.astype(np.uint8))
		print(i, image_index)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)

    generate_simulated_scenes(config, 'train', '2017')
    # generate_simulated_scenes(config, 'val',   '2017')
    # generate_simulated_scenes(config, 'test',  '2017')
    # generate_simulated_scenes(config, 'aux',   '2017')

    # # src_img = cv2.imread('scotland_house.png').astype(np.float32)/255
    # # dst_img = cv2.imread('scotland_plain.png').astype(np.float32)/255

    # # out_img = Monge_Kantorovitch_color_transfer(src_img, dst_img)
    # # out_img = clamp_array(out_img*255, 0, 255).astype(np.uint8)
    # # cv2.imwrite('scotland_transferred.png', out_img)

    # src_img = cv2.imread('scotland_house.png').astype(np.float32)
    # dst_img = cv2.imread('scotland_plain.png').astype(np.float32)

    # out_img = Monge_Kantorovitch_color_transfer(src_img, dst_img)
    # # out_img = clamp_array(out_img*255, 0, 255).astype(np.uint8)
    # cv2.imwrite('scotland_transferred.png', out_img)

