#!/usr/bin/env python

import os, sys, cv2
import json, scipy
import math, PIL, cairo
import numpy as np
import scipy.io
import pickle, random
import os.path as osp
from time import time
from config import get_config
from copy import deepcopy
from glob import glob
from utils import *

import torch, torchtext
from torch.utils.data import Dataset


class cityscapes_tranformation(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))

    def __call__(self, sample):
        gt_image = sample['gt_image']
        proposal_image = sample['proposal_image']
        sample['gt_image'] = normalize(gt_image, self.mean, self.std)
        sample['proposal_image'] = normalize(proposal_image, self.mean, self.std)
        return sample


class cityscapes_syn(Dataset):
    def __init__(self, config, split, transform=None):
        self.cfg = config
        self.split = split
        self.name = 'cityscapes_syn' + '_' + split
        self.transform = transform

        # cache dir
        self.root_dir  = osp.join(config.data_dir, 'cityscapes_syn')
        self.cache_dir = osp.abspath(osp.join(config.data_dir, 'caches'))

        if split == 'train':
            self.image_index = [i for i in range(1, 2876)]
        else:
            self.image_index = [i for i in range(2876, 2976)]

        # color palette
        info_path = osp.join(self.cache_dir, 'cityscapes.json')
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.palette = np.array(info['palette'], dtype=np.uint8)

    def num_classes(self):
        return self.palette.shape[0]

    def get_semantic_map(self, path):
        semantic = scipy.misc.imread(path)
        return self.encode_semantic_map(semantic)

    def encode_semantic_map(self, semantic):
        n = self.num_classes()
        h = semantic.shape[0]
        w = semantic.shape[1]

        out = np.zeros((n, h, w), dtype=np.float32)
        for k in range(n):
            eqR = semantic[:,:,0] == self.palette[k,0]
            eqG = semantic[:,:,1] == self.palette[k,1]
            eqB = semantic[:,:,2] == self.palette[k,2]
            out[k,:,:] = np.float32((eqR)&(eqG)&(eqB))
        return out

    def decode_semantic_map(self, semantic):
        prediction = np.argmax(semantic[:-1], 0)
        h = prediction.shape[0]
        w = prediction.shape[1]
        color_image = self.palette[prediction.ravel()].reshape((h,w,3))
        row, col = np.where(np.sum(semantic[:-1], 0)==0)
        color_image[row, col, :] = 0
        return color_image

    def print_semantic_map(semantic,path):
        semantic = semantic.transpose([2,3,1,0])
        prediction = np.argmax(semantic, axis=2)
        h = prediction.shape[0]
        w = prediction.shape[1]
        color_image = self.palette[prediction.ravel()].reshape((h,w,3))
        row, col, dump = np.where(np.sum(semantic, axis=2)==0)
        color_image[row, col, :]=0
        scipy.misc.imsave(path, color_image)

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, idx):
        entry = {}

        image_ind = self.image_index[idx]
        entry['image_index'] = image_ind

        if self.split == 'train':
            group_id = np.random.randint(1, 6)
        else:
            group_id = 1

        synthesis_dir = osp.join(self.root_dir, 'synthesis', 'traindata_synthesis_512_1024')
        proposal_image_dir = osp.join(synthesis_dir, 'traindata_mat', '%02d'%group_id)
        proposal_label_dir = osp.join(synthesis_dir, 'traindata_label', '%02d'%group_id)
        gt_image_dir = osp.join(self.root_dir, 'RGB512Full')
        gt_label_dir = osp.join(self.root_dir, 'Label512Full')

        # proposal image
        proposal_image_path = osp.join(proposal_image_dir, "%08d.mat" %image_ind)
        dic = scipy.io.loadmat(proposal_image_path)
        proposal_image = dic['proposal'][:,:,::-1].copy()
        proposal_image = proposal_image.astype(np.float32)

        # proposal mask
        proposal_mask = np.sum(proposal_image, axis=-1)
        proposal_mask[np.where(proposal_mask > 0)] = 1
        proposal_mask = proposal_mask[None, ...]

        # proposal label
        proposal_label_path = osp.join(proposal_label_dir, "%08d.png" %image_ind)
        proposal_label = self.get_semantic_map(proposal_label_path)
        proposal_label = np.concatenate((proposal_label, 1 - np.sum(proposal_label, axis=0, keepdims=True)), axis=0)

        # gt image
        gt_image_path = osp.join(gt_image_dir, "%08d.png" %image_ind)
        gt_image = cv2.imread(gt_image_path, cv2.IMREAD_COLOR).astype(np.float32)

        # gt label
        gt_label_path = osp.join(gt_label_dir, "%08d.png" %image_ind)
        gt_label = self.get_semantic_map(gt_label_path)
        gt_label = np.concatenate((gt_label, 1 - np.sum(gt_label, axis=0, keepdims=True)), axis=0)

        entry['proposal_image'] = proposal_image
        entry['proposal_label'] = proposal_label
        entry['proposal_mask'] = proposal_mask
        entry['gt_image'] = gt_image
        entry['gt_label'] = gt_label

        if self.transform:
            entry = self.transform(entry)

        return entry
