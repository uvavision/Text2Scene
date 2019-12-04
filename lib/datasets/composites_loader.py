#!/usr/bin/env python

import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
from collections import OrderedDict
from scipy import ndimage, misc
from skimage.transform import resize as array_resize
from nms.cpu_nms import cpu_nms
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
import torch, torchtext
from torch.utils.data import Dataset

from composites_config import get_config
from composites_utils import *


class sequence_loader(Dataset):
    def __init__(self, coco_imdb, nn_tables=None):
        self.cfg = coco_imdb.cfg
        self.db = coco_imdb
        self.nn_tables = nn_tables

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, idx):
        # start = time()
        entry = {}
        scene = self.db.scenedb[idx].copy()
        image_index = scene['image_index']
        entry['scene_index'] = int(idx)
        entry['image_index'] = int(scene['image_index'])
        entry['width'] = scene['width']
        entry['height'] = scene['height']

        ###################################################################
        ## Patch indices
        ###################################################################
        instance_inds = scene['instance_inds']
        patch_inds = []
        for i in range(len(instance_inds)):
            instance_ind = instance_inds[i]
            name = str(image_index).zfill(12) + '_' + str(instance_ind).zfill(12)
            patch_ind = self.db.name_to_patch_index[name]
            patch_inds.append(patch_ind)
        patch_inds, _ = pad_sequence(patch_inds, self.cfg.max_output_length, -1, None, -1, 0.0)
        entry['patch_inds'] = patch_inds

        ###################################################################
        ## Sentence
        ###################################################################
        group_sents = scene['captions']
        if self.db.split == 'train' and self.cfg.sent_group < 0:
            cid = np.random.randint(0, len(group_sents))
        else:
            cid = self.cfg.sent_group
        sentence = group_sents[cid]
        entry['sentence'] = sentence
        entry['caption_index'] = scene['caption_inds'][cid]

        ###################################################################
        ## Input tokens
        ###################################################################
        # Word indices
        tokens = [w for w in word_tokenize(sentence.lower())]
        tokens = further_token_process(tokens)
        word_inds = [self.db.lang_vocab.word_to_index(w) for w in tokens]
        word_inds = [wi for wi in word_inds if wi > self.cfg.EOS_idx]
        word_inds, word_msks = pad_sequence(word_inds, self.cfg.max_input_length, self.cfg.PAD_idx, None, self.cfg.EOS_idx, 1.0)
        entry['word_inds'] = word_inds.astype(np.int32)
        entry['word_lens'] = np.sum(word_msks).astype(np.int32)

        # Output inds, vecs, msks
        outputs = self.db.scene_to_prediction_outputs(scene)
        entry['out_inds'] = outputs['out_inds']
        entry['out_msks'] = outputs['out_msks']
        ###########################
        # out_vecs is deprecated
        entry['out_vecs'] = outputs['out_vecs']

        fg_inds = entry['out_inds'][:, 0].copy().astype(np.int32).flatten().tolist()
        fg_inds = [self.cfg.SOS_idx] + fg_inds
        entry['fg_inds'] = np.array(fg_inds)

        ###################################################################
        ## Images and Layouts
        ###################################################################
        entry['image_path'] = self.db.color_path_from_index(image_index)
        # print("other_time", time() - start)
        # start = time()
        background_images, foreground_images, negative_images,\
        foreground_resnets, negative_resnets = \
            self.db.render_volumes_for_a_scene(scene, return_sequence=True, nn_tables=self.nn_tables)
        pad_image = np.zeros_like(background_images[-1])
        entry['background'], _ = \
            pad_sequence(background_images,
                self.cfg.max_output_length,
                pad_image, pad_image, None, 0.0)
        pad_fg_vol = np.zeros_like(foreground_images[-1])
        entry['foreground'], _ = pad_sequence(foreground_images,
            self.cfg.max_output_length,
            pad_fg_vol, None, pad_fg_vol, 0.0)
        entry['negative'], _ = pad_sequence(negative_images,
            self.cfg.max_output_length,
            pad_fg_vol, None, pad_fg_vol, 0.0)
        pad_resnets = np.zeros_like(foreground_resnets[-1])
        entry['foreground_resnets'], _ = pad_sequence(foreground_resnets,
            self.cfg.max_output_length,
            pad_resnets, None, pad_resnets, 0.0)
        entry['negative_resnets'], _ = pad_sequence(negative_resnets,
            self.cfg.max_output_length,
            pad_resnets, None, pad_resnets, 0.0)
        # print("render_time", time() - start)
        # print('-------------------')

        return entry


class patch_vol_loader(Dataset):
    def __init__(self, coco_imdb):
        self.cfg = coco_imdb.cfg
        self.db = coco_imdb

    def __len__(self):
        return len(self.db.patchdb)

    def __getitem__(self, idx):
        entry = {}
        patch = self.db.patchdb[idx]
        image_index = patch['image_index']
        instance_ind = patch['instance_ind']
        entry['patch_vol'] = self.db.get_volume_from_indices(image_index, instance_ind)
        entry['patch_ind'] = idx
        if self.cfg.use_global_resnet:
            resnet_path = self.db.image_path_from_index(image_index, 'image_resnet152', 'pkl')
        else:
            resnet_path = self.db.patch_path_from_indices(image_index, instance_ind, 'patch_resnet152', 'pkl', True)
        with open(resnet_path, 'rb') as fid:
            resnet_features = pickle.load(fid)
        entry['patch_resnet'] = resnet_features
        return entry


class patch_loader(Dataset):
    def __init__(self, coco_imdb, field, ext, use_background, use_normalization):
        self.cfg = coco_imdb.cfg
        self.db = coco_imdb
        self.field = field
        self.ext = ext
        self.use_background = use_background
        self.use_normalization = use_normalization

    def __len__(self):
        return len(self.db.patchdb)

    def __getitem__(self, idx):
        entry = {}
        patch = self.db.patchdb[idx]
        image_index = patch['image_index']
        instance_ind = patch['instance_ind']
        patch_path = self.db.patch_path_from_indices(image_index, instance_ind, self.field, self.ext, self.use_background)
        patch_data = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
        if self.use_normalization:
            patch_data = normalize(patch_data)
        entry['patch_data'] = patch_data
        entry['patch_ind'] = idx
        return entry


class image_loader(Dataset):
    def __init__(self, coco_imdb, field, ext, use_normalization):
        self.cfg = coco_imdb.cfg
        self.db = coco_imdb
        self.field = field
        self.ext = ext
        self.use_normalization = use_normalization

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, idx):
        entry = {}
        scene = self.db.scenedb[idx]
        image_index = scene['image_index']
        image_path = self.db.image_path_from_index(image_index, self.field, self.ext)
        image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_data = cv2.resize(image_data, (224, 224))
        if self.use_normalization:
            image_data = normalize(image_data)
        entry['image_data'] = image_data
        entry['image_ind'] = idx
        return entry


class synthesis_loader(Dataset):
    def __init__(self, coco_imdb):
        self.cfg = coco_imdb.cfg
        self.db  = coco_imdb

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, idx):
        entry = {}
        scene = self.db.scenedb[idx]
        image_index = scene['image_index']
        h, w = self.cfg.output_image_size

        out_color_path = self.db.color_path_from_index(image_index)
        out_label_path = self.db.image_path_from_index(image_index, 'image_label', 'png')

        in_proposal_path = self.db.image_path_from_index(image_index, 'simulated_images', 'jpg')
        in_label_path = self.db.image_path_from_index(image_index, 'simulated_labels', 'png')
        in_mask_path = self.db.image_path_from_index(image_index, 'simulated_masks', 'png')

        in_proposal = cv2.imread(in_proposal_path, cv2.IMREAD_COLOR)
        in_mask     = cv2.imread(in_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        out_color   = cv2.imread(out_color_path, cv2.IMREAD_COLOR)

        in_label    = cv2.imread(in_label_path, cv2.IMREAD_GRAYSCALE)
        out_label   = cv2.imread(out_label_path, cv2.IMREAD_GRAYSCALE)
        if self.cfg.use_super_category:
            in_label = map(lambda x: self.db.class_ind_to_super_class_ind[x], in_label.flatten().tolist())
            in_label = np.array(list(in_label)).reshape((in_mask.shape[0], in_mask.shape[1]))
            out_label = map(lambda x: self.db.class_ind_to_super_class_ind[x], out_label.flatten().tolist())
            out_label = np.array(list(out_label)).reshape((in_mask.shape[0], in_mask.shape[1]))


        in_mask = 255.0 - in_mask

        # in_proposal, _, _ = create_squared_image(in_proposal, np.array([0, 0, 0]))
        # out_color, _, _   = create_squared_image(out_color, np.array([0, 0, 0]))
        # in_mask, _, _     = create_squared_image(in_mask[..., None], np.array([0]))
        # in_label, _, _    = create_squared_image(in_label[..., None], np.array([0]))
        # out_label, _, _   = create_squared_image(out_label[..., None], np.array([0]))

        in_proposal = cv2.resize(in_proposal, (h, w), interpolation = cv2.INTER_CUBIC)
        out_color = cv2.resize(out_color, (h, w), interpolation = cv2.INTER_CUBIC)
        in_mask = cv2.resize(in_mask.squeeze(), (h, w), interpolation = cv2.INTER_NEAREST)
        in_label = cv2.resize(in_label.squeeze(), (h, w), interpolation = cv2.INTER_NEAREST)
        out_label = cv2.resize(out_label.squeeze(), (h, w), interpolation = cv2.INTER_NEAREST)

        in_vol = np.concatenate((in_label[..., None], in_mask[..., None], in_proposal[:,:,::-1].copy()), -1)

        entry['input_vol'] = in_vol
        entry['gt_image'] = out_color[:,:,::-1].copy()
        entry['gt_label'] = out_label
        entry['image_index'] = image_index

        return entry


class proposal_loader(Dataset):
    def __init__(self, coco_imdb):
        self.cfg = coco_imdb.cfg
        self.db  = coco_imdb

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, idx):
        entry = {}
        scene = self.db.scenedb[idx]
        image_index = scene['image_index']
        out_h, out_w = self.cfg.output_image_size

        in_proposal_path = self.db.image_path_from_index(image_index, 'proposal_images', 'jpg')
        in_label_path = self.db.image_path_from_index(image_index, 'proposal_labels', 'png')
        in_mask_path = self.db.image_path_from_index(image_index, 'proposal_masks', 'png')

        # print('in_proposal_path', in_proposal_path)
        # print('in_mask_path', in_mask_path)
        # print('in_label_path', in_label_path)

        in_proposal = cv2.imread(in_proposal_path, cv2.IMREAD_COLOR)
        in_mask     = cv2.imread(in_mask_path, cv2.IMREAD_GRAYSCALE)
        in_label    = cv2.imread(in_label_path, cv2.IMREAD_GRAYSCALE)

        # bounding box of the mask
        nonzero_pixels = cv2.findNonZero(in_mask)
        x,y,w,h = cv2.boundingRect(nonzero_pixels)
        xyxy = np.array([int(x), int(y), int(x+w), int(y+h)])

        in_mask  = in_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        in_label = in_label[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] 
        in_proposal = in_proposal[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] 

        in_proposal = cv2.resize(in_proposal, (out_h, out_w), interpolation = cv2.INTER_CUBIC)
        in_mask = cv2.resize(in_mask, (out_h, out_w), interpolation = cv2.INTER_NEAREST)
        in_label = cv2.resize(in_label, (out_h, out_w), interpolation = cv2.INTER_NEAREST)

        in_mask = in_mask.astype(np.float32)
        in_mask = 255.0 - in_mask
        in_vol = np.concatenate((in_label[..., None], in_mask[..., None], in_proposal[:,:,::-1].copy()), -1)
        # print(in_vol.shape)

        entry['input_vol'] = in_vol
        entry['image_index'] = image_index
        entry['box'] = xyxy
        return entry

