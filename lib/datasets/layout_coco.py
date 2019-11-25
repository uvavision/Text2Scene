#!/usr/bin/env python

import os, sys, cv2, math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from layout_config import get_config
from layout_utils import *

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

import torch, torchtext
from torch.utils.data import Dataset


class layout_coco(Dataset):
    def __init__(self, config, split=None, transform=None):
        self.cfg = config
        self.split = split
        if split is not None:
            self.name = 'layout_coco' + '_' + split
        else:
            self.name = 'layout_coco'
        self.transform = transform
        self.root_dir  = osp.join(config.data_dir, 'coco')
        self.cache_dir = osp.abspath(osp.join(config.data_dir, 'caches'))
        maybe_create(self.cache_dir)
        
        # load COCO annotations, classes, class <-> id mappings
        self.cocoInstAPI = COCO(self.get_ann_file('instances'))
        self.cocoCaptAPI = COCO(self.get_ann_file('captions'))
        self.image_index = sorted(self.cocoInstAPI.getImgIds())
        cats = self.cocoInstAPI.loadCats(self.cocoInstAPI.getCatIds())
        self.classes = tuple(['<pad>', '<sos>', '<eos>'] + [c['name'] for c in cats])
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.class_to_coco_cat_id = \
            dict(zip([c['name'] for c in cats], self.cocoInstAPI.getCatIds()))

        # scene database
        self.scenedb = self.gt_scenedb()

        # vocab
        self.lang_vocab = self.build_lang_vocab(self.scenedb)

        # filter and sort
        if self.split != 'test':
            self.filter_scenedb()
            self.scenedb = [self.sort_objects(x) for x in self.scenedb]

        # grid mapping
        self.loc_map = CocoLocationMap(self.cfg)
        self.trans_map = CocoTransformationMap(self.cfg)
        colormap = create_colormap(len(self.classes))
        color_inds = np.random.permutation(range(len(colormap)))
        self.colormap = [colormap[x] for x in color_inds]
  
    def __len__(self):
        return len(self.scenedb)
    
    def __getitem__(self, idx):
        entry = {}
        scene = self.scenedb[idx].copy()
        entry['scene_idx'] = int(idx) 
        entry['image_idx'] = int(scene['img_idx'])
        entry['width'] = scene['width']
        entry['height'] = scene['height']

        ###################################################################
        ## Sentence
        ###################################################################
        group_sents = scene['captions']
        if self.split == 'train' and self.cfg.sent_group < 0:
            cid = np.random.randint(0, len(group_sents))
        else:
            cid = self.cfg.sent_group
        sentence = group_sents[cid]
        entry['sentence'] = sentence
        entry['cap_idx'] = scene['capIds'][cid]

        ###################################################################
        ## Indices
        ###################################################################
        # Word indices
        tokens = [w for w in word_tokenize(sentence.lower())]
        tokens = further_token_process(tokens)
        word_inds = [self.lang_vocab.word_to_index(w) for w in tokens]
        word_inds = [wi for wi in word_inds if wi > self.cfg.EOS_idx] 
        word_inds, word_msks = self.pad_sequence(word_inds, 
            self.cfg.max_input_length, self.cfg.PAD_idx, None, self.cfg.EOS_idx, 1.0)
        entry['word_inds'] = word_inds.astype(np.int32)
        entry['word_lens'] = np.sum(word_msks).astype(np.int32)

        # Output inds
        if len(scene['clses']) > 0:
            out_inds, out_msks = self.scene_to_output_inds(scene)  
            entry['out_inds'] = out_inds
            entry['out_msks'] = out_msks

            gt_fg_inds = deepcopy(out_inds[:,0]).astype(np.int32).flatten().tolist()
            gt_fg_inds = [self.cfg.SOS_idx] + gt_fg_inds
            gt_fg_inds = np.array(gt_fg_inds)
            entry['fg_inds'] = gt_fg_inds

            ###################################################################
            ## Images and Layouts
            ###################################################################
            entry['color_path'] = self.image_path_from_index(entry['image_idx'])
            vols = self.render_vols(out_inds, return_sequence=True)
            pad_vol = np.zeros_like(vols[-1])
            entry['background'], _ = \
                self.pad_sequence(vols, self.cfg.max_output_length, pad_vol, pad_vol, None, 0.0)

            ###################################################################
            ## Transformation
            ###################################################################
            if self.transform:
                entry = self.transform(entry)

        return entry

    def gt_scenedb(self):
        cache_file = osp.join(self.cache_dir, 'layout_coco_' + self.split + '.pkl')
        if osp.exists(cache_file):
            scenedb = pickle_load(cache_file)
            print('gt roidb loaded from {}'.format(cache_file))
            return scenedb
        scenedb = [self.load_coco_annotation(index) for index in self.image_index]
        if self.split != 'test':
            scenedb = self.load_split(self.split, scenedb)
        pickle_save(cache_file, scenedb)
        print('wrote gt roidb to {}'.format(cache_file))
        return scenedb

    def load_split(self, split, scenedb):
        split_path = osp.join(self.cache_dir, 'layout_' + split + '.txt')
        if osp.exists(split_path):
            split_inds = np.loadtxt(split_path, dtype=np.int32)
        else:
            inds = np.random.permutation(range(len(scenedb)))
            train_inds = inds[:-5000]
            val_inds  = inds[-5000:]
            
            train_path = osp.join(self.cache_dir, 'layout_train.txt')
            val_path  = osp.join(self.cache_dir, 'layout_val.txt')
            np.savetxt(train_path, sorted(train_inds), fmt='%d')
            np.savetxt(val_path,  sorted(val_inds),  fmt='%d')

            if split == 'train':
                split_inds = train_inds
            else:
                split_inds = val_inds
    
        split_inds = sorted(split_inds)
        split_db = [scenedb[i] for i in split_inds]
        return split_db

    def load_coco_annotation(self, index):
        """
        Loads COCO bounding boxes and caption annotations.
        Crowd instances are ignored.
        """
        im_ann = self.cocoInstAPI.loadImgs(index)[0]
        width  = im_ann['width']; height = im_ann['height']

        #######################################################################
        ## Make the image square
        #######################################################################
        max_dim = max(width, height)
        offset_x = 0 # int(0.5 * (max_dim - width))
        offset_y = max_dim - height # int(0.5 * (max_dim - height))

        #######################################################################
        ## Objects that are outside crowd regions
        #######################################################################
        objIds = self.cocoInstAPI.getAnnIds(imgIds=index, iscrowd=False)
        objs   = self.cocoInstAPI.loadAnns(objIds)
        capIds = self.cocoCaptAPI.getAnnIds(imgIds=index)
        caps   = self.cocoCaptAPI.loadAnns(capIds)

        #######################################################################
        ## Main information: normalized bounding boxes and class indices
        #######################################################################
        boxes = []
        clses = []
        areas = [] # for small objects filtering
        #######################################################################

        #######################################################################
        ## Lookup table to map from COCO category ids to our internal class indices
        ## Real object categories start from index 3
        #######################################################################
        start_idx = self.cfg.EOS_idx + 1
        coco_cat_id_to_class_ind = dict([(self.class_to_coco_cat_id[cls],
                                          self.class_to_ind[cls])
                                          for cls in self.classes[start_idx:]])

        #######################################################################
        ## For each object
        #######################################################################
        for i in range(len(objs)):
            obj = objs[i]
            #######################################################################
            ## Normalized bounding box
            #######################################################################
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))

            assert(x2 >= x1 and y2 >= y1)
            # area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

            x1 += offset_x; y1 += offset_y
            x2 += offset_x; y2 += offset_y
            cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
            nw = x2 - x1 + 1.0; nh = y2 - y1 + 1.0

            bb = np.array([cx, cy, nw, nh], dtype=np.float32)/max_dim
            #######################################################################
            ## Class index
            #######################################################################
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            area = bb[2] * bb[3]

            boxes.append(bb)
            clses.append(cls)
            areas.append(area)

        captions = [x['caption'].lower() for x in caps]

        return  {
            'img_idx'  : index,
            'captions' : captions,
            'boxes'    : np.array(boxes),
            'clses'    : np.array(clses),
            'areas'    : np.array(areas),
            'objIds'   : np.array(objIds),
            'capIds'   : np.array(capIds),
            'img_path' : self.image_path_from_index(index),
            'width'    : width,
            'height'   : height
        }

    def build_lang_vocab(self, scenedb):
        lang_vocab = Vocab('layout_lang_vocab')
        lang_vocab_path = osp.join(self.cache_dir, 'layout_lang_vocab.json')
        if osp.exists(lang_vocab_path):
            lang_vocab.load(lang_vocab_path)
        else:
            for i in range(len(scenedb)):
                group_sents = scenedb[i]['captions']
                for j in range(len(group_sents)):
                    lang_vocab.addSentence(group_sents[j])
            lang_vocab.filter_words(min_freq=5)
            lang_vocab.save(lang_vocab_path)
        lang_vocab.get_glovec()
        return lang_vocab

    def pad_sequence(self, inputs, max_length, pad_val, sos_val=None, eos_val=None, eos_msk=None):
        seq = inputs[:max_length]
        msk = [1.0] * len(seq)
        num_padding = max_length - len(seq)
        if sos_val is not None:
            if isinstance(sos_val, np.ndarray):
                seq = [sos_val.copy()] + seq
            else:
                seq = [sos_val] + seq
            msk = [1.0] + msk
        if eos_val is not None:
            if isinstance(eos_val, np.ndarray):
                seq.append(eos_val.copy())
            else:
                seq.append(eos_val)
            msk.append(eos_msk)
        for i in range(num_padding):
            if isinstance(pad_val, np.ndarray):
                seq.append(pad_val.copy())
            else:
                seq.append(pad_val)
            msk.append(0.0)
        return np.array(seq).astype(np.float32), np.array(msk).astype(np.float32)

    def get_ann_file(self, prefix):
        if self.split == 'val' or self.split == 'train':
            split = 'train'
        elif self.split == 'test':
            split = 'val'
        ann_path = osp.join(self.root_dir, 'annotations', prefix + '_' + split + '2017.json')
        assert osp.exists(ann_path), 'Path does not exist: {}'.format(ann_path)
        return ann_path

    def image_path_from_index(self, index):
        if self.split == 'val' or self.split == 'train':
            split = 'train'
        elif self.split == 'test':
            split = 'val'
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = osp.join(self.root_dir, 'images', split + '2017', file_name)
        assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    ####################################################################
    # Sort objects
    ####################################################################

    def sort_objects(self, scene):
        V = scene['areas'].copy()
        X = scene['boxes'][:, 0].copy()
        Y = scene['boxes'][:, 1].copy()
        H = scene['boxes'][:, 3].copy()
        Z = scene['clses'].copy()
        if self.cfg.object_first:
            indices = np.lexsort((X, -(Y+0.5*H), Z))
        else:
            indices = np.lexsort((Z, X, -(Y+0.5*H)))
        scene = self.select_objects(scene, indices)
        return scene

    ####################################################################
    # Filtering
    ####################################################################

    def select_objects(self, scene, indices):

        if len(indices) == 0:
            scene['boxes'] = np.zeros((0, 4))
            scene['clses'] = np.zeros((0,))
            scene['areas'] = np.zeros((0,))
            scene['objIds'] = np.zeros((0,))
        else:
            clses = scene['clses'].copy()
            boxes = scene['boxes'].copy()
            areas = scene['areas'].copy()
            objIds = scene['objIds'].copy()

            scene['boxes'] = boxes[indices, :].reshape((-1,4))
            scene['clses'] = clses[indices]
            scene['areas'] = areas[indices]
            scene['objIds'] = objIds[indices]

        return scene

    def filter_small_objects(self, scene):
        areas = scene['areas']
        if len(areas) < 1:
            return scene

        indices = np.where(areas > self.cfg.coco_min_area)[0]
        return self.select_objects(scene, indices)

    def filter_scenedb(self):
        def is_valid(entry):
            valid = entry['boxes'].shape[0] > 0
            return valid

        num = len(self.scenedb)
        self.scenedb = [self.filter_small_objects(x) for x in self.scenedb]
        filtered_scenedb = [entry for entry in self.scenedb if is_valid(entry)]
        num_after = len(filtered_scenedb)

        print('Filtered {} scenedb entries: {} -> {} '.format(num - num_after, num, num_after))
        self.scenedb = filtered_scenedb

    ####################################################################
    # Rendering
    ####################################################################
    def update_vol(self, vol, inds):
        if inds[0] <= self.cfg.EOS_idx:
            return vol

        w = vol.shape[-3]
        h = vol.shape[-2]
        xywh = self.index2box(inds[1:])
        xywh = xywh * np.array([w, h, w, h])
        xyxy = xywh_to_xyxy(xywh, w, h)

        vol[xyxy[1]:(xyxy[3]+1), xyxy[0]:(xyxy[2]+1), inds[0]] = 1.0

        return vol.astype(np.float32)

    def render_indices_as_output(self, entry, return_sequence=True):
        out_inds = deepcopy(np.array(entry['out_inds']))

        w = self.cfg.draw_size[0]
        h = self.cfg.draw_size[1]

        img = 255 * np.ones((h, w, 4), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            img, cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)

        xyxys = []
        imgs = []
        for i in range(len(out_inds)):
            cls_idx = out_inds[i, 0]
            if cls_idx <= self.cfg.EOS_idx:
                break
            other_inds = out_inds[i, 1:]
            color = self.colormap[cls_idx]
            xywh = self.index2box(other_inds)
            xywh = xywh * np.array([w, h, w, h])
            xyxy = xywh_to_xyxy(xywh, w, h)
            paint_box(ctx, color, xyxy)
            xyxys.append(xyxy)
            cls_txt = self.classes[cls_idx]
            paint_txt(ctx, cls_txt, xyxy)
            imgs.append(img[:,:,:-1].copy())
        
        if return_sequence:
            return np.stack(imgs, axis=0)
        else:
            return imgs[-1]

    def render_scene_as_output(self, scene, return_sequence=True, bg=None):
        w = self.cfg.draw_size[0]
        h = self.cfg.draw_size[1]

        img = 255 * np.ones((h, w, 4), dtype=np.uint8)
        if bg is not None:
            img[:,:,:3] = np.minimum(0.5 * bg + 128, 255)
        surface = cairo.ImageSurface.create_for_data(
            img, cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)

        xyxys = []; imgs = []
        clses = scene['clses']
        boxes = scene['boxes']
        for i in range(len(clses)):
            cls_idx = clses[i]
            if cls_idx <= self.cfg.EOS_idx:
                break
            color = self.colormap[cls_idx]
            xywh = boxes[i]
            xywh = xywh * np.array([w, h, w, h])
            xyxy = xywh_to_xyxy(xywh, w, h)
            paint_box(ctx, color, xyxy)
            xyxys.append(xyxy)

            cls_txt = self.classes[cls_idx]
            paint_txt(ctx, cls_txt, xyxy)
            imgs.append(img[:,:,:-1].copy())
        
        if return_sequence:
            return np.stack(imgs, axis=0)
        else:
            return imgs[-1]

    def render_vols(self, inds, return_sequence=False):
        w = self.cfg.input_size[0]
        h = self.cfg.input_size[1]
        
        vols = []
        vol = np.zeros((h, w, len(self.classes)), dtype=np.float32)
        for i in range(len(inds)):
            curr_inds = inds[i].copy()
            if curr_inds[0] <= self.cfg.EOS_idx:
                break
            vol = self.update_vol(vol, curr_inds)
            vols.append(vol.copy())

        if return_sequence:
            return vols
        else:
            return vols[-1]
     
    def box2index(self, box):
        coord_idx = self.loc_map.coord2index(box[:2])
        trans_idx = self.trans_map.wh2index(box[2:])
        return np.array([coord_idx, trans_idx[0], trans_idx[1]])

    def boxes2indices(self, boxes):
        coord_inds = self.loc_map.coords2indices(boxes[:,:2])
        trans_inds = self.trans_map.whs2indices(boxes[:,2:])
        out_inds = np.concatenate([coord_inds.reshape((-1, 1)), trans_inds], -1)
        return out_inds

    def index2box(self, inds):
        xy = self.loc_map.index2coord(inds[0])
        wh = self.trans_map.index2wh(inds[1:])
        box = np.concatenate((xy, wh), axis=-1)
        return box 
    
    def indices2boxes(self, inds):
        xys = self.loc_map.indices2coords(inds[:, 0])
        whs = self.trans_map.indices2whs(inds[:, 1:])
        boxes = np.concatenate((xys, whs), axis=-1)
        return boxes 

    ########################################################################
    ## Scene and indices
    ########################################################################
    def scene_to_output_inds(self, scene):
        class_inds = deepcopy(scene['clses'])
        boxes = deepcopy(scene['boxes'])
        other_inds = self.boxes2indices(boxes)
        out_inds = np.concatenate([class_inds.reshape((-1, 1)), other_inds], -1)
        out_inds, out_msks = self.pad_output_inds(out_inds)
        return out_inds.astype(np.int32), out_msks

    def output_inds_to_scene(self, inds):
        scene = {}

        n_objs = 0
        for i in range(len(inds)):
            if inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1
        
        val_inds = deepcopy(inds[:n_objs, :]).reshape((-1, 4)).astype(np.int)
        class_inds = val_inds[:, 0]
        other_inds = val_inds[:, 1:]

        scene['clses'] = class_inds
        scene['boxes'] = self.indices2boxes(other_inds)

        return scene 

    def pad_output_inds(self, inds):
        n_objs = 0
        for i in range(len(inds)):
            if inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1

        out_inds = deepcopy(inds[:n_objs, :]).reshape((-1, 4)).astype(np.int)
        # shorten if necessary
        out_inds = out_inds[:self.cfg.max_output_length]
        n_out = len(out_inds)
        n_pad = self.cfg.max_output_length - n_out
        out_msks = np.zeros((n_out+1, 4), dtype=float)
        out_msks[:n_out, :] = 1.0
        out_msks[n_out, 0] = 1.0
        eos_inds = np.zeros((1, 4))
        eos_inds[0, 0] = self.cfg.EOS_idx
        out_inds = np.concatenate([out_inds, eos_inds], 0)
        if n_pad > 0:
            pad_inds = np.ones((n_pad, 4)) * self.cfg.PAD_idx
            pad_msks = np.zeros((n_pad, 4))
            out_inds = np.concatenate([out_inds, pad_inds], 0)
            out_msks = np.concatenate([out_msks, pad_msks], 0)
        return out_inds, out_msks

    ########################################################################
    ## For captioning experiments
    ########################################################################
    def save_scene(self, out_dir, scene):
        clses = deepcopy(scene['clses'])
        boxes = deepcopy(scene['boxes'])

        img_idx = scene['img_idx']
        w = scene['width']; h = scene['height']

        out_list = []
        for i in range(len(clses)):
            if clses[i] <= self.cfg.EOS_idx:
                break
            label = self.classes[clses[i]]
            bb = boxes[i]
            # ob = self.recover_box(bb, w, h)
            # nb = self.normalize_box_for_captioning(ob, w, h)
            nb = 600 * bb
            entry = {}
            entry['left'] = nb[0]
            entry['top'] = nb[1]
            entry['width'] = nb[2]
            entry['height'] = nb[3]
            entry['label'] = label
            out_list.append(entry)
        
        out_dict = {}
        out_dict['image_id'] = img_idx
        out_dict['seq'] = out_list
        out_path = osp.join(out_dir, '%d.json'%img_idx)
        with open(out_path, 'w') as fp:
            json.dump(out_dict, fp, indent=4, sort_keys=True)

    def append_scene(self, scene):
        clses = np.array(deepcopy(scene['clses']))
        boxes = np.array(deepcopy(scene['boxes']))
        
        img_idx = scene['img_idx']
        w = scene['width']; h = scene['height']

        out_list = []
        for i in range(len(clses)):
            if clses[i] <= self.cfg.EOS_idx:
                break
            label = self.classes[clses[i]]
            bb = boxes[i]
            nb = 600 * bb
            entry = {}
            entry['left'] = max(nb[0] - 0.5 * nb[2],0)
            entry['top'] = max(nb[1] - 0.5 * nb[3], 0)
            entry['width'] = nb[2]
            entry['height'] = nb[3]
            entry['label'] = label
            out_list.append(entry)
        
        scene['image_id'] = img_idx
        scene['seq'] = out_list

        return scene

    def recover_box(self, bb, width, height):
        
        x = bb[0]; y = bb[1]; w = bb[2]; h = bb[3]
        max_dim = max(width, height)
        offset_x = 0 
        offset_y = max_dim - height 
        ix = max_dim * x - offset_x
        iy = max_dim * y - offset_y
        iw = max_dim * w 
        ih = max_dim * h

        lx = ix - 0.5 * iw
        ty = iy - 0.5 * ih
        lx = max(min(lx, width-1),0)
        ty = max(min(ty, height-1),0)
        iw = max(min(iw, width), 0)
        ih = max(min(ih, height),0)

        return np.array([lx, ty, iw, ih])

    def normalize_box_for_captioning(self, bb, width, height):
        foo = deepcopy(bb)
        foo = 600 * np.array(foo.flatten()).astype(np.float32)/np.array([width, height, width, height])
        return foo

    def overlay_boxes(self, entry, return_sequence=True):
        out_inds = deepcopy(np.array(entry['out_inds']))
        raw = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
        w = raw.shape[1]
        h = raw.shape[0]
        img = 255 * np.ones((h, w, 4), dtype=np.uint8)
        img[:,:,:3] = raw[:,:,::-1]
        surface = cairo.ImageSurface.create_for_data(
            img, cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)

        xyxys = []
        imgs = []
        for i in range(len(out_inds)):
            cls_idx = out_inds[i, 0]
            if cls_idx <= self.cfg.EOS_idx:
                break
            other_inds = out_inds[i, 1:]
            color = self.colormap[cls_idx]
            xywh = self.index2box(other_inds)
            # test recover
            ltwh = self.recover_box(xywh, entry['width'], entry['height'])
            xyxy = np.array([ltwh[0], ltwh[1], ltwh[0]+ltwh[2]-1, ltwh[1] + ltwh[3] - 1])
            paint_box(ctx, color, xyxy)
            xyxys.append(xyxy)

            cls_txt = self.classes[cls_idx]
            paint_txt(ctx, cls_txt, xyxy)
            
            imgs.append(img[:,:,:-1].copy())
        
        if return_sequence:
            return np.stack(imgs, axis=0)
        else:
            return imgs[-1]

    ########################################################################
    ## For demo
    ########################################################################
    def encode_sentence(self, sentence):
        tokens = [w for w in word_tokenize(sentence.lower())]
        tokens = further_token_process(tokens)
        word_inds = [self.lang_vocab.word_to_index(w) for w in tokens]
        word_inds = [wi for wi in word_inds if wi > self.cfg.EOS_idx] 
        word_inds, word_msks = self.pad_sequence(word_inds, 
            self.cfg.max_input_length, self.cfg.PAD_idx, None, self.cfg.EOS_idx, 1.0)
        return word_inds.astype(np.int32), np.sum(word_msks).astype(np.int32)
