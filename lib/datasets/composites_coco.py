#!/usr/bin/env python

import os, cv2, random
import numpy as np
import os.path as osp
from copy import deepcopy
from glob import glob
from collections import OrderedDict
from skimage.transform import resize as array_resize
from nms.cpu_nms import cpu_nms
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from composites_config import get_config
from composites_utils import *


class composites_coco(object):
    def __init__(self, config, split, year='2017'):
        self.cfg = config
        self.split = split
        self.year = year
        self.root_dir  = osp.join(config.data_dir, 'coco')
        self.cache_dir = osp.abspath(osp.join(config.data_dir, 'caches'))
        maybe_create(self.cache_dir)
        self.grid_map = NormalizedLocationMap(self.cfg)
        self.size_map = NormalizedTransformationMap(self.cfg)
        self.stuff_other_index = 183
        self.get_coco_general_info()
        self.color_palette = self.build_color_palette(len(self.classes))
        self.get_coco_split_info()
        self.lang_vocab = self.build_language_vocab()

        ###########################################################
        # use super category
        self.classes = self.super_classes
        self.category_ind_to_class = self.category_ind_to_super_class
        self.category_ind_to_class_ind = self.category_ind_to_super_class_ind
        for x in self.scenedb:
            x['clses'] = x['super_clses']
        
        self.cfg.output_vocab_size = len(self.classes)
        self.cfg.n_conv_hidden = 4 * self.cfg.output_vocab_size
        self.filter_scenedb()
        self.scenedb = [self.sort_objects(x) for x in self.scenedb]
        self.patchdb, self.name_to_patch_index = self.scenedb_to_patchdb(self.scenedb)
        indices = range(len(self.classes))
        self.patches_per_class = dict(zip(indices, [[] for i in indices]))
        for i in range(len(self.patchdb)):
            x = self.patchdb[i]
            category_id = x['class']
            self.patches_per_class[category_id].append(x)

    def get_coco_general_info(self):
        cache_file = osp.join(self.cache_dir, 'composites_coco_general.pkl')
        if osp.exists(cache_file):
            dataset_info = pickle_load(cache_file)
            print('dataset_info loaded from {}'.format(cache_file))

            self.classes = dataset_info['classes']
            self.super_classes = dataset_info['super_classes']

            self.class_to_ind = dataset_info['class_to_ind']
            self.super_class_to_ind = dataset_info['super_class_to_ind']

            self.category_ind_to_class = dataset_info['category_ind_to_class']
            self.category_ind_to_super_class = dataset_info['category_ind_to_super_class']

            self.category_ind_to_class_ind = dataset_info['category_ind_to_class_ind']
            self.category_ind_to_super_class_ind = dataset_info['category_ind_to_super_class_ind']   

            self.class_to_super_class = dataset_info['class_to_super_class']
            self.class_ind_to_super_class_ind = dataset_info['class_ind_to_super_class_ind']
        else:
            dataset_info = {}

            self.cocoInstAPI  = COCO(self.get_ann_file('instances'))
            self.cocoStuffAPI = COCO(self.get_ann_file('stuff'))
            self.classes, self.super_classes = [], []
            self.category_ind_to_class = OrderedDict()
            self.category_ind_to_class_ind = OrderedDict()
            self.category_ind_to_super_class = OrderedDict()
            self.category_ind_to_super_class_ind = OrderedDict()
            self.class_to_super_class = OrderedDict()
            self.class_ind_to_super_class_ind = OrderedDict()


            coco_thing_category_inds = self.cocoInstAPI.getCatIds()
            coco_stuff_category_inds = self.cocoStuffAPI.getCatIds()

            for category_id in coco_thing_category_inds:
                category = self.cocoInstAPI.loadCats(category_id)[0]
                name = category['name']
                self.classes.append(name)
                self.super_classes.append(name)
                self.category_ind_to_class[category_id] = name
                self.category_ind_to_super_class[category_id] = name
                self.class_to_super_class[name] = name

            for category_id in coco_stuff_category_inds:
                category = self.cocoStuffAPI.loadCats(category_id)[0]
                name = category['name']
                supercategory = category['supercategory']
                if name == 'other':
                    continue
                self.classes.append(name)
                if not (supercategory in self.super_classes):
                    self.super_classes.append(supercategory)
                self.category_ind_to_class[category_id] = name
                self.category_ind_to_super_class[category_id] = supercategory
                self.class_to_super_class[name] = supercategory
                
            self.classes = ['<pad>', '<sos>', '<eos>'] + self.classes
            self.super_classes = ['<pad>', '<sos>', '<eos>'] + self.super_classes
            self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
            self.super_class_to_ind = dict(zip(self.super_classes, range(len(self.super_classes))))
            
            for key, val in self.category_ind_to_class.items():
                self.category_ind_to_class_ind[key] = self.class_to_ind[val]

            for key, val in self.category_ind_to_super_class.items():
                self.category_ind_to_super_class_ind[key] = self.super_class_to_ind[val]

            for key, val in self.class_to_super_class.items():
                self.class_ind_to_super_class_ind[self.class_to_ind[key]] = self.super_class_to_ind[val]

            dataset_info['classes'] = self.classes
            dataset_info['super_classes'] = self.super_classes

            dataset_info['class_to_ind'] = self.class_to_ind
            dataset_info['super_class_to_ind'] = self.super_class_to_ind

            dataset_info['category_ind_to_class'] = self.category_ind_to_class
            dataset_info['category_ind_to_super_class'] = self.category_ind_to_super_class

            dataset_info['category_ind_to_class_ind'] = self.category_ind_to_class_ind
            dataset_info['category_ind_to_super_class_ind'] = self.category_ind_to_super_class_ind

            dataset_info['class_to_super_class'] = self.class_to_super_class
            dataset_info['class_ind_to_super_class_ind'] = self.class_ind_to_super_class_ind

            pickle_save(cache_file, dataset_info)
            print('wrote dataset_info to {}'.format(cache_file))

        # temp hack for the special tokens
        for i in range(3):
            self.class_ind_to_super_class_ind[i] = i

    def get_coco_split_info(self):
        cache_file = osp.join(self.cache_dir, 'composites_coco_%s.pkl'%(self.split))
        if osp.exists(cache_file):
            self.scenedb = pickle_load(cache_file)
            print('scenedb loaded from {}'.format(cache_file))
        else:
            if getattr(self, 'cocoInstAPI', None) is None:
                self.cocoInstAPI  = COCO(self.get_ann_file('instances'))
            if getattr(self, 'cocoCaptAPI', None) is None:
                self.cocoCaptAPI  = COCO(self.get_ann_file('captions'))
            if getattr(self, 'cocoStuffAPI', None) is None:
                self.cocoStuffAPI = COCO(self.get_ann_file('stuff'))
            # image_indices = sorted(self.cocoInstAPI.getImgIds())
            split_path = osp.join(self.cache_dir, 'composites_%s_split.txt'%self.split)
            split_inds = np.loadtxt(split_path, dtype=np.int32)
            split_inds = sorted(split_inds)
            scenedb = [self.load_coco_annotation(int(index)) for index in split_inds]
            pickle_save(cache_file, scenedb)
            print('wrote valdb to {}'.format(cache_file))
            self.scenedb = scenedb

    def build_color_palette(self, num_colors):
        color_palette_path = osp.join(self.cache_dir, 'composites_color_palette.txt')
        if osp.exists(color_palette_path):
            print("Loading color palette.")
            color_palette = np.genfromtxt(color_palette_path, dtype=np.int32)
        else:
            colormap = create_colormap(num_colors)
            color_inds = np.random.permutation(range(len(colormap)))
            colormap = [colormap[x] for x in color_inds]
            color_palette = clamp_array(np.array(colormap) * 255, 0, 255).astype(np.int32)
            np.savetxt(color_palette_path, color_palette, fmt='%d %d %d')
        return color_palette

    def build_language_vocab(self):
        lang_name  = 'composites_coco_lang_vocab'
        lang_vocab = Vocab(lang_name)
        lang_vocab_path = osp.join(self.cache_dir, lang_name+'.json')
        if osp.exists(lang_vocab_path):
            lang_vocab.load(lang_vocab_path)
        else:
            traindb_path = osp.join(self.cache_dir, 'composites_coco_train.pkl')
            scenedb = pickle_load(traindb_path)
            for i in range(len(scenedb)):
                group_sents = scenedb[i]['captions']
                for j in range(len(group_sents)):
                    lang_vocab.addSentence(group_sents[j])
            lang_vocab.filter_words(min_freq=5)
            lang_vocab.save(lang_vocab_path)
        lang_vocab.get_glovec()
        return lang_vocab

    def load_coco_annotation(self, index):
        # print(index, self.split)
        im_ann = self.cocoInstAPI.loadImgs(index)[0]
        width  = im_ann['width']; height = im_ann['height']

        #######################################################################
        ## Objects that are outside crowd regions
        #######################################################################
        objectIds  = self.cocoInstAPI.getAnnIds(imgIds=index, iscrowd=False)
        objects    = self.cocoInstAPI.loadAnns(objectIds)
        stuffIds   = self.cocoStuffAPI.getAnnIds(imgIds=index, iscrowd=False)
        stuffs     = self.cocoStuffAPI.loadAnns(stuffIds)
        captionIds = self.cocoCaptAPI.getAnnIds(imgIds=index)
        captions   = self.cocoCaptAPI.loadAnns(captionIds)

        #######################################################################
        ## Main information: bounding box, class, area, mask, instance id
        #######################################################################
        boxes, clses, super_clses, areas, masks, instance_inds = [], [], [], [], [], []
        #######################################################################


        ######################################################################
        # For each thing object
        ######################################################################
        for i in range(len(objects)):
            obj = objects[i]
            segm = obj['segmentation']
            #######################################################################
            ## RLE representation of the MASK
            #######################################################################
            if type(segm) == list:
                # polygon
                rle = COCOmask.frPyObjects(segm, height, width)
                rle = COCOmask.merge(rle)
            elif type(segm['counts']) == list:
                rle = COCOmask.frPyObjects(segm, height, width)
            else:
                rle = segm

            #######################################################################
            ## Area
            #######################################################################
            area = COCOmask.area(deepcopy(rle))

            #######################################################################
            ## Bounding box in the xywh format
            #######################################################################
            bbox = obj['bbox']
            x1 = np.max((0, bbox[0]))
            y1 = np.max((0, bbox[1]))
            x2 = np.min((width - 1, x1 + np.max((0, bbox[2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, bbox[3] - 1))))
            assert(x2 >= x1 and y2 >= y1)
            cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
            nw = x2 - x1 + 1.0; nh = y2 - y1 + 1.0
            bb = np.array([cx, cy, nw, nh], dtype=np.float32)
            #######################################################################
            ## Class index
            #######################################################################
            cls_idx = self.category_ind_to_class_ind[obj['category_id']]
            super_cls_idx = self.category_ind_to_super_class_ind[obj['category_id']]
            instance_index = str(obj['id']).zfill(12)

            boxes.append(bb)
            clses.append(cls_idx)
            super_clses.append(super_cls_idx)
            areas.append(area)
            masks.append(rle)
            instance_inds.append(instance_index)

        #######################################################################
        ## For each stuff object
        #######################################################################
        for i in range(len(stuffs)):
            stu = stuffs[i]
            #######################################################################
            ## Class index
            #######################################################################
            coco_stuff_category_ind = stu['category_id']
            # skip the 'other' (not annotated) category
            if coco_stuff_category_ind == self.stuff_other_index:
                continue
            cls_idx = self.category_ind_to_class_ind[coco_stuff_category_ind]
            super_cls_idx = self.category_ind_to_super_class_ind[coco_stuff_category_ind]

            #######################################################################
            ## RLE representation of the MASK
            #######################################################################
            segm = stu['segmentation']
            if type(segm) == list:
                # polygon
                rle = COCOmask.frPyObjects(segm, height, width)
                rle = COCOmask.merge(rle)
            elif type(segm['counts']) == list:
                rle = COCOmask.frPyObjects(segm, height, width)
            else:
                rle = segm

            #######################################################################
            ## Morphological opening
            #######################################################################
            rle_mask = COCOmask.decode(rle)
            rle_mask = clamp_array(255.0 * rle_mask, 0, 255).astype(np.uint8)
            morp_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
            rle_mask = cv2.morphologyEx(rle_mask, cv2.MORPH_OPEN, morp_kernel)

            #######################################################################
            ## Connected components
            #######################################################################
            n_components, labels, stats, _ = cv2.connectedComponentsWithStats(rle_mask)

            for j in range(1, n_components): # skip the background
                component_mask = (labels == j).astype(np.uint8)
                component_mask = np.asfortranarray(component_mask)
                component_rle = COCOmask.encode(component_mask)
                #######################################################################
                ## Bounding box in the xywh format
                #######################################################################
                bbox = COCOmask.toBbox(deepcopy(component_rle))
                area = COCOmask.area(deepcopy(component_rle))
                bbox = np.array(bbox).flatten()
                
                x1 = np.max((0, bbox[0]))
                y1 = np.max((0, bbox[1]))
                x2 = np.min((width - 1, x1 + np.max((0, bbox[2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, bbox[3] - 1))))
                assert(x2 >= x1 and y2 >= y1)
                cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
                nw = x2 - x1 + 1.0; nh = y2 - y1 + 1.0
                bb = np.array([cx, cy, nw, nh], dtype=np.float32)

                boxes.append(bb)
                clses.append(cls_idx)
                super_clses.append(super_cls_idx)
                areas.append(area)
                masks.append(component_rle)
                instance_index = str(stu['id']).zfill(12) + '_' + str(j).zfill(5)
                instance_inds.append(instance_index)

        captions = [x['caption'].lower() for x in captions]

        return  {
            'image_index' : index,
            'captions' : captions,
            'caption_inds' : np.array(captionIds),
            'instance_inds' : instance_inds,
            'boxes' : np.array(boxes),
            'clses' : np.array(clses),
            'super_clses': np.array(super_clses),
            'areas' : np.array(areas),
            'masks' : masks,
            'width'  : width,
            'height' : height
        }

    def scenedb_to_patchdb(self, scenedb):
        patchdb = []
        for i in range(len(scenedb)):
            scene = scenedb[i]
            masks = scene['masks']

            for j in range(len(masks)):
                entry = {}
                entry['scene_index'] = i
                entry['image_index'] = scene['image_index']
                entry['instance_ind'] = scene['instance_inds'][j]
                entry['name'] = str(entry['image_index']).zfill(12) + '_' + str(entry['instance_ind']).zfill(12)
                entry['box'] = scene['boxes'][j]
                entry['mask'] = masks[j]
                entry['class'] = scene['clses'][j]
                entry['area'] = scene['areas'][j]
                entry['width'] = scene['width']
                entry['height'] = scene['height']
                patchdb.append(entry)
        # name_to_patch_index: used to locate the patch
        name_to_patch_index = {}
        for i in range(len(patchdb)):
            x = patchdb[i]
            name_to_patch_index[x['name']] = i

        return patchdb, name_to_patch_index

    def encode_semantic_map(self, semantic):
        n = len(self.classes)
        h = semantic.shape[0]
        w = semantic.shape[1]
        out = np.zeros((h, w), dtype=np.int32)
        for k in range(n):
            eqR = semantic[:,:,0] == self.color_palette[k,0]
            eqG = semantic[:,:,1] == self.color_palette[k,1]
            eqB = semantic[:,:,2] == self.color_palette[k,2]
            msk = np.float32((eqR)&(eqG)&(eqB))
            out[msk>0]=k
        return out

    def decode_semantic_map(self, semantic):
        # prediction = np.argmax(semantic, 0)
        prediction = semantic
        h = prediction.shape[0]
        w = prediction.shape[1]
        color_image = self.color_palette[prediction.ravel()].reshape((h,w,3))
        # row, col = np.where(np.sum(semantic, 0)==0)
        row, col = np.where(semantic < 3)
        color_image[row, col, :] = 0
        return color_image
    
    ####################################################################
    # Paths
    ####################################################################
    def get_ann_file(self, prefix):
        if (self.split == 'train') or (self.split == 'val') or (self.split == 'aux'):
            split = 'train'
        else:
            split = 'val'
        ann_path = osp.join(self.root_dir, 'annotations', prefix + '_' + split + self.year+'.json')
        assert osp.exists(ann_path), 'Path does not exist: {}'.format(ann_path)
        return ann_path

    def color_path_from_index(self, index):
        if (self.split == 'train') or (self.split == 'val') or (self.split == 'aux'):
            split = 'train'
        else:
            split = 'val'
        if self.year == '2014':
            file_name = ('COCO_'+split+self.year+'_'+str(index).zfill(12) + '.jpg')
        else:
            file_name = (str(index).zfill(12) + '.jpg')
        image_path = osp.join(self.root_dir, 'images', split + self.year, file_name)
        # assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def image_path_from_index(self, index, field, ext):
        if (self.split == 'train') or (self.split == 'val') or (self.split == 'aux'):
            # train/val split
            split = 'train'
        else:
            split = 'val'
        file_name = (str(index).zfill(12) + '.' + ext)
        file_path = osp.join(self.root_dir, field, split + self.year, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def patch_path_from_indices(self, image_index, instance_index, field, ext, use_background=None):
        if (self.split == 'train') or (self.split == 'val') or (self.split == 'aux'):
            split = 'train'
        else:
            split = 'val'
        image_index_string = str(image_index).zfill(12)
        instance_index_string = str(instance_index).zfill(12)
        file_name = (image_index_string + '_' + instance_index_string + '.' + ext)
        if use_background is not None:
            if use_background:
                folder_name = field + '_with_bg'
            else:
                folder_name = field + '_without_bg'
        else:
            folder_name = field
        path = osp.join(self.root_dir, folder_name, split + self.year, image_index_string, file_name)
        # assert osp.exists(path), 'Path does not exist: {}'.format(path)
        return path

    ####################################################################
    # Sort objects
    ####################################################################

    def sort_objects(self, scene):
        # V = scene['areas'].copy()
        X = scene['boxes'][:, 0].copy()
        Y = scene['boxes'][:, 1].copy()
        H = scene['boxes'][:, 3].copy()
        Z = scene['clses'].copy()

        L = np.ones_like(Z).astype(np.float32)
        object_inds = np.where(Z < 83)[0]
        L[object_inds] = 0.0

        indices = np.lexsort((Z, X, -(Y + 0.5 * H), L))
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
            scene['instance_inds'] = []
            scene['masks'] = []
        else:
            clses = scene['clses'].copy()
            boxes = scene['boxes'].copy()
            areas = scene['areas'].copy()
            instance_inds = scene['instance_inds'].copy()
            masks = deepcopy(scene['masks'])

            scene['boxes'] = boxes[indices, :].reshape((-1,4))
            scene['clses'] = clses[indices]
            scene['areas'] = areas[indices]
            scene['instance_inds'] = [instance_inds[i] for i in indices]
            scene['masks'] = [masks[i] for i in indices]
        return scene

    def nms(self, scene, thresh):
        areas = scene['areas'].copy()
        if len(areas) < 1:
            return scene
        h = scene['height']
        w = scene['width']
        xywhs = scene['boxes'].copy()
        xyxys = xywhs_to_xyxys(xywhs, w, h)
        objs = np.concatenate([xyxys, areas[...,None]], -1).astype(np.float32)
        indices = np.array(cpu_nms(objs, thresh))
        # if len(indices) < len(areas):
        #     print(scene['image_index'], "%d --> %d"%(len(areas), len(indices)))
        return self.select_objects(scene, indices)

    def filter_small_objects(self, scene):
        clses = scene['clses'].copy()
        areas = scene['areas'].copy()
        if len(areas) < 1:
            return scene
        image_width = scene['width']
        image_height = scene['height']
        image_area = float(image_width * image_height)
        areas = areas/image_area

        indices = []
        for i in range(len(areas)):
            cate = clses[i]
            area = areas[i]
            if cate < 83 and area > self.cfg.coco_min_area:
                indices.append(i)
            elif area > 0.04:
                indices.append(i)

        indices = np.array(indices)
        return self.select_objects(scene, indices)

    def filter_concave_stuffs(self, scene):
        clses = scene['clses'].copy()
        xywhs = scene['boxes'].copy()
        areas = scene['areas'].copy()
        if len(areas) < 1:
            return scene
        ratios = areas/(xywhs[:,2]*xywhs[:,3]+self.cfg.eps)
        indices = []
        for i in range(len(areas)):
            cate = clses[i]
            ratio = ratios[i]
            if cate < 83 or ratio > self.cfg.coco_min_ratio:
                indices.append(i)
        indices = np.array(indices)
        return self.select_objects(scene, indices)

    def filter_scenedb(self):
        def is_valid_1(entry):
            valid = len(entry['clses']) > 0
            return valid
        def is_valid_2(entry):
            valid = (len(entry['clses']) > 0) and (len(entry['clses']) <= self.cfg.max_output_length)
            return valid
        num = len(self.scenedb)
        if self.split == 'train':
            self.scenedb = [self.nms(x, 0.8) for x in self.scenedb]
            self.scenedb = [entry for entry in self.scenedb if is_valid_1(entry)]
            self.scenedb = [self.filter_small_objects(x) for x in self.scenedb]
            # self.scenedb = [self.filter_concave_stuffs(x) for x in self.scenedb]                                                                    
            self.scenedb = [entry for entry in self.scenedb if is_valid_2(entry)]
            # self.scenedb = [entry for entry in self.scenedb if is_valid_1(entry)]                                                                   
        elif self.split == 'test':
            self.scenedb = [self.filter_small_objects(x) for x in self.scenedb]
            self.scenedb = [entry for entry in self.scenedb if is_valid_1(entry)]
        else:
            self.scenedb = [entry for entry in self.scenedb if is_valid_1(entry)]
        num_after = len(self.scenedb)
        print('Filtered {} scenedb entries: {} -> {} '.format(num - num_after, num, num_after))

    ########################################################################
    ## Scene and indices
    ########################################################################
    def box2index(self, normalized_xywh):
        coord_idx = self.grid_map.coord2index(normalized_xywh[:2])
        trans_idx = self.size_map.wh2index(normalized_xywh[2:])
        return np.array([coord_idx, trans_idx[0], trans_idx[1]])

    def boxes2indices(self, normalized_xywhs):
        coord_inds = self.grid_map.coords2indices(normalized_xywhs[:,:2])
        trans_inds = self.size_map.whs2indices(normalized_xywhs[:,2:])
        out_inds = np.concatenate([coord_inds.reshape((-1, 1)), trans_inds], -1)
        return out_inds

    def index2box(self, inds):
        xy = self.grid_map.index2coord(inds[0])
        wh = self.size_map.index2wh(inds[1:])
        box = np.concatenate((xy, wh), axis=-1)
        return box

    def indices2boxes(self, inds):
        xys = self.grid_map.indices2coords(inds[:, 0])
        whs = self.size_map.indices2whs(inds[:, 1:])
        boxes = np.concatenate((xys, whs), axis=-1)
        return boxes

    def pad_output_inds(self, inds):
        n_objs = 0
        for i in range(len(inds)):
            if inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1
        # only indices indicating objects are valid
        out_inds = deepcopy(inds[:n_objs, :]).reshape((-1, 4)).astype(np.int32)
        # shorten if necessary
        out_inds = out_inds[:self.cfg.max_output_length]
        n_out = len(out_inds)
        n_pad = self.cfg.max_output_length - n_out
        out_msks = np.zeros((n_out+1, 4), dtype=float)
        out_msks[:n_out, :] = 1.0
        # The model is supposed to predict EOS at the n_out-th step
        out_msks[n_out, 0] = 1.0
        eos_inds = np.zeros((1, 4))
        eos_inds[0, 0] = self.cfg.EOS_idx
        out_inds = np.concatenate([out_inds, eos_inds], 0)
        # pad if necessary
        if n_pad > 0:
            pad_inds = np.ones((n_pad, 4)) * self.cfg.PAD_idx
            pad_msks = np.zeros((n_pad, 4))
            out_inds = np.concatenate([out_inds, pad_inds], 0)
            out_msks = np.concatenate([out_msks, pad_msks], 0)
        return out_inds, out_msks

    def scene_to_prediction_outputs(self, scene):
        class_inds = deepcopy(scene['clses'])
        xywhs      = deepcopy(scene['boxes'])
        normalized_xywhs = normalize_xywhs(xywhs, scene['width'], scene['height'])
        other_inds = self.boxes2indices(normalized_xywhs)
        out_inds = np.concatenate([class_inds.reshape((-1, 1)), other_inds], -1)
        out_inds, out_msks = self.pad_output_inds(out_inds)
        out_inds = out_inds.astype(np.int32)

        # output vectors in pca feature space
        image_index = scene['image_index']
        instance_indices = scene['instance_inds']
        # out_vecs = []
        # for i in range(len(instance_indices)):
        #     instance_ind = instance_indices[i]
        #     feature_path = self.patch_path_from_indices(image_index, instance_ind, 'patch_feature', 'pkl', self.cfg.use_patch_background)
        #     if osp.exists(feature_path):
        #         with open(feature_path, 'rb') as fid:
        #             vec = pickle.load(fid)
        #     else:
        #         vec = np.zeros((self.cfg.n_patch_features,))
        #     out_vecs.append(vec)
        out_vecs = [np.zeros((self.cfg.n_patch_features,)) for i in range(len(instance_indices))]
        pad_vec = np.zeros_like(out_vecs[-1])
        out_vecs, _ = pad_sequence(out_vecs, self.cfg.max_output_length, pad_vec, None, pad_vec, 0.0)

        outputs = {}
        outputs['out_inds'] = out_inds
        outputs['out_msks'] = out_msks
        outputs['out_vecs'] = out_vecs

        return outputs

    def prediction_outputs_to_scene(self, outputs, nn_tables):
        out_inds = np.array(outputs['out_inds'])
        out_vecs = np.array(outputs['out_vecs'])

        scene = {}
        n_objs = 0
        for i in range(len(out_inds)):
            if out_inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1

        valid_inds = out_inds[:n_objs, :].reshape((-1, 4)).astype(np.int32)
        class_inds = valid_inds[:, 0]
        other_inds = valid_inds[:, 1:]
        normalized_xywhs = self.indices2boxes(other_inds)

        # the output may be "larger"
        scene['width'] = self.cfg.output_image_size[1]
        scene['height'] = self.cfg.output_image_size[0]
        scene['clses'] = class_inds
        scene['boxes'] = unnormalize_xywhs(normalized_xywhs,
                            scene['width'], scene['height'])

        # # retreving the nearest patch in feature space
        # patches = []
        # for i in range(n_objs):
        #     category_id = class_inds[i]
        #     query_vector = out_vecs[i]
        #     patch = nn_tables.retrieve(category_id, query_vector)
        #     patches.append(patch)
        # scene['patches'] = patches
        return scene

    ####################################################################
    # Rendering
    ####################################################################

    def render_label(self, ref_scene, return_sequence=False):
        w = ref_scene['width']
        h = ref_scene['height']
        label = np.zeros((h, w), dtype=np.uint8)
        label_seq = []
        clses = ref_scene['clses']
        masks = ref_scene['masks']
        for i in range(len(clses)):
            cls_idx = clses[i]
            if cls_idx <= self.cfg.EOS_idx:
                break
            rle_rep = masks[i]
            mask = COCOmask.decode(rle_rep)
            label[mask>0] = cls_idx
            label_seq.append(label.copy())

        if len(label_seq) == 0:
            label_seq.append(label)

        if return_sequence:
            return label_seq
        else:
            return label_seq[-1]

    def get_volume_from_indices(self, image_index, instance_index):
        label_path = self.patch_path_from_indices(image_index, instance_index, 'patch_label', 'png', None)
        # print(label_path)
        mask_path  = self.patch_path_from_indices(image_index, instance_index, 'patch_mask',  'png', None)
        color_path = self.patch_path_from_indices(image_index, instance_index, 'patch_color', 'jpg', self.cfg.use_patch_background)
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        mask  = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # if self.cfg.use_super_category:
        #     label = map(lambda x: self.class_ind_to_super_class_ind[x], label.flatten().tolist())
        #     label = np.array(list(label)).reshape((mask.shape[0], mask.shape[1]))
        # use super category
        label = map(lambda x: self.class_ind_to_super_class_ind[x], label.flatten().tolist())
        label = np.array(list(label)).reshape((mask.shape[0], mask.shape[1]))
        return np.concatenate([label[..., None], mask[..., None], color], -1)

    def render_volumes_for_a_scene(self, ref_scene, return_sequence=True, nn_tables=None):
        input_width  = ref_scene['width']
        input_height = ref_scene['height']
        image_index  = ref_scene['image_index']

        bg_width  = self.cfg.input_image_size[1]
        bg_height = self.cfg.input_image_size[0]
        fg_width  = self.cfg.input_patch_size[1]
        fg_height = self.cfg.input_patch_size[0]

        # For foreground patch
        raw_image = cv2.imread(self.color_path_from_index(image_index), cv2.IMREAD_COLOR)
        input_image, _, _ = create_squared_image(raw_image, np.array([0, 0, 0]))
        input_image = cv2.resize(input_image, (bg_width, bg_height), interpolation = cv2.INTER_CUBIC)
        bg_label = np.zeros((bg_height, bg_width), dtype=np.uint8)
        bg_mask = np.zeros((bg_height, bg_width), dtype=np.float32)

        bg_seq = []
        fg_seq, neg_seq, fg_resnets, neg_resnets = [], [], [], []
        clses = ref_scene['clses']
        boxes = ref_scene['boxes']
        masks = ref_scene['masks']
        instance_inds = ref_scene['instance_inds']

        for i in range(len(clses)):
            cls_idx = clses[i]
            if cls_idx <= self.cfg.EOS_idx:
                break
            rle_rep = masks[i]
            mask = COCOmask.decode(rle_rep)
            squared_mask, _, _ = create_squared_image(mask[..., None], np.array([0]))
            squared_mask = cv2.resize(squared_mask, (bg_width, bg_height), interpolation = cv2.INTER_NEAREST)
            bg_mask = np.maximum(bg_mask, squared_mask)
            bg_label[squared_mask>0] = cls_idx
            tmp_mask = bg_mask[..., None].copy()
            bg_vols = np.concatenate([bg_label[..., None], tmp_mask * 255, input_image * tmp_mask], -1).copy()
            bg_seq.append(bg_vols)

        for i in range(len(bg_seq)):
            cls_idx = clses[i]
            # foreground layers
            fg_instance_ind = instance_inds[i]
            fg_vols = self.get_volume_from_indices(image_index, fg_instance_ind)
            fg_seq.append(fg_vols)

            if self.cfg.use_global_resnet:
                fg_resnet_path = self.image_path_from_index(image_index, 'image_resnet152', 'pkl')
            else:
                fg_resnet_path = self.patch_path_from_indices(image_index, fg_instance_ind, 'patch_resnet152', 'pkl', True)
            fg_resnets.append(pickle_load(fg_resnet_path))

            # Negative sample
            pred_feat_path = self.patch_path_from_indices(image_index, fg_instance_ind, 'predicted_feature', 'pkl', None)
            if (nn_tables is not None) and osp.exists(pred_feat_path) and (random.random() < 0.5) and self.cfg.use_hard_mining:
                query_vector = pickle_load(pred_feat_path)
                n_samples = min(self.cfg.n_nntable_trees, len(self.patches_per_class[cls_idx]))
                candidate_patches = nn_tables.retrieve(cls_idx, query_vector, n_samples)
                # print('nn sample', len(candidate_patches))
            else:
                candidate_patches = self.patches_per_class[cls_idx]
                # print('random sample', len(candidate_patches))
            assert(len(candidate_patches) > 1)
            candidate_instance_ind = fg_instance_ind
            candidate_patch = None
            while (candidate_instance_ind == fg_instance_ind):
                cid = np.random.randint(0, len(candidate_patches))
                candidate_patch = candidate_patches[cid]
                candidate_instance_ind = candidate_patch['instance_ind']
            neg_image_index = candidate_patch['image_index']
            neg_instance_ind = candidate_patch['instance_ind']
            neg_vols = self.get_volume_from_indices(neg_image_index, neg_instance_ind)
            # neg_layers = array_resize(neg_layers, (fg_width, fg_height, 3+len(self.classes)),
            #     order=1, mode='reflect', preserve_range=True, anti_aliasing=True)
            neg_seq.append(neg_vols)
            if self.cfg.use_global_resnet:
                neg_resnet_path = self.image_path_from_index(neg_image_index, 'image_resnet152', 'pkl')
            else:
                neg_resnet_path = self.patch_path_from_indices(neg_image_index, neg_instance_ind, 'patch_resnet152', 'pkl', True)
            neg_resnet_features = pickle_load(neg_resnet_path)
            neg_resnets.append(neg_resnet_features)

        if return_sequence:
            return bg_seq, fg_seq, neg_seq, fg_resnets, neg_resnets
        else:
            return bg_seq[-1], fg_seq[-1], neg_seq[-1], fg_resnets[-1], neg_resnets[-1]

    def render_reference_scene(self, ref_scene, return_sequence=True):
        w = ref_scene['width']; h = ref_scene['height']
        input_image = cv2.imread(self.color_path_from_index(ref_scene['image_index']), cv2.IMREAD_COLOR)
        composed_image = 128 * np.ones((h, w, 3), dtype=np.uint8)

        imgs = []
        clses = ref_scene['clses']
        boxes = ref_scene['boxes']
        masks = ref_scene['masks']
        instance_inds = ref_scene.get('instance_inds', range(len(clses)))
        for i in range(len(clses)):
            cls_idx = clses[i]
            if cls_idx <= self.cfg.EOS_idx:
                break
            rle_rep = masks[i]
            mask = COCOmask.decode(rle_rep)
            composed_image = compose(composed_image, input_image, mask)
            output_image = composed_image.copy()
            xywh = boxes[i]
            xyxy = xywh_to_xyxy(xywh, w, h)
            # print(self.classes[cls_idx], -(xywh[1]+0.5*xywh[3]))
            cv2.rectangle(output_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 0), 2)
            cv2.putText(output_image, self.classes[cls_idx],
                (xyxy[0], xyxy[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            imgs.append(output_image)

        if return_sequence:
            return np.stack(imgs, axis=0)
        else:
            return imgs[-1]

    ####################################################################
    # For demo
    ####################################################################
    def encode_sentence(self, sentence):
        tokens = [w for w in word_tokenize(sentence.lower())]
        tokens = further_token_process(tokens)
        word_inds = [self.lang_vocab.word_to_index(w) for w in tokens]
        word_inds = [wi for wi in word_inds if wi > self.cfg.EOS_idx]
        word_inds, word_msks = pad_sequence(word_inds, self.cfg.max_input_length, self.cfg.PAD_idx, None, self.cfg.EOS_idx, 1.0)
        return word_inds.astype(np.int32), np.sum(word_msks).astype(np.int32)
