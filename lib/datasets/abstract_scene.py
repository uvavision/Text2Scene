#!/usr/bin/env python

import os, sys, cv2, json, math, PIL
import cairo, random, torch, torchtext
import numpy as np
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from torch.utils.data import Dataset


# from abstract_evaluator import scene_graph, evaluator
from abstract_config import get_config
from abstract_utils import *


class abstract_scene(Dataset):
    def __init__(self, config, split=None, transform=None):
        self.cfg = config
        self.split = split
        if split is not None:
            self.name = 'AbstractScenes' + '_' + split
        else:
            self.name = 'AbstractScenes'
        self.transform = transform
        self.root_dir  = osp.join(config.data_dir, 'AbstractScenes_v1.1')
        self.cache_dir = osp.join(config.data_dir, 'caches')
        maybe_create(self.cache_dir)

        #################################################################
        # Grid location wrapper
        #################################################################
        self.location_map = LocationMap(config)
        #################################################################
        # Each entry in the scenedb stores meta information of the scene
        #################################################################
        scenedb = self.load_scene(self.root_dir)
        #################################################################
        # Language dictionary
        #################################################################
        self.lang_vocab = self.build_lang_vocab(scenedb)
        #################################################################
        # Segment dictionary 
        #################################################################
        self.segment_vocab = self.build_segment_vocab(scenedb)
        #################################################################
        # Sort the objects in the order they should be drawn
        #################################################################
        for i in range(len(scenedb)):
            scenedb[i] = self.sort_segments(scenedb[i])

        #################################################################
        # Help create bounding boxes
        #################################################################
        self.load_segment_sizes()
        if split is not None:
            # based on the AbstractScenes_v1.1 dataset splits in the paper
            # "Learning the Visual Interpretation of Sentences"
            self.scenedb = self.load_split(split, scenedb)
            # self.scenedb = scenedb[:-988] if split is 'train' else scenedb[-988:]
        else:
            self.scenedb = scenedb
        
    def __len__(self):
        return len(self.scenedb)

    def __getitem__(self, idx):
        entry = {}
        entry['scene_idx'] = int(idx) 
        scene = deepcopy(self.scenedb[idx])
        entry['image_idx'] = scene['scene_idx']

        ###################################################################
        ## Sentence
        ###################################################################
        group_sents = scene['scene_sentences']
        if self.split == 'train' and self.cfg.sent_group < 0:
            cid = np.random.randint(0, len(group_sents))
        else:
            cid = self.cfg.sent_group
        entry['sentence'] = ' '.join(group_sents[cid])

        ###################################################################
        ## Text inputs
        ###################################################################
        # Word embedding
        all_sents = deepcopy(group_sents[cid])
        if self.split == 'train' and self.cfg.shuffle_sents:
            shuffled_inds = np.random.permutation(range(len(all_sents)))
            shuffled_sents = [all_sents[i] for i in shuffled_inds]
        else:
            shuffled_sents = all_sents

        word_inds, word_lens = [], []
        for k in range(len(shuffled_sents)):
            s = shuffled_sents[k]
            cur_tokens = [w for w in word_tokenize(s.lower())]
            cur_tokens = further_token_process(cur_tokens)
            cur_inds   = [self.lang_vocab.word2index[w] for w in cur_tokens]
            mask_value = 0.0
            if k == len(shuffled_sents) - 1:
                mask_value = 1.0
            cur_inds, cur_mask = self.pad_sequence(
                cur_inds, self.cfg.max_input_length, self.cfg.PAD_idx, None, self.cfg.EOS_idx, mask_value)
            cur_len = np.sum(cur_mask)
            word_inds.append(cur_inds)
            word_lens.append(cur_len)
        entry['word_inds'] = np.stack(word_inds, 0).astype(np.int32)
        entry['word_lens'] = np.stack(word_lens, 0).astype(np.int32)

        ###################################################################
        ## Outputs
        ###################################################################
        out_inds, out_msks = self.scene_to_output_inds(scene)
        out_inds = out_inds.astype(np.int32)
        entry['out_inds'] = out_inds
        entry['out_msks'] = out_msks.astype(np.float32)

        ###################################################################
        ## Images 
        ###################################################################
        entry['color_path'] = scene['image_path']

        sos_img = cv2.imread(self.background_path(), cv2.IMREAD_COLOR)
        sos_img = cv2.resize(sos_img, (self.cfg.input_size[0], self.cfg.input_size[1]))
        pad_img = np.zeros_like(sos_img)
        bg_imgs = self.render_scene_as_input(scene, return_sequence=True)
        bg_imgs_list = [x for x in bg_imgs]

        entry['background'], _ = \
            self.pad_sequence(bg_imgs_list, self.cfg.max_output_length, pad_img, sos_img, None, 0.0)
        gt_fg_inds = deepcopy(out_inds[:,0]).astype(np.int32).flatten().tolist()
        gt_fg_inds = [self.cfg.SOS_idx] + gt_fg_inds
        gt_fg_inds = np.array(gt_fg_inds)
        entry['fg_inds'] = gt_fg_inds

        gt_hmaps = self.heat_maps(out_inds[:,3])
        gt_hmaps = [x for x in gt_hmaps]
        gt_hmaps = gt_hmaps[:-1]
        pad_hmap = np.ones((self.cfg.grid_size[1], self.cfg.grid_size[0]))
        gt_hmaps = [pad_hmap] + gt_hmaps
        entry['hmaps'] = np.stack(gt_hmaps, 0)

        ###################################################################
        ## Transformation
        ###################################################################
        if self.transform:
            entry = self.transform(entry)

        return entry

    def heat_maps(self, batch_where_inds):
        bsize = len(batch_where_inds)
        gw = self.cfg.grid_size[0]
        gh = self.cfg.grid_size[1]
        
        loc_maps = np.zeros((bsize, gh, gw), dtype=np.float32)
        for i in range(bsize):
            wid = batch_where_inds[i]
            pos = self.location_map.index2coord(wid)
            x = float(pos[0])/self.cfg.image_size[0]
            y = float(pos[1])/self.cfg.image_size[1]
            x = max(min(int(gw * x), gw-1), 0)
            y = max(min(int(gh * y), gh-1), 0)
            loc_maps[i, y, x] = 1.0

        return loc_maps

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
    
    def sort_segments(self, scene):
        # Sort the objects in the order they should be drawn
        # Rules: [1] "Sky" objects go first
        #        [2] Objects with larger "scale" index (e.g. far away) should be drawn first
        #        [3] Annotated order
        segments = deepcopy(scene['segments'])
        positions = np.array(scene['positions']).copy()
        scales = np.array(scene['scales']).copy()
        flips = np.array(scene['flips']).copy()

        o1 = np.ones((len(segments),))
        for i in range(len(segments)):
            if segments[i][0] == 's':
                o1[i] = 0
            # elif segments[i][0] == 't' or segments[i][0] == 'c':
            #     o1[i] = 2
        o2 = - scales.astype(np.float)
        o3 = [self.segment_vocab.word2index[w] for w in segments]
        # indices = np.lexsort((o3, o2, o1))
        indices = np.lexsort((o2, o1))
        # print(indices)
        scene['segments'] = [segments[i] for i in indices]
        scene['positions'] = positions[indices]
        scene['scales'] = scales[indices]
        scene['flips'] = flips[indices]
        return scene

    def sort_segment_vocab(self, segment_vocab):
        # Sort the objects so that the indices of the generated sequences follow some order pattern
        segments = sorted([x for x in segment_vocab.word2index.keys()], key=natural_keys)
        order = [4.0] * len(segments)
        for i in range(len(segments)):
            if segments[i] == '<pad>':
                order[i] = 0.0
            elif segments[i] == '<sos>':
                order[i] = 1.0
            elif segments[i] == '<eos>':
                order[i] = 2.0
            # "Sky" objects go first
            elif segments[i][0] == 's':
                order[i] = 3.0
            # Accessories should be drawn after persons ??
            elif segments[i][0] == 't' or segments[i][0] == 'c':
                order[i] = 5.0
        # stable sort, note that np.argsort is unstable
        indices = np.lexsort((order,))
        for i in range(len(indices)):
            segment_vocab.index2word[i] = segments[indices[i]]
            segment_vocab.word2index[segments[indices[i]]] = i
        return segment_vocab
        
    def segment_path(self, name):
        return osp.join(self.root_dir, 'Pngs', name+'.png')
    
    def background_path(self):
        return osp.join(self.root_dir, 'Pngs', 'background.png')

    def load_scene(self, db_dir):
        scenedb_path = osp.join(self.cache_dir, 'abstract_scene.json')
        if osp.exists(scenedb_path):
            scenedb = json_load(scenedb_path)
            scenedb = OrderedDict(sorted(scenedb.items()))
        else:
            scenedb = OrderedDict()
            prev_class_idx = -1
            fp = open(osp.join(db_dir, 'Scenes_10020.txt'), 'r')
            for line in fp:
                info = line.strip().split()
                if len(info) < 2:
                    # First line
                    continue
                elif len(info) == 2:
                    # start a new scene
                    class_idx = int(info[0])
                    if prev_class_idx != class_idx:
                        instance_idx = 0
                        prev_class_idx = class_idx
                    else:
                        instance_idx += 1
                        assert(instance_idx < 10)
                    scene_idx = 10 * class_idx + instance_idx
                    # scene_idx now is a string
                    scene_idx = '%09d'%scene_idx
                    scenedb[scene_idx] = {
                        'class_idx': class_idx, 
                        'instance_idx': instance_idx, 
                        'scene_idx': scene_idx,
                        'segments': [], 'positions': [], 
                        'scales': [], 'flips': []
                    }
                else:
                    segment = osp.splitext(info[0])[0]
                    pos = [int(info[3]), int(info[4])]
                    scale = int(info[5])
                    flip = int(info[6])

                    scenedb[scene_idx]['segments'].append(segment)
                    scenedb[scene_idx]['positions'].append(pos)
                    scenedb[scene_idx]['flips'].append(flip)
                    scenedb[scene_idx]['scales'].append(scale)
            fp.close()

            # Each class has 1 sentence
            class_sents = {}
            with open(osp.join(db_dir, 'Sentences_1002.txt'), 'r') as fp:
                for cnt, line in enumerate(fp):
                    class_sents[cnt] = line

            # Each scene has two groups of sentences
            # Each group contain 3 sentences
            scene_sents_1 = self.load_sentences(osp.join(db_dir, 'SimpleSentences/SimpleSentences1_10020.txt'))
            scene_sents_2 = self.load_sentences(osp.join(db_dir, 'SimpleSentences/SimpleSentences2_10020.txt'))

            for scene_idx in scenedb.keys():
                entry = scenedb[scene_idx]
                entry['class_sentence'] = class_sents[entry['class_idx']]
                entry['image_path'] = osp.join(db_dir, 'RenderedScenes', 'Scene%d_%d.png'%(entry['class_idx'], entry['instance_idx']))
                assert(osp.exists(entry['image_path']))
                entry['scene_sentences'] = []
                if scene_sents_1.get(scene_idx, None) is not None:
                    entry['scene_sentences'].append(scene_sents_1[scene_idx])
                if scene_sents_2.get(scene_idx, None) is not None:
                    entry['scene_sentences'].append(scene_sents_2[scene_idx])
                # pad so that each scene has two groups of sentences
                if len(entry['scene_sentences']) == 1:
                    pad_sent = deepcopy(entry['scene_sentences'][0])
                    entry['scene_sentences'].append(pad_sent)
                entry['scene_idx'] = scene_idx
                # entry['background_path'] = self.background_path()
            scenedb = OrderedDict(sorted(scenedb.items()))
            json_save(scenedb_path, scenedb)

        current_number = len(scenedb)
        filtered_scenedb = [v for k, v in scenedb.items() if len(v['scene_sentences']) > 0]
        scenedb = filtered_scenedb
        # print('Filtered scenes with no caption: %d --> %d'%(current_number, len(scenedb)))
        current_number = len(scenedb)
        filtered_scenedb = [v for v in scenedb if len(v['segments']) > 0]
        scenedb = filtered_scenedb
        # print('Filtered scenes with no object: %d --> %d'%(current_number, len(scenedb)))
        return scenedb

    def load_sentences(self, file_path):
        sentences = {}

        fp = open(file_path, 'r')
        for line in fp:
            info = line.strip().split()
            if len(info) < 3:
                continue
            scene_idx = int(info[0])
            # scene_idx now is a string
            scene_idx = '%09d'%scene_idx
            sent_idx = int(info[1])
            sent = ' '.join(info[2:])
            if sent_idx > 2:
                continue
            if sentences.get(scene_idx, None) is None:
                sentences[scene_idx] = []
            sentences[scene_idx].append(sent)
        fp.close()
        return sentences

    def load_segment_sizes(self):
        wh_path = osp.join(self.cache_dir, 'abstract_segment_sizes.json')
        if osp.exists(wh_path):
            whs = json_load(wh_path)
            whs = OrderedDict(sorted(whs.items()))
        else:
            whs = OrderedDict()
            for i in range(self.cfg.EOS_idx+1, len(self.segment_vocab.index2word)):
                name = self.segment_vocab.index2word[i]
                path = self.segment_path(name)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                alpha = img[:,:,-1]
                # cv2.imwrite(name+'.jpg', alpha)
                # miny, minx, maxy, maxx = regionprops(alpha)[0]['bbox']
                w = img.shape[1]; h = img.shape[0]
                minx = w; miny = h; maxx = 0; maxy = 0
                for j in range(w):
                    for k in range(h):
                        a = alpha[k, j]
                        if a > 50:
                            minx = min(minx, j); maxx = max(maxx, j)
                            miny = min(miny, k); maxy = max(maxy, k)
                whs[name] = [maxx - minx + 1, maxy - miny + 1]
            json_save(wh_path, whs)
        self.segment_sizes = whs

    def build_lang_vocab(self, scenedb):
        lang_vocab = Vocab('abstract_lang_vocab')
        lang_vocab_path = osp.join(self.cache_dir, 'abstract_lang_vocab.json')
        if osp.exists(lang_vocab_path):
            lang_vocab.load(lang_vocab_path)
        else:
            for i in range(len(scenedb)):
                group_sents = deepcopy(scenedb[i]['scene_sentences'])
                for j in range(len(group_sents)):
                    sentence = ' '.join(group_sents[j])
                    lang_vocab.addSentence(sentence)
            lang_vocab.save(lang_vocab_path)
        lang_vocab.get_glovec()
        return lang_vocab
    
    def build_segment_vocab(self, scenedb):
        segment_vocab = Vocab('abstract_segment_vocab')
        segment_vocab_path = osp.join(self.cache_dir, 'abstract_segment_vocab.json')
        if osp.exists(segment_vocab_path):
            segment_vocab.load(segment_vocab_path)
        else:
            for i in range(len(scenedb)):
                segments = deepcopy(scenedb[i]['segments'])
                for x in segments:
                    segment_vocab.addWord(x)
            
            segment_vocab = self.sort_segment_vocab(segment_vocab)
            segment_vocab.save(segment_vocab_path)  
        
        return segment_vocab

    def render_scene_as_input(self, scene, return_sequence=False, draw_boxes=False):
        surface = cairo.ImageSurface.create_from_png(self.background_path())
        ctx = cairo.Context(surface)
        assert(len(scene['segments']) > 0)
        if draw_boxes and scene.get('boxes', None) is None:
            scene = self.create_bboxes(scene)
        sequence = []
        for i in range(len(scene['segments'])):
            pos = scene['positions'][i]
            flip = scene['flips'][i]
            scale = self.cfg.scales[scene['scales'][i]]
            segment_image = cv2.imread(self.segment_path(scene['segments'][i]), cv2.IMREAD_UNCHANGED)
            segment_image = cv2.resize(segment_image, (0,0), fx=scale, fy=scale)
            H,W,_ = segment_image.shape
            if flip == 1:
                segment_image = np.flip(segment_image, axis=1).copy()
            segment_surf = cairo.ImageSurface.create_for_data(
                segment_image, cairo.FORMAT_ARGB32, W, H)
            ox = pos[0]-W/2; oy = pos[1]-H/2
            ctx.save()
            ctx.translate(ox, oy)
            ctx.set_source_surface(segment_surf)
            ctx.paint()
            ctx.restore()
            if draw_boxes:
                xyxy = scene['boxes'][i]
                paint_box(ctx, (0, 1, 0), xyxy)
            frame = surface_to_image(surface)
            frame = cv2.resize(frame, (self.cfg.input_size[1], self.cfg.input_size[0]))
            sequence.append(frame)

        if return_sequence:
            return np.stack(sequence, axis=0)
        else:
            return sequence[-1]

    def render_scene_as_output(self, scene, return_sequence=False, draw_boxes=False):
        surface = cairo.ImageSurface.create_from_png(self.background_path())
        ctx = cairo.Context(surface)
        assert(len(scene['segments']) > 0)
        if draw_boxes and scene.get('boxes', None) is None:
            scene = self.create_bboxes(scene)
        sequence = []
        for i in range(len(scene['segments'])):
            pos = scene['positions'][i]
            flip = scene['flips'][i]
            scale = self.cfg.scales[scene['scales'][i]]
            segment_image = cv2.imread(self.segment_path(scene['segments'][i]), cv2.IMREAD_UNCHANGED)
            segment_image = cv2.resize(segment_image, (0,0), fx=scale, fy=scale)
            H,W,_ = segment_image.shape
            if flip == 1:
                segment_image = np.flip(segment_image, axis=1).copy()
            segment_surf = cairo.ImageSurface.create_for_data(
                segment_image, cairo.FORMAT_ARGB32, W, H)
            ox = pos[0]-W/2; oy = pos[1]-H/2
            ctx.save()
            ctx.translate(ox, oy)
            ctx.set_source_surface(segment_surf)
            ctx.paint()
            ctx.restore()
            if draw_boxes:
                xyxy = scene['boxes'][i]
                paint_box(ctx, (0, 1, 0), xyxy)
            frame = surface_to_image(surface)
            sequence.append(frame)

        if return_sequence:
            return np.stack(sequence, axis=0)
        else:
            return sequence[-1]

    def load_split(self, split, scenedb):
        split_path = osp.join(self.cache_dir, 'abstract_' + split + '_split.txt')
        if osp.exists(split_path):
            split_inds = np.loadtxt(split_path, dtype=np.int32)
        else:
            n_scenes = len(scenedb)
            other_inds, test_inds = [], []
            for i in range(n_scenes):
                curr_sidx = int(scenedb[i]['scene_idx'])
                # Follow the split from the paper:
                # Learning the Visual Interpretation of Sentences
                if curr_sidx >= 8000 and curr_sidx < 9000:
                    test_inds.append(i)
                else:
                    other_inds.append(i)
            
            # np.random.shuffle(other_inds)
            train_inds = other_inds[:8500]
            val_inds   = other_inds[8500:]
            
            train_path = osp.join(self.cache_dir, 'abstract_train_split.txt')
            np.savetxt(train_path, sorted(train_inds), fmt='%d')
            val_path   = osp.join(self.cache_dir, 'abstract_val_split.txt')
            np.savetxt(val_path,   sorted(val_inds),   fmt='%d')
            test_path  = osp.join(self.cache_dir, 'abstract_test_split.txt')
            np.savetxt(test_path,  sorted(test_inds),  fmt='%d')

            print('train split', len(train_inds))
            print('val split',   len(val_inds))
            print('test split',  len(test_inds))


            if split == 'train':
                split_inds = train_inds
            elif split == 'val':
                split_inds = val_inds
            else:
                split_inds = test_inds
    
        split_inds = sorted(split_inds)
        split_db = [scenedb[i] for i in split_inds]
        return split_db

    ########################################################################
    ## Scene and indices
    ########################################################################
    def segment_to_triplet(self, segment_idx):
        if segment_idx < 24:
            obj_idx = segment_idx
            pose_idx = 0
            expr_idx = 0
        elif (segment_idx > 23 and segment_idx < 59):
            obj_idx = 24
            attr_idx = segment_idx - 24
            pose_idx = attr_idx // 5
            expr_idx = attr_idx % 5
        elif (segment_idx > 58 and segment_idx < 94):
            obj_idx = 25
            attr_idx = segment_idx - 59
            pose_idx = attr_idx // 5
            expr_idx = attr_idx % 5
        else:
            obj_idx = segment_idx - 68
            pose_idx = 0
            expr_idx = 0
        return [obj_idx, pose_idx, expr_idx]
    
    def triplet_to_segment(self, tr):
        obj_idx, pose_idx, expr_idx = tr
        if obj_idx < 24:
            segment_idx = obj_idx
        elif obj_idx == 24:
            segment_idx = 24 + 5 * pose_idx + expr_idx
        elif obj_idx == 25:
            segment_idx = 59 + 5 * pose_idx + expr_idx
        else:
            segment_idx = obj_idx + 68
        return segment_idx

    def scene_to_output_inds(self, scene):
        clp_inds = [self.segment_vocab.word2index[w] for w in scene['segments']]
        ope_inds = [self.segment_to_triplet(x) for x in clp_inds]
        coords   = np.array([x.copy() for x in scene['positions']])
        pos_inds = self.location_map.coords2indices(coords).tolist()
        sca_inds = [x for x in scene['scales']]
        flp_inds = [x for x in scene['flips']]
        ope_inds = np.array(ope_inds)
        pos_inds = np.array(pos_inds).reshape((-1, 1))
        sca_inds = np.array(sca_inds).reshape((-1, 1))
        flp_inds = np.array(flp_inds).reshape((-1, 1))
        out_inds = np.concatenate([ope_inds, pos_inds, sca_inds, flp_inds], -1)
        out_inds, out_msks = self.pad_output_inds(out_inds)
        return out_inds, out_msks
        
    def output_inds_to_scene(self, inds):
        scene = {}
        n_objs = 0
        for i in range(len(inds)):
            if inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1
        
        val_inds = deepcopy(inds[:n_objs, :]).reshape((-1, 6)).astype(np.int)
        ope_inds = val_inds[:, :3]
        pos_inds = val_inds[:, 3]
        sca_inds = val_inds[:, 4]
        flp_inds = val_inds[:, 5]

        clp_inds = [self.triplet_to_segment(ope_inds[i].tolist()) for i in range(len(ope_inds))]

        scene['segments'] = [self.segment_vocab.index2word[i] for i in clp_inds]
        scene['positions'] = self.location_map.indices2coords(pos_inds)
        scene['scales'] = sca_inds
        scene['flips'] = flp_inds

        return scene 
    
    def pad_output_inds(self, inds):
        n_objs = 0
        for i in range(len(inds)):
            if inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1

        out_inds = deepcopy(inds[:n_objs, :]).reshape((-1, 6)).astype(np.int)
        # shorten if necessary
        out_inds = out_inds[:self.cfg.max_output_length]
        n_out = len(out_inds)
        n_pad = self.cfg.max_output_length - n_out
        out_msks = np.zeros((n_out+1, 6), dtype=float)
        out_msks[:n_out, :] = 1.0
        out_msks[n_out, 0] = 1.0
        for i in range(out_inds.shape[0]):
            # Pose and Expression are ignored for non-human objects
            if (out_inds[i, 0] != 24) and (out_inds[i, 0] != 25):
                out_msks[i, 1:3] = 0 
        eos_inds = np.zeros((1, 6))
        eos_inds[0, 0] = self.cfg.EOS_idx
        out_inds = np.concatenate([out_inds, eos_inds], 0)
        if n_pad > 0:
            pad_inds = np.ones((n_pad, 6)) * self.cfg.PAD_idx
            pad_msks = np.zeros((n_pad, 6))
            out_inds = np.concatenate([out_inds, pad_inds], 0)
            out_msks = np.concatenate([out_msks, pad_msks], 0)
        return out_inds, out_msks

    def create_bboxes(self, scene):
        ori_whs = np.array([self.segment_sizes[x] for x in scene['segments']])
        scales  = np.array([self.cfg.scales[x] for x in scene['scales']])
        radius  = 0.5 * ori_whs * scales[:, None]
        boxes   = []
        for i in range(len(scene['positions'])):
            xy = scene['positions'][i]
            wh = radius[i]
            bb = np.array([xy[0]-wh[0], xy[1]-wh[1], xy[0]+wh[0], xy[1]+wh[1]])
            boxes.append(bb)
        scene['boxes'] = boxes
        return scene

    ########################################################################
    ## Evaluation
    ########################################################################
    def json_to_scene(self, json_path):
        meta = json_load(json_path)
        meta_objs = meta['objects']
        segments, scales, flips, positions, categories, attributes = [],[],[],[],[],[]
        for i in range(len(meta_objs)):
            entry = meta_objs[i]
            png = entry['png']
            name = osp.splitext(png)[0]
            segments.append(name)
            flips.append(int(entry['flip']))
            xy = np.array([float(entry['x']), float(entry['y'])])
            positions.append(xy)
            scales.append(float(entry['z']))
            categories.append(float(entry['category']))
            attributes.append(float(entry['attribute']))

        scales = np.array(scales).astype(np.int32)
        flips = np.array(flips).astype(np.int32)
        positions = np.stack(positions, 0)

        o1 = np.ones((len(segments),))
        for i in range(len(segments)):
            if segments[i][0] == 's':
                o1[i] = 0
        o2 = - scales.astype(np.float)
        o3 = np.array(categories)
        o4 = np.array(attributes)
        indices = np.lexsort((o4, o3, o2, o1))
        scene = {}
        scene['segments'] = [segments[i] for i in indices]
        scene['positions'] = positions[indices]
        scene['scales'] = scales[indices]
        scene['flips'] = flips[indices]

        return scene

    ########################################################################
    ## Demo
    ########################################################################
    def encode_sentences(self, sentences):
        word_inds, word_lens = [], []
        for k in range(len(sentences)):
            s = sentences[k]
            cur_tokens = [w for w in word_tokenize(s.lower())]
            cur_tokens = further_token_process(cur_tokens)
            cur_inds   = [self.lang_vocab.word2index[w] for w in cur_tokens]
            mask_value = 0.0
            if k == len(sentences) - 1:
                mask_value = 1.0
            cur_inds, cur_mask = self.pad_sequence(
                cur_inds, self.cfg.max_input_length, self.cfg.PAD_idx, None, self.cfg.EOS_idx, mask_value)
            cur_len = np.sum(cur_mask)
            word_inds.append(cur_inds)
            word_lens.append(cur_len)
        word_inds = np.stack(word_inds, 0).astype(np.int32)
        word_lens = np.stack(word_lens, 0).astype(np.int32)
        return word_inds, word_lens

