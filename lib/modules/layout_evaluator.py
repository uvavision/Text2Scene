#!/usr/bin/env python

import numpy as np
import cv2, math
import PIL, cairo
from copy import deepcopy
from layout_utils import *

import torch

###############################################################
# Evaluation
###############################################################

class scene_graph(object):
    def __init__(self, db, scene, scene_inds, from_indices):
        self.db = db
        self.cfg = db.cfg 
        if from_indices:
            self.get_unigrams_from_indices(scene_inds)
        else:
            self.get_unigrams_from_scene(scene)
        self.get_bigrams()

    def get_unigrams_from_indices(self, scene_inds):
        ref_inds = deepcopy(scene_inds)
        # collect only "real" objects
        n_objs = 0
        for i in range(len(ref_inds)):
            if ref_inds[i, 0] <= self.cfg.EOS_idx:
                break
            n_objs += 1
        if n_objs == 0:
            self.unigrams = []
            return self.unigrams
        
        ref_scene = self.db.output_inds_to_scene(ref_inds)
        ref_xywhs = ref_scene['boxes']
        ref_xyxys = normalized_xywhs_to_xyxys(ref_xywhs)
        ref_srs   = self.db.trans_map.whs2coords(ref_xywhs[:, 2:])
        
        unigrams = []
        for i in range(n_objs):
            oid = ref_inds[i, 0]
            normsr = ref_srs[i]
            normpos = ref_xywhs[i, :2]
            normbox = ref_xyxys[i]
            # object idx, scale, ratio, box, coord
            univec = np.array([oid, *normsr, *normbox, *normpos])
            unigrams.append(univec.astype(np.float32))
        
        obj_inds = [x[0] for x in unigrams]
        x_values = [x[-2] for x in unigrams]
        y_values = [x[-1] for x in unigrams]
        sorted_inds = np.lexsort((y_values, x_values, obj_inds))
        self.unigrams = [unigrams[i] for i in sorted_inds]
        
        # print('sorted unigrams', [x[0] for x in self.unigrams])
        return self.unigrams
        
    def get_unigrams_from_scene(self, scene):
        # scene should have fields: 'clses', 'boxes'

        ref_scene = deepcopy(scene)
        ref_inds = ref_scene['clses']

        # collect only "real" objects
        n_objs = 0
        for i in range(len(ref_inds)):
            if ref_inds[i] <= self.cfg.EOS_idx:
                break
            n_objs += 1
        if n_objs == 0:
            self.unigrams = []
            return self.unigrams
        
        ref_xywhs = ref_scene['boxes']
        ref_xyxys = normalized_xywhs_to_xyxys(ref_xywhs)
        ref_srs   = self.db.trans_map.whs2coords(ref_xywhs[:, 2:])

        unigrams = []
        for i in range(n_objs):
            oid = ref_inds[i]
            normsr = ref_srs[i]
            normpos = ref_xywhs[i, :2]
            normbox = ref_xyxys[i]
            # object idx, scale, ratio, box, coord
            univec  = np.array([oid, *normsr, *normbox, *normpos])
            unigrams.append(univec.astype(np.float32))
        
        obj_inds = [x[0] for x in unigrams]
        x_values = [x[-2] for x in unigrams]
        y_values = [x[-1] for x in unigrams]
        sorted_inds = np.lexsort((y_values, x_values, obj_inds))
        self.unigrams = [unigrams[i] for i in sorted_inds]
        
        # print('sorted unigrams', [x[0] for x in self.unigrams])
        return self.unigrams

    def get_bigrams(self):
        n_objs = len(self.unigrams)
        if n_objs < 2:
            self.bigrams = []
        else:
            bigrams = []
            for i in range(n_objs-1):
                src_xy = self.unigrams[i][-2:]
                src_bb = self.unigrams[i][-6:-2]
                src_id = self.unigrams[i][0]

                for j in range(i+1, n_objs):
                    tgt_xy = self.unigrams[j][-2:]
                    tgt_bb = self.unigrams[j][-6:-2]
                    tgt_id = self.unigrams[j][0]

                    ovr = bb_iou(src_bb, tgt_bb)
                    if ovr == 0:
                        continue

                    diff = src_xy - tgt_xy
                    x = diff[0]; y = diff[1]
                    l = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + self.cfg.eps)
                    v = diff / (l+self.cfg.eps)
                    t = 0.5 * (math.atan2(v[1], v[0]) + math.pi)/math.pi
                    # obj idx 1, obj idx 2, obj coord 1, obj coord 2, offset xy, offset polar
                    bivec = np.array([src_id, tgt_id, src_xy[0], src_xy[1], tgt_xy[0], tgt_xy[1], x, y, l, t])
                    bigrams.append(bivec.astype(np.float32))
            
            #TODO should sort in some way
            obj1_inds = [x[0] for x in bigrams]
            obj2_inds = [x[1] for x in bigrams]
            l_values = [x[-2] for x in bigrams]
            t_values = [x[-1] for x in bigrams]
            sorted_inds = np.lexsort((t_values, l_values, obj2_inds, obj1_inds))
            self.bigrams = [bigrams[i] for i in sorted_inds]
            # print('sorted bigrams', [x[0:2] for x in self.bigrams])
        return self.bigrams


class eval_info(object):
    def __init__(self, config, scores):
        self.cfg = config
        self.scores = deepcopy(scores)

    def unigram_F3(self):
        P = self.unigram_P(); R = self.unigram_R()
        eps = self.cfg.eps; wei = self.cfg.recall_weight
        return (1.0 + wei) * P * R / (R + wei * P + eps)

    def bigram_F3(self):
        P = self.bigram_P(); R = self.bigram_R()
        eps = self.cfg.eps; wei = self.cfg.recall_weight
        return (1.0 + wei) * P * R / (R + wei * P + eps)
    
    def unigram_reward(self):
        rew = self.unigram_F3() * (1.0 + 0.5 * (self.scale() + self.ratio()))
        return rew

    def bigram_reward(self):
        rew = self.bigram_F3() * (1.0 + self.bigram_coord())
        return rew

    def reward(self):
        return self.unigram_reward() + self.bigram_reward()

    def mean_bigram_P(self):
        recall = self.bigram_R()
        mask = (recall >= 0.0).astype(np.float32)
        precision = self.bigram_P()
        mean_precision = np.sum(precision * mask)/(np.sum(mask) + self.cfg.eps)
        return mean_precision

    def mean_bigram_R(self):
        recall = self.bigram_R()
        mask = (recall >= 0.0).astype(np.float32)
        mean_recall = np.sum(recall * mask)/(np.sum(mask) + self.cfg.eps)
        return mean_recall
    
    def mean_bigram_coord(self):
        recall = self.bigram_R()
        mask = (recall >= 0.0).astype(np.float32)
        coords = self.bigram_coord()
        mean_coords = np.sum(coords * mask)/(np.sum(mask) + self.cfg.eps)
        return mean_coords

    ################################################################
    # Property
    ################################################################
    def unigram_P(self):
        return self.scores[:, 0]

    def unigram_R(self):
        return self.scores[:, 1]

    def bigram_P(self):
        return self.scores[:, 2]
    
    def bigram_R(self):
        return self.scores[:, 3]

    def scale(self):
        return self.scores[:, 4]

    def ratio(self):
        return self.scores[:, 5]

    def unigram_coord(self):
        return self.scores[:, 6]
    
    def bigram_coord(self):
        return self.scores[:, 7]
    
    
class evaluator(object):
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg 

    # def evaluate_scene(self, pred_scene, gt_scene):
    #     self.gt_graph   = scene_graph(self.db, gt_scene)
    #     self.pred_graph = scene_graph(self.db, pred_scene)
    #     return self.evaluate_graph(self.pred_graph, self.gt_graph)

    def evaluate_graph(self, pred_graph, gt_graph):
        assert(len(gt_graph.unigrams) > 0)

        if len(pred_graph.unigrams) == 0:
            scores = np.zeros((8, ), dtype=np.float32)
            # # indicate this entry is not used for precision calculation
            # scores[0] = -1.0; scores[2] = -1.0 
            return scores
        
        # P, R, scale, ratio, coord
        unigram_comps = self.unigram_reward(pred_graph.unigrams, gt_graph.unigrams)

        # P, R, coord
        if len(gt_graph.bigrams) == 0:
            # indicate this entry is not used for recall calculation
            bigram_comps = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            bigram_comps = self.bigram_reward(pred_graph.bigrams, gt_graph.bigrams)

        scores = unigram_comps[:2].tolist() + bigram_comps[:2].tolist() + unigram_comps[2:].tolist() + [bigram_comps[-1]]
        scores = np.array(scores, dtype=np.float32)
        return scores
    
    def unigram_reward(self, pred_unigrams, gt_unigrams):
        self.common_pred_unigrams, self.common_gt_unigrams, self.unigram_dists = \
            self.find_common_unigrams(pred_unigrams, gt_unigrams)
        n_common = len(self.common_pred_unigrams)

        if n_common == 0:
            return np.zeros((5,), dtype=np.float32)

        n_pred = len(pred_unigrams)
        n_gt   = len(gt_unigrams)

        precision = float(n_common)/n_pred
        recall    = float(n_common)/n_gt

        scale_s = batch_gaussian1d(self.common_pred_unigrams[:, 1], self.common_gt_unigrams[:, 1], self.cfg.sigmas[2])
        scale_s = np.mean(scale_s)

        ratio_s = batch_gaussian1d(self.common_pred_unigrams[:, 2], self.common_gt_unigrams[:, 2], 0.2)
        ratio_s = np.mean(ratio_s)

        coord_s = np.mean(self.unigram_dists)

        return np.array([precision, recall, scale_s, ratio_s, coord_s], dtype=np.float32)
    
    def bigram_reward(self, pred_bigrams, gt_bigrams):
        self.common_pred_bigrams, self.common_gt_bigrams, self.bigram_dists = \
            self.find_common_bigrams(pred_bigrams, gt_bigrams)
        n_common = len(self.common_pred_bigrams)
        if n_common == 0:
            return np.zeros((3,), dtype=np.float32)
        
        n_pred = len(pred_bigrams)
        n_gt   = len(gt_bigrams)

        precision = float(n_common)/n_pred
        recall    = float(n_common)/n_gt
        coord_s   = np.mean(self.bigram_dists)
        return np.array([precision, recall, coord_s], dtype=np.float32)

    def find_common_unigrams(self, pred_unigrams, gt_unigrams):
        common_pred_unigrams, common_gt_unigrams, unigram_gaussians = [], [], []

        n_gt   = len(gt_unigrams)
        msk_gt = np.zeros((n_gt, ))

        for i in range(len(pred_unigrams)):
            pred_entry = pred_unigrams[i]
            gt_entry = None; gt_idx = -1; gaussian = 0.0

            for j in range(len(gt_unigrams)):
                candidate = gt_unigrams[j]
                if (candidate[0] != pred_entry[0]) or (msk_gt[j] > 0):
                    # obj ids don't match or the entry is not available any more
                    continue
                if gt_entry is None:
                    gt_entry = candidate
                    gt_idx = j
                    gaussian = self.unigram_gaussian(pred_entry, candidate)
                    continue
                curr_gaussian = self.unigram_gaussian(pred_entry, candidate)
                if curr_gaussian > gaussian:
                    gaussian = curr_gaussian
                    gt_entry = candidate
                    gt_idx = j
            
            if gt_entry is not None:
                common_pred_unigrams.append(pred_entry)
                common_gt_unigrams.append(gt_entry)
                unigram_gaussians.append(gaussian)
                msk_gt[gt_idx] = 1

        if len(common_pred_unigrams) > 0:
            common_pred_unigrams = np.stack(common_pred_unigrams, 0)
            common_gt_unigrams   = np.stack(common_gt_unigrams,   0)

        return common_pred_unigrams, common_gt_unigrams, unigram_gaussians

    def find_common_bigrams(self, pred_bigrams, gt_bigrams):
        common_pred_bigrams, common_gt_bigrams, distances = [], [], []
        n_gt = len(gt_bigrams)
        assert(n_gt > 0)
        msk_gt = np.zeros((n_gt, ))

        for i in range(len(pred_bigrams)):
            pred_entry = pred_bigrams[i]
            gt_entry = None; gaussian = 0.0; gt_idx = -1

            for j in range(len(gt_bigrams)):
                candidate = gt_bigrams[j]
                if (candidate[0] != pred_entry[0]) or (candidate[1] != pred_entry[1]) or (msk_gt[j] > 0): 
                    # obj ids don't match
                    continue
                
                if gt_entry is None:
                    gt_entry = candidate
                    gaussian = self.bigram_gaussian(pred_entry, candidate)
                    gt_idx = j
                    continue
                
                curr_gaussian = self.bigram_gaussian(pred_entry, candidate)
                if curr_gaussian > gaussian:
                    gaussian = curr_gaussian
                    gt_entry = candidate
                    gt_idx = j

            if gt_entry is not None:
                common_pred_bigrams.append(pred_entry)
                common_gt_bigrams.append(gt_entry)
                distances.append(gaussian)
                msk_gt[gt_idx] = 1

        if len(common_pred_bigrams) > 0:
            common_pred_bigrams = np.stack(common_pred_bigrams, 0)
            common_gt_bigrams = np.stack(common_gt_bigrams, 0)

        return common_pred_bigrams, common_gt_bigrams, distances
    
    def unigram_gaussian(self, A, B):
        return gaussian2d(A[-2:], B[-2:], self.cfg.sigmas[:2])

    def bigram_gaussian(self, A, B):
        if self.cfg.rel_mode == 0:
            return gaussian2d(A[-4:-2], B[-4:-2], self.cfg.sigmas[:2])
        else:
            sigmas = self.cfg.sigmas[2:]

            A_l = A[-2]; B_l = B[-2]
            A_t = A[-1]; B_t = B[-1]
            v_l = (A_l - B_l)/float(sigmas[0])
            max_t = np.maximum(A_t, B_t)
            min_t = np.minimum(A_t, B_t)
            v_t = np.minimum(max_t - min_t, abs(min_t + 1.0 - max_t))/float(sigmas[1])

            d = math.exp(-0.5 * (v_l * v_l + v_t * v_t))
            return d
    