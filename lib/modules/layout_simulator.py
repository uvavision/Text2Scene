#!/usr/bin/env python

import numpy as np
import cv2, math
import PIL, cairo
from copy import deepcopy
import matplotlib.pyplot as plt
from layout_utils import *
from modules.layout_evaluator import *

import torch


class simulator(object):
    def __init__(self, imdb, batch_size=None):
        self.db = imdb
        self.cfg = imdb.cfg
        self.batch_size = batch_size if batch_size is not None else self.cfg.batch_size
        self.eval = evaluator(imdb)

    def reset(self):
        self.scenes = []
        frames = []
        for i in range(self.batch_size):
            scene = {}
            scene['out_inds'] = []
            frame = np.zeros(
                (self.cfg.input_size[1], self.cfg.input_size[0], self.cfg.output_cls_size), 
                dtype=np.float32)
            scene['curr_vol'] = frame
            self.scenes.append(scene)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        return vol_to_tensor(frames)
        
    def batch_render_to_pytorch(self, batch_pred_inds):
        assert(len(batch_pred_inds) == self.batch_size)
        outputs = []
        for i in range(self.batch_size):
            frame = self.update(self.scenes[i], batch_pred_inds[i])
            outputs.append(frame)
        outputs = np.stack(outputs, 0)
        return vol_to_tensor(outputs)
        
    def update(self, scene, input_inds):
        ##############################################################
        # Update the scene and the last instance of the scene
        ##############################################################
        pred_inds = deepcopy(input_inds).flatten()
        # print('pred_inds', pred_inds)
        scene['out_inds'].append(pred_inds)
        vol = scene['curr_vol']
        scene['curr_vol'] = self.db.update_vol(vol, pred_inds)
        return deepcopy(scene['curr_vol'])

    def batch_redraw(self, return_sequence=False):
        outputs = []
        for i in range(len(self.scenes)):
            frames = self.db.render_indices_as_output(self.scenes[i], return_sequence)
            if not return_sequence:
                frames = frames[None, ...]
            outputs.append(frames)
        return outputs

    ###############################################################
    # Evaluation
    ###############################################################
    def evaluate_indices(self, scene, gt_scene_inds):
        pred_inds  = deepcopy(scene['out_inds'])
        pred_inds  = np.stack(pred_inds, 0)
        pred_graph = scene_graph(self.db, None, pred_inds,     True)
        gt_graph   = scene_graph(self.db, None, gt_scene_inds, True)
        return self.eval.evaluate_graph(pred_graph, gt_graph)

    def evaluate_scene(self, scene, gt_scene):
        pred_inds = deepcopy(scene['out_inds'])
        pred_inds = np.stack(pred_inds, 0)
        pred_scene = self.db.output_inds_to_scene(pred_inds)
        pred_graph = scene_graph(self.db, pred_scene, None, False)
        gt_graph   = scene_graph(self.db, gt_scene,   None, False)
        return self.eval.evaluate_graph(pred_graph, gt_graph)

    def batch_evaluation(self, batch_gt_scene):
        infos = []
        for i in range(len(self.scenes)):
            info = self.evaluate_scene(self.scenes[i], batch_gt_scene[i])
            infos.append(info)
        return infos

    def beam_evaluation(self, gt_scene):
        infos = []
        for i in range(len(self.scenes)):
            info = self.evaluate_scene(self.scenes[i], gt_scene)
            infos.append(info)
        return infos

    def copy_scene(self, scene):
        ###########################################################
        # Deep copy a scene
        ###########################################################
        new_scene = deepcopy(scene)
        return new_scene

    def select(self, indices):
        ###########################################################
        # Select scenes given the indices, used for beam search
        ###########################################################
        new_scenes = []
        for x in indices:
            Y = self.copy_scene(self.scenes[x])
            new_scenes.append(Y)
        self.scenes = new_scenes
        self.batch_size = len(self.scenes)

    def get_batch_inds_and_masks(self):
        out_inds, out_msks = [], []
        for i in range(len(self.scenes)):
            curr_out_inds = np.stack(self.scenes[i]['out_inds'], 0)
            curr_scene = self.db.output_inds_to_scene(curr_out_inds)
            curr_out_inds, curr_out_msks = \
                self.db.scene_to_output_inds(curr_scene)
            out_inds.append(curr_out_inds)
            out_msks.append(curr_out_msks)
        out_inds = np.stack(out_inds, 0)
        out_msks = np.stack(out_msks, 0)
        return out_inds.astype(np.int32), out_msks.astype(np.float32)

