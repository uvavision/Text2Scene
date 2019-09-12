#!/usr/bin/env python

import numpy as np
import cv2, math
import PIL, cairo
from copy import deepcopy
from abstract_utils import *
from modules.abstract_evaluator import *

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
            scene['surface']  = cairo.ImageSurface.create_from_png(self.db.background_path())
            scene['out_inds'] = []
            self.scenes.append(scene)
            frame = surface_to_image(scene['surface'])
            frame = cv2.resize(frame, (self.cfg.input_size[0], self.cfg.input_size[1]))
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        return img_to_tensor(frames)
        
    def batch_render_to_pytorch(self, batch_pred_inds):
        assert(len(batch_pred_inds) == self.batch_size)
        outputs = []
        for i in range(self.batch_size):
            frame = self.update(self.scenes[i], batch_pred_inds[i])
            outputs.append(frame)
        outputs = np.stack(outputs, 0)
        return img_to_tensor(outputs)

    def batch_location_maps(self, batch_where_inds):
        return self.db.heat_maps(batch_where_inds)

    def update(self, scene, input_inds):
        ##############################################################
        # Update the scene and the last instance of the scene
        ##############################################################
        pred_inds = deepcopy(input_inds).flatten()
        scene['out_inds'].append(pred_inds)
        ref_inds = np.stack(scene['out_inds'], 0)
        # ref_scene = self.db.output_inds_to_scene(ref_inds)

        ##############################################################
        # Foreground segment
        ##############################################################
        if pred_inds[0] <= self.cfg.EOS_idx:
            frame = surface_to_image(scene['surface'])
        else:
            cid = self.db.triplet_to_segment(pred_inds[:3].tolist())
            segment_name = self.db.segment_vocab.index2word[cid]
            segment_image = cv2.imread(self.db.segment_path(segment_name), cv2.IMREAD_UNCHANGED)
            coord = self.db.location_map.index2coord(pred_inds[3])
        
            ##############################################################
            # Scale
            ##############################################################
            scale = self.cfg.scales[pred_inds[4]]
            segment_image = cv2.resize(segment_image, (0,0), fx=scale, fy=scale)

            ##############################################################
            # Flip or not
            ##############################################################
            if pred_inds[5] == 1:
                segment_image = np.flip(segment_image, axis=1).copy()
            
            ##############################################################
            # Position
            ##############################################################
            H, W, _ = segment_image.shape
            ox = coord[0]-W/2; oy = coord[1]-H/2

            ##############################################################
            # Draw
            ##############################################################
            segment_surf = cairo.ImageSurface.create_for_data(
                segment_image, cairo.FORMAT_ARGB32, W, H)

            surface = scene['surface']
            ctx = cairo.Context(surface)
            ctx.save()
            ctx.translate(ox, oy)
            ctx.set_source_surface(segment_surf)
            ctx.paint()
            ctx.restore()
            scene['surface'] = surface

            ##############################################################
            # Get output
            ##############################################################
            frame = surface_to_image(surface)
        
        frame = cv2.resize(frame, (self.cfg.input_size[0], self.cfg.input_size[1]))
        return frame

    def batch_redraw(self, return_sequence=False):
        outputs = []
        for i in range(len(self.scenes)):
            pred_inds = np.array(deepcopy(self.scenes[i]['out_inds']))
            ref_scene = self.db.output_inds_to_scene(pred_inds)
            # print('pred_inds', pred_inds)
            frames = self.db.render_scene_as_output(ref_scene, return_sequence=return_sequence)
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
        new_scene = {}
        new_scene['out_inds'] = deepcopy(scene['out_inds'])

        surface = scene['surface']
        pimg = PIL.Image.frombuffer("RGBA", 
            (surface.get_width(), surface.get_height()),
            surface.get_data(),
            "raw", "RGBA", 0, 1)
        frame = np.array(pimg)
        new_scene['surface'] = \
            cairo.ImageSurface.create_for_data(
                frame.copy(), 
                surface.get_format(), 
                surface.get_width(), 
                surface.get_height())
        
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

