#!/usr/bin/env python

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from nntable import AllCategoriesTables

from composites_utils import *


class simulator(object):
    def __init__(self, db, batch_size=None, nn_table=None):
        self.db = db
        self.cfg = db.cfg
        self.batch_size = batch_size if batch_size is not None else self.cfg.batch_size
        if nn_table is None:
            self.nn_table = AllCategoriesTables(db)
            self.nn_table.build_nntables_for_all_categories()
        else:
            self.nn_table = nn_table

    def reset(self):
        self.scenes = []
        frames = []
        if self.cfg.use_color_volume:
            channel_dim = 3 * self.cfg.output_vocab_size
        else:
            channel_dim = 4 + self.cfg.output_vocab_size
        for i in range(self.batch_size):
            scene = {}
            scene['out_inds'] = []
            scene['out_vecs'] = []
            scene['out_patches'] = []
            frame = np.zeros(
                (   self.cfg.input_image_size[1],
                    self.cfg.input_image_size[0],
                    channel_dim
                )
            )
            scene['last_frame'] = frame
            scene['last_label'] = np.zeros(
                (   self.cfg.input_image_size[1],
                    self.cfg.input_image_size[0]
                ), dtype=np.int32
            )
            scene['last_mask'] = np.zeros(
                (   self.cfg.input_image_size[1],
                    self.cfg.input_image_size[0]
                ), dtype=np.float32
            )
            self.scenes.append(scene)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames)

    def batch_render_to_pytorch(self, out_inds, out_vecs):
        assert(len(out_inds) == self.batch_size)
        outputs = []
        for i in range(self.batch_size):
            frame = self.update_scene(self.scenes[i],
                {'out_inds': out_inds[i], 'out_vec': out_vecs[i]})
            outputs.append(frame)
        outputs = np.stack(outputs, 0)
        return torch.from_numpy(outputs)

    def batch_redraw(self, return_sequence=False):
        out_frames, out_noises, out_masks, out_labels, out_scenes = [], [], [], [], []
        for i in range(len(self.scenes)):
            predicted_scene = self.db.prediction_outputs_to_scene(self.scenes[i], self.nn_table)
            predicted_scene['patches'] = self.scenes[i]['out_patches']
            frames, noises, masks, labels = self.render_predictions_as_output(predicted_scene, return_sequence)
            if not return_sequence:
                frames = frames[None, ...]
                noises = noises[None, ...]
                masks  = masks[None, ...]
                labels = labels[None, ...]
            out_frames.append(frames)
            out_noises.append(noises)
            out_masks.append(masks)
            out_labels.append(labels)
            out_scenes.append(predicted_scene)
        return out_frames, out_noises, out_masks, out_labels, out_scenes

    def render_predictions_as_output(self, scene, return_sequence):
        width  = scene['width']
        height = scene['height']
        clses  = scene['clses']
        boxes  = scene['boxes']
        patches = scene['patches']

        if self.cfg.use_color_volume:
            channel_dim = 3 * self.cfg.output_vocab_size
        else:
            channel_dim = 4 + self.cfg.output_vocab_size

        frame = np.zeros((height, width, channel_dim))
        noise = np.zeros((height, width, channel_dim))
        label = np.zeros((height, width), dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.float32)

        out_frames, out_noises, out_labels, out_masks = [], [], [], []
        for i in range(len(clses)):
            cls_ind = clses[i]
            xywh = boxes[i]
            patch = patches[i]
            xyxy = xywh_to_xyxy(xywh, width, height)
            if self.cfg.use_color_volume:
                frame[:,:,3*cls_ind:3*(cls_ind+1)], mask, _, label, noise[:,:,3*cls_ind:3*(cls_ind+1)] = \
                    patch_compose_and_erose(frame[:,:,3*cls_ind:3*(cls_ind+1)], mask, label, \
                        xyxy, patch, self.db, noise[:,:,3*cls_ind:3*(cls_ind+1)])
            else:
                frame[:,:,-3:], mask, _, label, noise[:,:,-3:] = \
                    patch_compose_and_erose(frame[:,:,-3:], mask, label, xyxy, patch, self.db, noise[:,:,-3:])
                frame[:,:,-4] = np.maximum(mask*255, frame[:,:,-4])
                frame[:,:,cls_ind] = np.maximum(mask*255, frame[:,:,cls_ind])
            out_frames.append(frame.copy())
            out_noises.append(noise.copy())
            out_labels.append(label.copy())
            out_masks.append(mask.copy())

        if len(clses) == 0:
            out_frames.append(frame.copy())
            out_noises.append(noise.copy())
            out_labels.append(label.copy())
            out_masks.append(mask.copy())

        if return_sequence:
            return np.stack(out_frames, 0), np.stack(out_noises, 0), np.stack(out_masks, 0), np.stack(out_labels, 0)
        else:
            return out_frames[-1], out_noises[-1], out_masks[-1], out_labels[-1]

    def update_scene(self, scene, step_prediction):
        ##############################################################
        # Update the scene and the last instance of the scene
        ##############################################################
        out_inds = step_prediction['out_inds'].flatten()
        out_vec  = step_prediction['out_vec'].flatten()
        scene['out_inds'].append(out_inds)
        scene['out_vecs'].append(out_vec)
        scene['last_frame'], scene['last_mask'], scene['last_label'], current_patch = \
            self.update_frame(scene['last_frame'], scene['last_mask'], scene['last_label'], out_inds, out_vec)
        scene['out_patches'].append(current_patch)
        return scene['last_frame']

    def update_frame(self, input_frame, input_mask, input_label, input_inds, input_vec):
        if input_inds[0] <= self.cfg.EOS_idx:
            return input_frame, input_mask, input_label, None
        w = input_frame.shape[-2]
        h = input_frame.shape[-3]
        cls_ind = input_inds[0]
        xywh = self.db.index2box(input_inds[1:])
        xywh = xywh * np.array([w, h, w, h])
        xyxy = xywh_to_xyxy(xywh, w, h)
        patch = self.nn_table.retrieve(cls_ind, input_vec)[0]
        # print(patch)
        # print(patch['name'])

        # update the frame
        if self.cfg.use_color_volume:
            input_frame[:,:,3*cls_ind:3*(cls_ind+1)], input_mask, _, input_label, _ = \
                patch_compose_and_erose(input_frame[:,:,3*cls_ind:3*(cls_ind+1)], input_mask, input_label, xyxy, patch, self.db)
        else:
            input_frame[:,:,-3:], input_mask, _, input_label, _ = \
                patch_compose_and_erose(input_frame[:,:,-3:], input_mask, input_label, xyxy, patch, self.db)
            input_frame[:,:,-4] = np.maximum(255*input_mask, input_frame[:,:,-4])
            input_frame[:,:,cls_ind] = np.maximum(255*input_mask, input_frame[:,:,cls_ind])
        return input_frame, input_mask, input_label, patch
