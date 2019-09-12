#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np

import torch
import torch.nn as nn

from modules.composites_encoder import TextEncoder, VolumeEncoder, ShapeEncoder
from modules.composites_decoder import WhatDecoder, WhereDecoder
from modules.composites_simulator import simulator

from torch.distributions.categorical import Categorical


class PuzzleModel(nn.Module):
    def __init__(self, db):
        super(PuzzleModel, self).__init__()
        self.db  = db
        self.cfg = db.cfg
        self.simulator = simulator
        self.image_encoder = VolumeEncoder(self.cfg)
        self.text_encoder  = TextEncoder(self.db)
        self.what_decoder  = WhatDecoder(self.cfg)
        self.where_decoder = WhereDecoder(self.cfg)
        self.shape_encoder = ShapeEncoder(self.cfg)

    def forward(self, inputs):
        input_inds, input_lens, input_bg_imgs, input_fg_onehots, \
        positive_patches, negative_patches, \
        positive_resnets, negative_resnets = inputs
        inf_outs, env  = self.teacher_forcing(input_inds, input_lens, input_bg_imgs, input_fg_onehots)
        positive_feats = self.shape_encoder(positive_patches, positive_resnets)
        negative_feats = self.shape_encoder(negative_patches, negative_resnets)
        return inf_outs, env, positive_feats, negative_feats

    def teacher_forcing(self, input_inds, input_lens, input_bg_imgs, input_fg_onehots):
        bgfs = self.image_encoder(input_bg_imgs)
        text_states = self.text_encoder(input_inds, input_lens)
        enc_embs, enc_rfts, enc_msks, enc_hids = text_states

        prev_states = (bgfs, input_fg_onehots[:,:-1], None)
        what_states = self.what_decoder(prev_states, text_states)
        obj_logits, rnn_outs, nxt_hids, prev_bgfs, what_ctx, what_wei = what_states

        where_inputs = (rnn_outs, input_fg_onehots[:,1:], prev_bgfs, what_ctx)
        where_states = self.where_decoder(where_inputs, text_states)
        coord_logits, attri_logits, patch_vectors, where_ctx, where_wei = where_states

        inf_outs = (obj_logits, coord_logits, attri_logits, patch_vectors, enc_msks, what_wei, where_wei)

        return inf_outs, None

    def collect_logits_and_vectors(self, inf_outs, sample_inds):
        """
        inf_outs containing
            - **obj_logits**     (bsize, tlen, output_vocab_size)
            - **coord_logits**   (bsize, tlen, grid_dim)
            - **attri_logits**   (bsize, tlen, scale_ratio_dim, grid_dim)
            - **patch_vectors**  (bsize, tlen, patch_feature_dim, grid_dim)
        sample_inds              (bsize, tlen, 4)
        """

        obj_logits, coord_logits, attri_logits, patch_vectors, _, _, _ = inf_outs

        obj_inds   = sample_inds[:, :, 0].unsqueeze(-1)
        coord_inds = sample_inds[:, :, 1].unsqueeze(-1)
        scale_inds = sample_inds[:, :, 2].unsqueeze(-1)
        ratio_inds = sample_inds[:, :, 3].unsqueeze(-1)

        bsize, tlen, tsize, grid_dim = patch_vectors.size()
        aux_pos_inds = coord_inds.expand(bsize, tlen, tsize).unsqueeze(-1)
        sample_vectors = torch.gather(patch_vectors, -1, aux_pos_inds).squeeze(-1)

        bsize, tlen, tsize, grid_dim = attri_logits.size()
        aux_pos_inds = coord_inds.expand(bsize, tlen, tsize).unsqueeze(-1)
        local_logits = torch.gather(attri_logits, -1, aux_pos_inds).squeeze(-1)

        scale_logits = local_logits[:, :, :self.cfg.num_scales]
        ratio_logits = local_logits[:, :, self.cfg.num_scales:]

        sample_obj_logits   = torch.gather(obj_logits,   -1, obj_inds)
        sample_coord_logits = torch.gather(coord_logits, -1, coord_inds)
        sample_scale_logits = torch.gather(scale_logits, -1, scale_inds)
        sample_ratio_logits = torch.gather(ratio_logits, -1, ratio_inds)

        sample_logits = torch.cat(
            [ sample_obj_logits, sample_coord_logits,  sample_scale_logits, sample_ratio_logits ],
            -1
        ).contiguous()

        return sample_logits, sample_vectors

    def collect_accuracies(self, inf_outs, sample_inds):
        """
        inf_outs containing
            - **obj_logits**   (bsize, tlen, output_vocab_size)
            - **coord_logits** (bsize, tlen, grid_dim)
            - **attri_logits** (bsize, tlen, scale_ratio_dim, grid_dim)
        sample_inds            (bsize, tlen, 4)
        """

        obj_logits, coord_logits, attri_logits, _, _, _, _ = inf_outs

        obj_inds   = sample_inds[:, :, 0]
        coord_inds = sample_inds[:, :, 1]
        scale_inds = sample_inds[:, :, 2]
        ratio_inds = sample_inds[:, :, 3]

        bsize, tlen, tsize, grid_dim = attri_logits.size()
        aux_pos_inds = coord_inds.view(bsize, tlen, 1).expand(bsize, tlen, tsize).unsqueeze(-1)
        local_logits = torch.gather(attri_logits, -1, aux_pos_inds).squeeze(-1)

        scale_logits = local_logits[:, :, :self.cfg.num_scales]
        ratio_logits = local_logits[:, :, self.cfg.num_scales:]

        _, pred_obj_inds   = torch.max(obj_logits,   -1)
        _, pred_coord_inds = torch.max(coord_logits, -1)
        _, pred_scale_inds = torch.max(scale_logits, -1)
        _, pred_ratio_inds = torch.max(ratio_logits, -1)


        obj_accu   = torch.eq(pred_obj_inds,   obj_inds  ).float()
        coord_accu = torch.eq(pred_coord_inds, coord_inds).float()
        scale_accu = torch.eq(pred_scale_inds, scale_inds).float()
        ratio_accu = torch.eq(pred_ratio_inds, ratio_inds).float()

        sample_accus = torch.stack(
            [obj_accu, coord_accu, scale_accu, ratio_accu],
            -1
        )

        return sample_accus

    def inference(self, input_inds, input_lens, start_step, explore_rate, explore_mode, ref_inds, ref_vecs, nn_table):
        """
        Inputs:
            - **input_inds**   (bsize, src_len)
            - **input_lens**   (bsize, )
            - **start_step**   when to start the exploration
            - **explore_rate** at what rate to explore, othervise use the ref_inds
            - **explore_mode** top1 or multinomial
            - **ref_inds**     (bsize, tlen, 4)
            - **ref_vecs**     (bsize, tlen, patch_feature_dim)

        Outputs:
            inf_outs containing
            - **obj_logits**    (bsize, tlen, output_vocab_size)
            - **coord_logits**  (bsize, tlen, grid_dim)
            - **attri_logits**  (bsize, tlen, scale_ratio_dim, grid_dim)
            - **patch_vectors** (bsize, tlen, patch_feature_dim, grid_dim)
            - **encoder_msks**  (bsize, src_len)
            - **what_att**      (bsize, tlen, src_len)
            - **where_att**     (bsize, tlen, src_len)

            env: simulator to reproduce the predicted scene
        """
        ##############################################################
        # Preparing the env and the first background images
        ##############################################################
        bsize = input_inds.size(0)
        env = self.simulator(self.db, bsize, nn_table)
        bg_imgs = env.reset().unsqueeze(1)
        if self.cfg.cuda:
            bg_imgs = bg_imgs.cuda()

        ##############################################################
        # Text encoder input
        ##############################################################
        text_states = self.text_encoder(input_inds, input_lens)
        enc_embs, enc_rfts, enc_msks, enc_hids = text_states
        ##############################################################
        # Main Loop
        ##############################################################
        obj_logits_list   = []
        coord_logits_list = []
        attri_logits_list = []
        patch_vector_list = []
        what_attn_list    = []
        where_attn_list   = []

        prev_states = (bg_imgs, None, None)
        for i in range(self.cfg.max_output_length + 1): # add one EOS token
            # reference ground truth indices for teacher forcing
            curr_inds = None
            curr_vecs = None
            if ref_inds is not None:
                curr_inds = ref_inds[:, i].unsqueeze(1)
            if ref_vecs is not None:
                curr_vecs = ref_vecs[:, i].unsqueeze(1)

            # scheduled sample & curriculum learning
            if i < start_step:
                # do not explore, use the ref_inds, explore_rate should be smaller than zero
                curr_explore_rate = -0.1
            else:
                curr_explore_rate = explore_rate

            obj_logits, coord_logits, attri_logits, patch_vectors, \
            what_wei, where_wei, \
            next_bg_imgs, next_fgfs, next_hids = \
                self.scheduled_sample_step(env, prev_states, text_states,
                    curr_explore_rate, explore_mode, curr_inds, curr_vecs)

            prev_states = (next_bg_imgs, next_fgfs, next_hids)

            obj_logits_list.append(obj_logits)
            coord_logits_list.append(coord_logits)
            attri_logits_list.append(attri_logits)
            patch_vector_list.append(patch_vectors)
            if self.cfg.what_attn:
                what_attn_list.append(what_wei)
            if self.cfg.where_attn > 0:
                where_attn_list.append(where_wei)

        out_obj_logits    = torch.cat(obj_logits_list,   dim=1)
        out_coord_logits  = torch.cat(coord_logits_list, dim=1)
        out_attri_logits  = torch.cat(attri_logits_list, dim=1)
        out_patch_vectors = torch.cat(patch_vector_list, dim=1)
        out_enc_msks      = enc_msks

        out_what_wei, out_where_wei = None, None
        if self.cfg.what_attn:
            out_what_wei  = torch.cat(what_attn_list,  dim=1)
        if self.cfg.where_attn > 0:
            out_where_wei = torch.cat(where_attn_list, dim=1)

        inf_outs = (out_obj_logits, out_coord_logits, out_attri_logits, out_patch_vectors, out_enc_msks, out_what_wei, out_where_wei)

        return inf_outs, env

    def scheduled_sample_step(self, env, prev_states, text_states, \
        explore_rate, explore_mode, ref_inds, ref_vecs):
        """
        scheduled sample & curriculum learning: one step
        Inputs:
            env

            prev_states containing
            - **prev_bg_imgs** (bsize, 1, channel, height, width)
            - **prev_fgfs**    (bsize, 1, output_vocab_size)
            - **prev_hids**    [tuple of](layer, bsize, tgt_dim, gh, gw)

            text_states
            - **embs** (bsize, slen, emb_dim)
            - **rfts** (bsize, slen, src_dim)
            - **msks** (bsize, slen)
            - **hids** [tuple of](layer, bsize, src_dim)

            explore_rate, explore_mode
            ref_inds (bsize, 1, 4)
            ref_vecs (bsize, 1, patch_feature_dim)

        Outputs:
            - **obj_logits**    (bsize, 1, output_vocab_size)
            - **coord_logits**  (bsize, 1, grid_dim)
            - **attri_logits**  (bsize, 1, scale_ratio_dim, grid_dim)
            - **patch_vectors** (bsize, 1, patch_feature_dim, grid_dim)
            - **what_att**      (bsize, 1, src_len)
            - **where_att**     (bsize, 1, src_len)

            - **next_bg_imgs**  (bsize, 1, channel, height, width)
            - **next_fgfs**     (bsize, 1, output_vocab_size)
            - **next_hids**     [tuple of](layer, bsize, tgt_dim, gh, gw)   
        """

        ##############################################################
        # Decode what
        ##############################################################
        prev_bg_imgs, prev_fgfs, prev_hids = prev_states
        # print('prev_bg_imgs.size()', prev_bg_imgs.size())
        bgfs = self.image_encoder.inference(prev_bg_imgs)
        what_inputs = (bgfs, prev_fgfs, prev_hids)

        what_outs = self.what_decoder(what_inputs, text_states)
        obj_logits, rnn_outs, nxt_hids, prev_bgfs, what_ctx, what_wei = what_outs
        expl_inds = self.decode_what(obj_logits, explore_mode)

        ##############################################################
        # schedule sampling
        ##############################################################
        # sample_prob = torch.FloatTensor(expl_inds.size(0)).uniform_(0, 1)
        # if self.cfg.cuda:
        #     sample_prob = sample_prob.cuda()
        sample_prob = obj_logits.new_empty(expl_inds.size(0)).uniform_(0, 1)
        sample_mask = torch.lt(sample_prob, explore_rate).float()
        if sample_mask.data.sum() == 0:
            obj_inds = ref_inds[:, 0, 0].clone().unsqueeze(-1)
        elif sample_mask.data.min() == 1:
            obj_inds = expl_inds.clone()
        else:
            sample_inds = sample_mask.nonzero().view(-1)
            obj_inds = ref_inds[:, 0, 0].clone().unsqueeze(-1)
            obj_inds.index_copy_(0, sample_inds, expl_inds.index_select(0, sample_inds))


        ##############################################################
        # onehots
        ##############################################################
        # fgfs = torch.zeros(obj_inds.size(0), self.cfg.output_vocab_size).float()
        # if self.cfg.cuda:
        #     fgfs = fgfs.cuda()
        fgfs = obj_logits.new_zeros((obj_inds.size(0), self.cfg.output_vocab_size))
        fgfs.scatter_(-1, obj_inds, 1.0)
        curr_fgfs = fgfs.unsqueeze(1)

        ##############################################################
        # Decode where
        ##############################################################
        where_inputs = (rnn_outs, curr_fgfs, prev_bgfs, what_ctx)
        where_outs = self.where_decoder(where_inputs, text_states)
        coord_logits, attri_logits, patch_vectors, where_ctx, where_wei = where_outs

        expl_inds, expl_vecs = self.decode_where(coord_logits, attri_logits, patch_vectors, explore_mode)

        if explore_rate < self.cfg.eps:
            where_inds = ref_inds[:, 0, 1:].clone()
            where_vecs = ref_vecs[:, 0].clone()
        else:
            where_inds = expl_inds.clone()
            where_vecs = expl_vecs.clone()

        sample_inds = torch.cat([obj_inds, where_inds], 1)
        sample_vecs = where_vecs.clone()

        ##############################################################
        # Render next states
        ##############################################################
        simulator_inds = sample_inds.cpu().data.numpy()
        simulator_vecs = sample_vecs.cpu().data.numpy()
        # Render the next background images
        next_bg_imgs = env.batch_render_to_pytorch(simulator_inds, simulator_vecs).unsqueeze(1)
        # Formatted as pytorch tensor/variable
        if self.cfg.cuda:
            next_bg_imgs = next_bg_imgs.cuda()

        return obj_logits, coord_logits, attri_logits, patch_vectors, what_wei, where_wei, next_bg_imgs, curr_fgfs, nxt_hids

    def decode_what(self, input_logits, sample_mode):
        """
        Decode the object prediction
        Inputs: input_logits, sample_mode
            - **input_logits** (bsize, 1, cls_size)
            - **sample_mode**
                0: top 1, 1: multinomial

        Outputs:
            - **sample_inds**   (bsize, 1)
        """

        ##############################################################
        # Sampling next words
        ##############################################################
        logits = input_logits.squeeze(1)
        if sample_mode == 0:
            # top 1
            _, sample_inds = torch.max(logits + 1.0, dim=-1, keepdim=True)
        else:
            # multinomial
            sample_inds = Categorical(logits).sample().unsqueeze(-1)
        return sample_inds

    def decode_where(self, input_coord_logits, input_attri_logits, input_patch_vectors, sample_mode):
        """
        Inputs:
            where_states containing
            - **coord_logits**  (bsize, 1, grid_dim)
            - **attri_logits**  (bsize, 1, scale_ratio_dim, grid_dim)
            - **patch_vectors** (bsize, 1, patch_feature_dim, grid_dim)
            sample_mode
              0: top 1, 1: multinomial

        Outputs
            - **sample_inds**   (bsize, 3)
            - **sample_vecs**   (bsize, patch_feature_dim)
        """

        ##############################################################
        # Sampling locations
        ##############################################################
        coord_logits = input_coord_logits.squeeze(1)
        if sample_mode == 0:
            _, sample_coord_inds = torch.max(coord_logits + 1.0, dim=-1, keepdim=True)
        else:
            sample_coord_inds = Categorical(coord_logits).sample().unsqueeze(-1)

        ##############################################################
        # Sampling attributes and patch vectors
        ##############################################################

        patch_vectors = input_patch_vectors.squeeze(1)
        bsize, tsize, grid_dim = patch_vectors.size()
        aux_pos_inds = sample_coord_inds.expand(bsize, tsize).unsqueeze(-1)
        sample_patch_vectors = torch.gather(patch_vectors, -1, aux_pos_inds).squeeze(-1)

        attri_logits = input_attri_logits.squeeze(1)
        bsize, tsize, grid_dim = attri_logits.size()
        aux_pos_inds = sample_coord_inds.expand(bsize, tsize).unsqueeze(-1)
        local_logits = torch.gather(attri_logits, -1, aux_pos_inds).squeeze(-1)

        scale_logits = local_logits[:, :self.cfg.num_scales]
        ratio_logits = local_logits[:, self.cfg.num_scales:]

        if sample_mode == 0:
            _, sample_scale_inds = torch.max(scale_logits + 1.0, dim=-1, keepdim=True)
            _, sample_ratio_inds = torch.max(ratio_logits + 1.0, dim=-1, keepdim=True)
        else:
            sample_scale_inds = Categorical(scale_logits).sample().unsqueeze(-1)
            sample_ratio_inds = Categorical(ratio_logits).sample().unsqueeze(-1)

        sample_inds = torch.cat(
            [sample_coord_inds, sample_scale_inds, sample_ratio_inds],
            -1
        )

        return sample_inds, sample_patch_vectors
