#!/usr/bin/env python

import cv2, random, json 
import numpy as np

import torch
import torch.nn as nn

from modules.layout_encoder import TextEncoder, VolumeEncoder
from modules.layout_decoder import WhatDecoder, WhereDecoder
from modules.layout_simulator import simulator
from torch.distributions.categorical import Categorical


class DrawModel(nn.Module):
    def __init__(self, imdb):
        super(DrawModel, self).__init__()
        self.db  = imdb
        self.cfg = imdb.cfg
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.simulator = simulator
        self.image_encoder = VolumeEncoder(self.cfg)
        self.text_encoder  = TextEncoder(self.db)        
        self.what_decoder  = WhatDecoder(self.cfg)
        self.where_decoder = WhereDecoder(self.cfg)

    def inference(self, input_inds, input_lens, start_step, explore_rate, explore_mode, ref_inds):
        """
        Inputs: 
            - **input_inds**   (bsize, src_len)
            - **input_lens**   (bsize, )
            - **start_step** 
            - **explore_rate**
            - **explore_mode** 
            - **ref_inds**     (bsize, tlen, 4)

        Outputs: 
            inf_outs containing
            - **obj_logits**   (bsize, tlen, output_cls_size)
            - **coord_logits** (bsize, tlen, grid_dim)
            - **attri_logits** (bsize, tlen, sr_dim, grid_dim)
            - **encoder_msks** (bsize, src_len)
            - **what_att**     (bsize, tlen, src_len)
            - **where_att**    (bsize, tlen, src_len)
            env: simulator to reproduce the predicted scene
        """
        ##############################################################
        # Preparing the env and the first background images
        ##############################################################
        bsize = input_inds.size(0)
        env = self.simulator(self.db, bsize)
        bg_imgs = env.reset().unsqueeze(1)
        if self.cfg.cuda:
            bg_imgs = bg_imgs.cuda()
        # print('bg_imgs', bg_imgs.size())

        ##############################################################
        # Text encoder input
        ##############################################################
        encoder_states = self.text_encoder(input_inds, input_lens)
        enc_rfts, enc_embs, enc_msks, enc_hids = encoder_states 
        ##############################################################
        # Main Loop
        ##############################################################
        obj_logits_list, coord_logits_list, attri_logits_list = [], [], []
        what_attn_list, where_attn_list = [], []

        prev_states = (bg_imgs, None, None)
        for i in range(self.cfg.max_output_length + 1): # add one EOS token
            # reference ground truth indices for teacher forcing
            curr_inds = None
            if ref_inds is not None:
                curr_inds = ref_inds[:, i].unsqueeze(1)

            # scheduled sample & curriculum learning
            if i < start_step:
                curr_explore_rate = -0.1
            else:
                curr_explore_rate = explore_rate

            obj_logits, coord_logits, attri_logits, what_wei, where_wei, next_bg_imgs, nxt_hids, curr_fgfs = \
                self.scheduled_sample_step(env, prev_states, encoder_states, 
                    curr_explore_rate, explore_mode, curr_inds)

            prev_states = (next_bg_imgs, curr_fgfs, nxt_hids)
            
            obj_logits_list.append(obj_logits)
            coord_logits_list.append(coord_logits)
            attri_logits_list.append(attri_logits)
            if self.cfg.what_attn:
                what_attn_list.append(what_wei)
            if self.cfg.where_attn > 0:
                where_attn_list.append(where_wei)

        out_obj_logits   = torch.cat(obj_logits_list, dim=1)
        out_coord_logits = torch.cat(coord_logits_list, dim=1)
        out_attri_logits = torch.cat(attri_logits_list, dim=1)
        out_enc_msks = enc_msks

        # inf_outs = {}
        # inf_outs['obj_logits'] = torch.cat(obj_logits_list, dim=1)
        # inf_outs['coord_logits'] = torch.cat(coord_logits_list, dim=1)
        # inf_outs['attri_logits'] = torch.cat(attri_logits_list, dim=1)
        # inf_outs['encoder_msks'] = encoder_states['msks']

        # if self.cfg.what_attn:
        #     inf_outs['what_att_logits']  = torch.cat(what_attn_list, dim=1)
        # if self.cfg.where_attn > 0:
        #     inf_outs['where_att_logits'] = torch.cat(where_attn_list, dim=1)
        out_what_wei, out_where_wei = None, None
        if self.cfg.what_attn:
            out_what_wei  = torch.cat(what_attn_list, dim=1)
        if self.cfg.where_attn > 0:
            out_where_wei = torch.cat(where_attn_list, dim=1)

        inf_outs = (out_obj_logits, out_coord_logits, out_attri_logits, out_enc_msks, out_what_wei, out_where_wei)

        return inf_outs, env

    def scheduled_sample_step(self, env, prev_states, encoder_states, \
        explore_rate, explore_mode, ref_inds):
        """
        scheduled sample & curriculum learning: one step
        Inputs: 
            env
            prev_states containing 
            - **bg_imgs** (bsize, 1, channel, height, width)
            - **fgfs**    (bsize, 1, output_cls_size)
            - **hids**    [tuple of](layer, bsize, tgt_dim, gh, gw)

            encoder_states
            - **rfts** (bsize, slen, src_dim)
            - **embs** (bsize, slen, emb_dim)
            - **msks** (bsize, slen)
            - **hids** [tuple of](layer, bsize, src_dim)

            explore_rate, explore_mode
            ref_inds (bsize, 4)

        Outputs: 
            what_states containing
            - **obj_logits** (bsize, 1, output_cls_size):  
            - **what_att**   (bsize, 1, src_len)
            - **bg_imgs**    (bsize, 1, channel, height, width)
            - **hids**       [tuple of](layer, bsize, tgt_dim, gh, gw)
            - **fgfs**       (bsize, 1, output_cls_size)

            where_states containing
            - **coord_logits** (bsize, 1, grid_dim)
            - **attri_logits** (bsize, 1, sr_dim, grid_dim)
            - **where_att**    (bsize, 1, src_len)
        """

        ##############################################################
        # Decode what
        ##############################################################
        prev_bg_imgs, prev_fgfs, prev_hids = prev_states
        bgfs = self.image_encoder(prev_bg_imgs)
        what_inputs = (bgfs, prev_fgfs, prev_hids)

        what_outs = self.what_decoder(what_inputs, encoder_states)
        obj_logits, rnn_outs, nxt_hids, prev_bgfs, what_ctx, what_wei = what_outs
        expl_inds = self.decode_what(obj_logits, explore_mode)

        ##############################################################
        # schedule sampling
        ##############################################################
        sample_prob = torch.FloatTensor(expl_inds.size(0)).uniform_(0, 1)
        if self.cfg.cuda:
            sample_prob = sample_prob.cuda()
        sample_mask = torch.lt(sample_prob, explore_rate).float()
        if sample_mask.data.sum() == 0:
            obj_inds = ref_inds[:,0,0].clone().unsqueeze(-1)
        elif sample_mask.data.min() == 1:
            obj_inds = expl_inds.clone()
        else:
            sample_inds = sample_mask.nonzero().view(-1)
            obj_inds = ref_inds[:,0,0].clone().unsqueeze(-1)
            obj_inds.index_copy_(0, sample_inds, expl_inds.index_select(0, sample_inds))


        ##############################################################
        # onehots
        ##############################################################
        fgfs = torch.zeros(obj_inds.size(0), self.cfg.output_cls_size).float()
        if self.cfg.cuda:
            fgfs = fgfs.cuda()
        fgfs.scatter_(1, obj_inds, 1.0)
        curr_fgfs = fgfs.unsqueeze(1)

        ##############################################################
        # Decode where
        ##############################################################
        where_inputs = (rnn_outs, curr_fgfs, prev_bgfs, what_ctx)
        where_outs = self.where_decoder(where_inputs, encoder_states)
        coord_logits, attri_logits, where_ctx, where_wei = where_outs

        expl_inds  = self.decode_where(coord_logits, attri_logits, explore_mode)
        
        if explore_rate < self.cfg.eps:
            where_inds = ref_inds[:, 0, 1:].clone()
        else:
            where_inds = expl_inds.clone()

        sample_inds = torch.cat([obj_inds, where_inds], 1)
    
        ##############################################################
        # Render next states
        ##############################################################
        # print('sample_inds', sample_inds.size())
        # Render the next background images
        next_bg_imgs = env.batch_render_to_pytorch(
            sample_inds.cpu().data.numpy()).unsqueeze(1)
        # Formatted as pytorch tensor/variable
        if self.cfg.cuda:
            next_bg_imgs = next_bg_imgs.cuda()

        # what_outs['bg_imgs'] = next_bg_imgs

        return obj_logits, coord_logits, attri_logits, what_wei, where_wei, next_bg_imgs, nxt_hids, curr_fgfs

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
            # print('top 1:', sample_inds)
        else:
            # multinomial 
            sample_inds = Categorical(logits).sample().unsqueeze(-1)
            # print('multinomial:', sample_inds)
        return sample_inds

    def decode_where(self, input_coord_logits, input_attri_logits, sample_mode):
        """
        Inputs: 
            where_states containing
            - **coord_logits** (bsize, 1, grid_dim)
            - **attri_logits** (bsize, 1, sr_dim, grid_dim)
            sample_mode
              0: top 1, 1: multinomial
            
        Outputs
            - **sample_inds**   (bsize, 3)
        """

        ##############################################################
        # Sampling locations
        ##############################################################

        coord_logits = input_coord_logits.squeeze(1)
        if sample_mode == 0:
            _, sample_coord_inds = torch.max(coord_logits + 1.0, dim=-1, keepdim=True)
            # print('top 1:', sample_coord_inds)
        else:
            sample_coord_inds = Categorical(coord_logits).sample().unsqueeze(-1)
            # print('multinomial:', sample_coord_inds)
        
        ##############################################################
        # Sampling attributes
        ##############################################################

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

        return sample_inds

    def forward(self, inputs):
        input_inds, input_lens, input_bg_imgs, input_fg_onehots = inputs
        return self.teacher_forcing(input_inds, input_lens, input_bg_imgs, input_fg_onehots)

    def teacher_forcing(self, input_inds, input_lens, input_bg_imgs, input_fg_onehots):
        
        bgfs = self.image_encoder(input_bg_imgs)
        encoder_states = self.text_encoder(input_inds, input_lens)
        enc_rfts, enc_embs, enc_msks, enc_hids = encoder_states 

        prev_states = (bgfs, input_fg_onehots[:,:-1], None)
        what_states = self.what_decoder(prev_states, encoder_states)
        obj_logits, rnn_outs, nxt_hids, prev_bgfs, what_ctx, what_wei = what_states

        where_inputs = (rnn_outs, input_fg_onehots[:,1:], prev_bgfs, what_ctx)
        where_states = self.where_decoder(where_inputs, encoder_states)
        coord_logits, attri_logits, where_ctx, where_wei = where_states

        # inf_outs = {}
        # inf_outs['obj_logits']   = what_states['obj_logits']
        # inf_outs['coord_logits'] = where_states['coord_logits']
        # inf_outs['attri_logits'] = where_states['attri_logits']
        # inf_outs['encoder_msks'] = encoder_states['msks']
        # if self.cfg.what_attn:
        #     inf_outs['what_att_logits']  = what_states['attn_wei']
        # if self.cfg.where_attn > 0:
        #     inf_outs['where_att_logits'] = where_states['attn_wei']
    
        inf_outs = (obj_logits, coord_logits, attri_logits, enc_msks, what_wei, where_wei)

        return inf_outs, None

    def collect_logits(self, inf_outs, sample_inds):
        """
        inf_outs containing
            - **obj_logits**   (bsize, tlen,     output_cls_size)
            - **coord_logits** (bsize, tlen,     grid_dim)
            - **attri_logits** (bsize, tlen, sr_dim, grid_dim)
        sample_inds            (bsize, tlen, 4)
        """

        obj_logits, coord_logits, attri_logits, _, _, _ = inf_outs
        
        obj_inds   = sample_inds[:, :, 0].unsqueeze(-1)
        coord_inds = sample_inds[:, :, 1].unsqueeze(-1)
        scale_inds = sample_inds[:, :, 2].unsqueeze(-1)
        ratio_inds = sample_inds[:, :, 3].unsqueeze(-1)

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

        return sample_logits

    def collect_accuracies(self, inf_outs, sample_inds):
        """
        inf_outs containing
            - **obj_logits**   (bsize, tlen,     output_cls_size)
            - **coord_logits** (bsize, tlen,     grid_dim)
            - **attri_logits** (bsize, tlen, sr_dim, grid_dim)
        sample_inds            (bsize, tlen, 4)
        """

        obj_logits, coord_logits, attri_logits, _, _, _ = inf_outs

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

    ##########################################################################
    # Beam search
    ##########################################################################
    def decode_topk_what(self, obj_probs, accum_logprobs, accum_seq_msks, K):
        #####################################################################
        # Top N probs for each sample
        #####################################################################
        curr_probs = obj_probs.squeeze(1)
        curr_probs = curr_probs.clamp(min=self.cfg.eps)

        bsize, vsize = curr_probs.size()
        N = min(vsize, K)
        # add 1.0 for numerical purpose
        _, top_inds = torch.topk(curr_probs + 1.0, N, -1)
        curr_probs  = torch.gather(curr_probs, -1, top_inds.view(bsize, N))
        curr_logprobs = torch.log(curr_probs)
    
        #####################################################################
        # Accumulate the log probs
        if bsize != 1:
            # If not the first step
            # For paths already terminated, the beam size should be one
            dead_end_mask = -torch.ones(bsize, N) * 3.40282e+37
            dead_end_mask[:, 0] = 0.0
            if self.cfg.cuda:
                dead_end_mask = dead_end_mask.cuda()

            curr_logprobs = curr_logprobs * accum_seq_msks.unsqueeze(-1)
            dead_logprobs = dead_end_mask * (1.0 - accum_seq_msks.unsqueeze(-1))

            logprobs = accum_logprobs.unsqueeze(-1) + curr_logprobs + dead_logprobs
        else:
            logprobs = curr_logprobs
        #####################################################################
        
        #####################################################################
        # Sort the log probs
        logprobs = logprobs.view(-1)
        top_inds = top_inds.view(-1)
        # Top K log probs for all samples
        _, lin_inds = torch.topk(torch.exp(logprobs), K, 0)
        #####################################################################

        output_accum_logprobs = torch.index_select(logprobs, 0, lin_inds)
        output_inds = torch.index_select(top_inds, 0, lin_inds)
        beam_inds = (lin_inds / N).long()

        #####################################################################
        # Update mask
        output_accum_seq_msks = torch.index_select(accum_seq_msks, 0, beam_inds)
        mask = torch.ge(-output_inds, -self.cfg.EOS_idx)
        output_accum_seq_msks.masked_fill_(mask, 0.0)
        output_inds = output_inds.unsqueeze(-1)
        #####################################################################

        return output_accum_logprobs, output_accum_seq_msks, output_inds, beam_inds

    def decode_topk_step(self, accum_logprobs, accum_seq_msks, prev_states, encoder_states, env, K):
        """
        Decode one step
        Inputs: 
            - **accum_logprobs**  (bsize, )
                tensor containing accumulated log likelihood of different paths
            - **accum_seq_msks** (bsize, )
                tensor indicating whether the sequences are ended

            prev_states containing 
            - **bg_imgs** (bsize, 1, channel, height, width)
            - **fgfs**    (bsize, 1, output_cls_size)
            - **hids**    [tuple of](layer, bsize, tgt_dim, gh, gw)

            encoder_states
            - **rfts** (bsize, slen, src_dim)
            - **embs** (bsize, slen, emb_dim)
            - **msks** (bsize, slen)
            - **hids** [list of][tuple of](layer, bsize, src_dim)

            - **env** 
                simulator containing the state of the scene
            - **K**
                beam size
            
        Outputs: 
            - **next_logprobs** (bsize, )
                tensor containing accumulated log likelihood of different paths
            - **next_seq_msks** (bsize, )
                tensor indicating whether the sequences are ended

            next_states containing 
            - **bg_imgs** (bsize, 1, channel, height, width)
            - **fgfs**    (bsize, 1, output_cls_size)
            - **hids**    [tuple of](layer, bsize, tgt_dim, gh, gw)

            next_encoder_states
            - **rfts** (bsize, slen, src_dim)
            - **embs** (bsize, slen, emb_dim)
            - **msks** (bsize, slen)
            - **hids** [list of][tuple of](layer, bsize, src_dim)
        """

        ##############################################################
        # Decode what
        ##############################################################
        prev_bg_imgs, prev_fgfs, prev_hids = prev_states
        bgfs = self.image_encoder(prev_bg_imgs)
        what_inputs = (bgfs, prev_fgfs, prev_hids)

        what_outs = self.what_decoder(what_inputs, encoder_states)
        obj_logits, rnn_outs, nxt_hids, prev_bgfs, what_ctx, what_wei = what_outs

        accum_logprobs, accum_seq_msks, obj_inds, beam_inds_1 = \
            self.decode_topk_what(obj_logits, accum_logprobs, accum_seq_msks, K)
        
        ##############################################################
        # Update inputs for "where" prediction
        ##############################################################
        rnn_outs = torch.index_select(rnn_outs, 0, beam_inds_1)
        nxt_hids = torch.index_select(nxt_hids, 1, beam_inds_1)
        prev_bgfs  = torch.index_select(prev_bgfs, 0, beam_inds_1)
        if self.cfg.what_attn:
            what_ctx = torch.index_select(what_ctx, 0, beam_inds_1)
            what_wei = torch.index_select(what_wei, 0, beam_inds_1)

        enc_rfts, enc_embs, enc_msks, enc_hids = encoder_states
        enc_rfts = torch.index_select(enc_rfts, 0, beam_inds_1)
        enc_embs = torch.index_select(enc_embs, 0, beam_inds_1)
        enc_msks = torch.index_select(enc_msks, 0, beam_inds_1)
        encoder_states = (enc_rfts, enc_embs, enc_msks, enc_hids)


        ##############################################################
        # onehots
        ##############################################################
        fgfs = torch.zeros(obj_inds.size(0), self.cfg.output_cls_size).float()
        if self.cfg.cuda:
            fgfs = fgfs.cuda()
        fgfs.scatter_(1, obj_inds, 1.0)
        curr_fgfs = fgfs.unsqueeze(1)

        ##############################################################
        # Decode where
        ##############################################################
        where_inputs = (rnn_outs, curr_fgfs, prev_bgfs, what_ctx)
        where_outs = self.where_decoder(where_inputs, encoder_states)
        coord_logits, attri_logits, where_ctx, where_wei = where_outs

        expl_inds  = self.decode_where(coord_logits, attri_logits, 0)
        where_inds = expl_inds.clone()
        sample_inds = torch.cat([obj_inds, where_inds], 1)

        ##############################################################
        # Render next states
        ##############################################################
        env.select(beam_inds_1.cpu().data.numpy())
        # print('sample_inds', sample_inds.size())
        # Render the next background images
        next_bg_imgs = env.batch_render_to_pytorch(
            sample_inds.cpu().data.numpy()).unsqueeze(1)
        # Formatted as pytorch tensor/variable
        if self.cfg.cuda:
            next_bg_imgs = next_bg_imgs.cuda()
    
        return accum_logprobs, accum_seq_msks, (next_bg_imgs, curr_fgfs, nxt_hids), encoder_states

    def topk_inference(self, input_inds, input_lens, K, start_step = 0, ref_inds=None):
        assert (input_inds.size(0) == 1)
        assert (input_lens.size(0) == 1)

        if start_step > 0:
            assert(ref_inds is not None)

        ##############################################################
        # Preparing background images
        ##############################################################
        env = self.simulator(self.db, 1)
        bg_imgs = env.reset().unsqueeze(1)
        if self.cfg.cuda:
            bg_imgs = bg_imgs.cuda()

        ##############################################################
        # Preparing initial log probs
        ##############################################################
        accum_logprobs = torch.zeros(K)
        accum_seq_msks = torch.ones(K)
        if self.cfg.cuda:
            accum_logprobs = accum_logprobs.cuda()
            accum_seq_msks = accum_seq_msks.cuda()

        ##############################################################
        # Text encoder input
        ##############################################################
        encoder_states = self.text_encoder(input_inds, input_lens)
        enc_rfts, enc_embs, enc_msks, enc_hids = encoder_states 

        ##############################################################
        # Main Loop
        ##############################################################
        prev_states = (bg_imgs, None, None)
        for i in range(self.cfg.max_output_length + 1): # add one EOS token
            curr_inds = None
            if ref_inds is not None:
                curr_inds = ref_inds[:, i].unsqueeze(1)

            if i < start_step:
                _, _, _, _, _, next_bg_imgs, nxt_hids, curr_fgfs = \
                    self.scheduled_sample_step(env, prev_states, encoder_states, 
                        -0.1, 0, curr_inds)
                prev_states = (next_bg_imgs, curr_fgfs, nxt_hids)
            else:
                accum_logprobs, accum_seq_msks, \
                prev_states, encoder_states = self.decode_topk_step(
                    accum_logprobs, accum_seq_msks, 
                    prev_states, encoder_states, env, K)

            if torch.sum(accum_seq_msks) == 0:
                break        
        
        return env

