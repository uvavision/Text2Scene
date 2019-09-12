#!/usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.conv_rnn import ConvGRU, ConvLSTM
from modules.attention import Attention
from modules.separable_convolution import separable_conv2d
from composites_utils import *


class WhatDecoder(nn.Module):
    def __init__(self, config):
        super(WhatDecoder, self).__init__()

        self.cfg = config
        # whether to use separable conv2d
        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d
        #################################################################
        # Dimensions for the recurrent model
        #################################################################
        # If the encoder is bidirectional, double the size of the hidden state
        factor = 2 if config.bidirectional else 1
        src_dim = factor * config.n_src_hidden
        tgt_dim = factor * config.n_tgt_hidden

        emb_dim = config.n_embed
        bgf_dim = config.n_conv_hidden
        fgf_dim = config.output_vocab_size

        #################################################################
        # Conv RNN
        #################################################################
        input_dim = bgf_dim
        if config.use_fg_to_pred == 1:
            # use prev_fg_onehot as input, seems not working
            input_dim += fgf_dim
        rnn_cell = config.rnn_cell.lower()
        if rnn_cell == 'gru':
            self.rnn = ConvGRU(input_dim, tgt_dim, config.n_rnn_layers, 3, bias=True, dropout=config.rnn_dropout_p)
        elif rnn_cell == 'lstm':
            self.rnn = ConvLSTM(input_dim, tgt_dim, config.n_rnn_layers, 3, bias=True, dropout=config.rnn_dropout_p)
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        #################################################################
        # Spatial attention
        #################################################################
        if self.cfg.what_attn_2d:
            attn2d_layers = []
            if self.cfg.use_normalization:
                attn2d_layers.append(conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1, bias=False))
                attn2d_layers.append(nn.LayerNorm([tgt_dim//2, self.cfg.grid_size[0], self.cfg.grid_size[1]]))
            else:
                attn2d_layers.append(conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1, bias=True))
            attn2d_layers.append(nn.LeakyReLU(0.2, inplace=True))
            attn2d_layers.append(conv2d(tgt_dim//2, 1, kernel_size=3, stride=1, padding=1))
            self.spatial_attn = nn.Sequential(*attn2d_layers)

        #################################################################
        # Attention
        #################################################################
        if self.cfg.what_attn:
            in_dim = tgt_dim
            out_dim = src_dim
            if self.cfg.attn_emb:
                # whether to include the language embedding vector as output
                out_dim += emb_dim
            # if self.cfg.use_bg_to_pred:
            #     in_dim += bgf_dim
            if self.cfg.use_fg_to_pred == 2:
                in_dim += fgf_dim
            self.attention = Attention(config.attn_type, out_dim, in_dim)

        #################################################################
        # Object decoder
        #################################################################
        input_dim = tgt_dim
        if self.cfg.what_attn:
            input_dim += src_dim
            if self.cfg.attn_emb:
                input_dim += emb_dim
        if self.cfg.use_bg_to_pred:
            input_dim += bgf_dim
        if self.cfg.use_fg_to_pred == 2:
            input_dim += fgf_dim

        hidden_dim = tgt_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, fgf_dim)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def recurrent_forward(self, prev_feats, prev_hids):
        curr_feats, curr_hids, _ = self.rnn(prev_feats, prev_hids)

        if self.cfg.what_attn_2d:
            bsize, tlen, tdim, gh, gw = curr_feats.size()
            nsize = bsize * tlen
            flatten_outs = curr_feats.view(nsize, tdim, gh, gw)

            attn_map = self.spatial_attn(flatten_outs)
            attn_map = attn_map.view(nsize, gh, gw)
            attn_map = attn_map.view(nsize, gh * gw)
            attn_map = F.softmax(attn_map, dim=-1)
            attn_map = attn_map.view(nsize, gh, gw)
            attn_map = attn_map.view(bsize, tlen, gh, gw)
            attn_map = attn_map.view(bsize, tlen, 1, gh, gw)

            prev_alpha_feats = prev_feats * attn_map
            curr_alpha_feats = curr_feats * attn_map

            prev_feats_1d = torch.sum(torch.sum(prev_alpha_feats, -1), -1)
            curr_feats_1d = torch.sum(torch.sum(curr_alpha_feats, -1), -1)
        else:
            prev_feats_1d = torch.mean(torch.mean(prev_feats, -1), -1)
            curr_feats_1d = torch.mean(torch.mean(curr_feats, -1), -1)

        return curr_feats, curr_hids, prev_feats_1d, curr_feats_1d

    def forward(self, prev_states, encoder_states):
        """
        Inputs:
            prev_states containing
            - **prev_bgfs**  (bsize, tlen, bgf_dim, height, width)
            - **prev_fgfs**  (bsize, tlen, fgf_dim)
            - **prev_hids**  [tuple of](layer, bsize, tgt_dim, height, width)

            encoder_states containing
            - **enc_embs** (bsize, slen, emb_dim)
            - **enc_rfts** (bsize, slen, src_dim)
            - **enc_msks** (bsize, slen)
            - **enc_hids** [tuple of](layer, bsize, src_dim)

        Outputs:
            - **obj_logits**   (bsize, tlen, output_vocab_size)
            - **rnn_feats_2d** (bsize, tlen, tgt_dim, height, width)
            - **nxt_hids_2d**  [tuple of](layer, bsize, tgt_dim, height, width)
            - **prev_bgfs**    (bsize, tlen, bgf_dim, height, width)
            - **att_ctx**      (bsize, tlen, src_dim (+ emb_dim))
            - **att_wei**      (bsize, tlen, slen)
        """
        #################################################################
        # Unfold the inputs
        #################################################################
        prev_bgfs, prev_fgfs, prev_hids = prev_states
        enc_embs, enc_rfts, enc_msks, enc_hids = encoder_states

        bsize, tlen, bgf_dim, gh, gw = prev_bgfs.size()
        fgf_dim = self.cfg.output_vocab_size

        #################################################################
        # Initialize the foreground onehot representation if necessary
        #################################################################
        if self.cfg.use_fg_to_pred > 0:
            if prev_fgfs is None:
                prev_fgfs = torch.zeros(bsize, tlen, fgf_dim).float()
                start_inds = self.cfg.SOS_idx * torch.ones(bsize, tlen, 1).long()
                prev_fgfs.scatter_(2, start_inds, 1.0)
                if self.cfg.cuda:
                    prev_fgfs = prev_fgfs.cuda()
            if self.cfg.use_fg_to_pred == 1:
                prev_fgfs = prev_fgfs.view(bsize, tlen, fgf_dim, 1, 1)
                prev_fgfs = prev_fgfs.expand(bsize, tlen, fgf_dim, gh, gw)

        #################################################################
        # Initialize the hidden states if necessary
        #################################################################
        if prev_hids is None:
            prev_hids = self.init_state(enc_hids)

        #################################################################
        # If use fgf as input for the conv rnn
        #################################################################
        prev_feats = prev_bgfs
        if self.cfg.use_fg_to_pred == 1:
            prev_feats = torch.cat([prev_feats, prev_fgfs], 2)

        #################################################################
        # Recurrent forward
        #################################################################
        rnn_feats_2d, nxt_hids_2d, img_feats_1d, rnn_feats_1d = \
            self.recurrent_forward(prev_feats, prev_hids)

        att_src = rnn_feats_1d
        if self.cfg.use_fg_to_pred == 2:
            att_src = torch.cat([att_src, prev_fgfs], -1)

        #################################################################
        # Attention
        #################################################################
        att_ctx, att_wei = None, None
        if self.cfg.what_attn:
            encoder_feats = enc_rfts
            if self.cfg.attn_emb:
                encoder_feats = torch.cat([encoder_feats, enc_embs], -1)
            att_ctx, att_wei = self.attention(att_src, encoder_feats, enc_msks)

        #################################################################
        # Combine features
        #################################################################
        combined = att_src
        if self.cfg.what_attn:
            combined = torch.cat((combined, att_ctx), dim=2)
        if self.cfg.use_bg_to_pred:
            combined = torch.cat((combined, img_feats_1d), dim=2)

        #################################################################
        # Classifer
        #################################################################
        logits = self.decoder(combined)
        obj_logits = F.softmax(logits,  -1)

        return obj_logits, rnn_feats_2d, nxt_hids_2d, prev_bgfs, att_ctx, att_wei

    def init_state(self, encoder_hidden):
        if isinstance(encoder_hidden, tuple):
            # LSTM
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            # GRU
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.cfg.bidirectional:
            new_h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        else:
            new_h = h
        lsize, bsize, hsize = new_h.size()
        new_h = new_h.view(lsize, bsize, hsize, 1, 1)
        new_h = new_h.expand(lsize, bsize, hsize, self.cfg.grid_size[1], self.cfg.grid_size[0])
        return new_h


class WhereDecoder(nn.Module):
    def __init__(self, config):
        super(WhereDecoder, self).__init__()
        self.cfg = config
        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d
        #################################################################
        # Dimensions
        #################################################################
        # If the encoder is bidirectional, double the size of the hidden state
        factor = 2 if config.bidirectional else 1
        src_dim = factor * config.n_src_hidden
        tgt_dim = factor * config.n_tgt_hidden
        emb_dim = config.n_embed
        bgf_dim = config.n_conv_hidden
        fgf_dim = config.output_vocab_size

        #################################################################
        # Attention
        #################################################################
        if self.cfg.what_attn and self.cfg.where_attn > 0:
            in_dim = fgf_dim
            out_dim = src_dim
            if self.cfg.attn_emb:
                out_dim += emb_dim
            if self.cfg.where_attn == 2:
                in_dim += out_dim
            self.attention = Attention(config.attn_type, out_dim, in_dim)

            if self.cfg.where_attn_2d:
                in_dim_2d = out_dim + tgt_dim
                attn2d_layers = []
                if self.cfg.use_normalization:
                    attn2d_layers.append(conv2d(in_dim_2d, tgt_dim//2, kernel_size=3, stride=1, padding=1, bias=False))
                    attn2d_layers.append(nn.LayerNorm([tgt_dim//2, self.cfg.grid_size[0], self.cfg.grid_size[1]]))
                else:
                    attn2d_layers.append(conv2d(in_dim_2d, tgt_dim//2, kernel_size=3, stride=1, padding=1, bias=True))
                attn2d_layers.append(nn.LeakyReLU(0.2, inplace=True))
                attn2d_layers.append(conv2d(tgt_dim//2, 1, kernel_size=3, stride=1, padding=1))
                self.spatial_attn = nn.Sequential(*attn2d_layers)

        #################################################################
        # Location decoder
        #################################################################
        input_dim = tgt_dim + fgf_dim
        if self.cfg.what_attn:
            input_dim += src_dim
            if self.cfg.attn_emb:
                input_dim += emb_dim

        if self.cfg.use_bg_to_locate:
            input_dim += bgf_dim

        output_dim = 1 + self.cfg.num_scales + self.cfg.num_ratios + self.cfg.n_patch_features

        if config.use_normalization:
            self.decoder = nn.Sequential(
                conv2d(input_dim, tgt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LayerNorm([tgt_dim, self.cfg.grid_size[0], self.cfg.grid_size[1]]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LayerNorm([tgt_dim//2, self.cfg.grid_size[0], self.cfg.grid_size[1]]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(tgt_dim//2, tgt_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LayerNorm([tgt_dim//2, self.cfg.grid_size[0], self.cfg.grid_size[1]]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(tgt_dim//2, output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.decoder = nn.Sequential(
                conv2d(input_dim, tgt_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(tgt_dim//2, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(tgt_dim//2, output_dim, kernel_size=3, stride=1, padding=1),
            )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, what_states, encoder_states):
        """
        Inputs:
            what_states containing
            - **rnn_outs**  (bsize, tlen, tgt_dim, height, width)
            - **curr_fgfs** (bsize, tlen, output_vocab_size)
            - **prev_bgfs** (bsize, tlen, bgf_dim, height, width)
            - **what_ctx**  (bsize, tlen, src_dim[+emb_dim])

            encoder_states containing
            - **enc_embs** (bsize, slen, emb_dim)
            - **enc_rfts** (bsize, slen, src_dim)
            - **enc_msks** (bsize, slen)
            - **enc_hids** [tuple of](layer, bsize, src_dim)

        Outputs:
            - **coord_logits**  (bsize, tgt_len, grid_dim)
            - **attri_logits**  (bsize, tgt_len, scale_ratio_dim, grid_dim)
            - **patch_vectors** (bsize, tgt_len, patch_dim, grid_dim)
            - **att_ctx**  (bsize, tlen, src_dim[+emb_dim]))
            - **att_wei**  (bsize, tlen, slen)
        """
        #################################################################
        # Unfold the inputs
        #################################################################
        rnn_outs, curr_fgfs, prev_bgfs, what_ctx = what_states
        enc_embs, enc_rfts,  enc_msks, enc_hids = encoder_states

        bsize, tlen, fgf_dim = curr_fgfs.size()
        bsize, tlen, tgt_dim, gh, gw = rnn_outs.size()

        #################################################################
        # Attention context
        #################################################################
        att_ctx, att_wei = None, None
        if self.cfg.what_attn:
            if self.cfg.where_attn == 0:
                att_ctx = what_ctx
            else:
                encoder_feats = enc_rfts
                if self.cfg.attn_emb:
                    encoder_feats = torch.cat([encoder_feats, enc_embs], -1)
                if self.cfg.where_attn == 1:
                    query = curr_fgfs
                else:
                    query = torch.cat([curr_fgfs, what_ctx], -1)
                att_ctx, att_wei = self.attention(query, encoder_feats, enc_msks)

            bsize, tlen, att_dim = att_ctx.size()
            # Replicate
            ctx_2d = att_ctx.view(bsize, tlen, att_dim, 1, 1)
            ctx_2d = ctx_2d.expand(bsize, tlen, att_dim, gh, gw)

            if self.cfg.where_attn > 0 and self.cfg.where_attn_2d:
                attn_input = torch.cat([rnn_outs, ctx_2d], dim=2)
                bsize, tlen, tdim, gh, gw = attn_input.size()
                nsize = bsize * tlen
                flatten_outs = attn_input.view(nsize, tdim, gh, gw)

                attn_map = self.spatial_attn(flatten_outs)
                attn_map = attn_map.view(nsize, gh, gw)
                attn_map = attn_map.view(nsize, gh * gw)
                attn_map = F.softmax(attn_map, dim=-1)
                attn_map = attn_map.view(nsize, gh, gw)
                attn_map = attn_map.view(bsize, tlen, gh, gw)
                attn_map = attn_map.view(bsize, tlen, 1, gh, gw)

                attn_rnn_outs = rnn_outs * attn_map
            else:
                attn_rnn_outs = rnn_outs
        else:
            attn_rnn_outs = rnn_outs

        #################################################################
        # Expand fg features
        #################################################################

        fg_2d = curr_fgfs.view(bsize, tlen, fgf_dim, 1, 1)
        fg_2d = fg_2d.expand(bsize, tlen, fgf_dim, gh, gw)

        #################################################################
        # Concatenate with fg features
        #################################################################
        if self.cfg.what_attn:
            combined = torch.cat([attn_rnn_outs, fg_2d, ctx_2d], dim=2)
        else:
            combined = torch.cat([attn_rnn_outs, fg_2d], dim=2)
        if self.cfg.use_bg_to_locate:
            combined = torch.cat([combined, prev_bgfs], dim=2)

        #################################################################
        # Classifer
        #################################################################
        bsize, tlen, fsize, gh, gw = combined.size()
        combined = combined.view(bsize * tlen, fsize, gh, gw)
        logits = self.decoder(combined)
        nsize, fsize, gh, gw = logits.size()
        assert(nsize == bsize * tlen)
        logits = logits.view(bsize, tlen, fsize, gh, gw)

        #################################################################
        # Reshape the outputs
        #################################################################
        logits = logits.view(bsize, tlen, fsize, -1)
        coord_logits = logits[:,:,0,:]
        scale_logits = logits[:,:,1:(self.cfg.num_scales+1), :]
        ratio_logits = logits[:,:,(self.cfg.num_scales+1):(self.cfg.num_scales+self.cfg.num_ratios+1), :]
        patch_vectors = logits[:,:,(self.cfg.num_scales+self.cfg.num_ratios+1):, :]

        coord_logits = F.softmax(coord_logits, dim=-1)
        scale_logits = F.softmax(scale_logits, dim=-2)
        ratio_logits = F.softmax(ratio_logits, dim=-2)
        attri_logits = torch.cat([scale_logits, ratio_logits], 2)
        # patch_vectors = torch.tanh(patch_vectors)
        patch_vectors = F.normalize(patch_vectors, dim=-2)
        return coord_logits, attri_logits, patch_vectors, att_ctx, att_wei
