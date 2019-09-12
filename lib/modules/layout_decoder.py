#!/usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.conv_rnn import ConvGRU, ConvLSTM
from modules.attention import Attention
from layout_utils import *


class WhatDecoder(nn.Module):
    def __init__(self, config):
        super(WhatDecoder, self).__init__()

        self.cfg = config
        #################################################################
        # Dimensions
        #################################################################
        # If the encoder is bidirectional, double the size of the hidden state
        factor = 2 if config.bidirectional else 1
        src_dim = factor * config.n_src_hidden
        tgt_dim = factor * config.n_tgt_hidden 

        emb_dim = config.n_embed
        bgf_dim = config.n_conv_hidden
        fgf_dim = config.output_cls_size

        #################################################################
        # Conv RNN
        #################################################################
        input_dim = bgf_dim
        if config.use_fg_to_pred == 1: # use prev_fg_onehot as input
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
        if self.cfg.attn_2d:
            if self.cfg.use_bn:
                self.spatial_attn = nn.Sequential(
                    nn.Conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(tgt_dim//2),
                    nn.ReLU(True),
                    nn.Conv2d(tgt_dim//2, 1, kernel_size=3, stride=1, padding=1),
                )
            else:
                self.spatial_attn = nn.Sequential(
                    nn.Conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(tgt_dim//2),
                    nn.ReLU(True),
                    nn.Conv2d(tgt_dim//2, 1, kernel_size=3, stride=1, padding=1),
                )

        #################################################################
        # Attention
        #################################################################
        if self.cfg.what_attn:
            in_dim = tgt_dim
            out_dim = src_dim
            if self.cfg.attn_emb:
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
            # nn.Linear(input_dim, hidden_dim, bias=out_bias),
            nn.Linear(input_dim, hidden_dim, bias=True),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, fgf_dim)
        )
            
        # self.init_weights()
 
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def segment_forward(self, prev_feats, prev_hids):
        curr_outs, curr_hids, _ = self.rnn(prev_feats, prev_hids)

        if self.cfg.attn_2d:
            bsize, tlen, tdim, gh, gw = curr_outs.size()
            nsize = bsize * tlen
            flatten_outs = curr_outs.view(nsize, tdim, gh, gw)

            attn_map = self.spatial_attn(flatten_outs)
            attn_map = attn_map.view(nsize, gh, gw)
            attn_map = attn_map.view(nsize, gh * gw)
            attn_map = F.softmax(attn_map, dim=-1)
            attn_map = attn_map.view(nsize, gh, gw)
            attn_map = attn_map.view(bsize, tlen, gh, gw)
            attn_map = attn_map.view(bsize, tlen, 1, gh, gw)

            prev_alpha_feats = prev_feats * attn_map
            curr_alpha_outs  = curr_outs  * attn_map

            prev_attn_feats = torch.sum(torch.sum(prev_alpha_feats, -1), -1)
            curr_attn_outs  = torch.sum(torch.sum(curr_alpha_outs, -1), -1)
        else:
            prev_attn_feats = torch.mean(torch.mean(prev_feats, -1), -1)
            curr_attn_outs  = torch.mean(torch.mean(curr_outs, -1), -1)

        # outs = {}
        # outs['curr_outs'] = curr_outs
        # outs['curr_hids'] = curr_hids
        # outs['prev_attn_feats'] = prev_attn_feats
        # outs['curr_attn_outs'] = curr_attn_outs

        return curr_outs, curr_hids, prev_attn_feats, curr_attn_outs

    def forward(self, prev_states, encoder_states):
        """
        Inputs: 
            prev_states containing
            - **bgfs**  (bsize, tlen, bgf_dim, height, width)
            - **fgfs**  (bsize, tlen, fgf_dim)
            - **hids**  [tuple of](layer, bsize, tgt_dim, height, width)

            encoder_states containing
            - **rfts** (bsize, slen, src_dim)
            - **embs** (bsize, slen, emb_dim)
            - **msks** (bsize, slen)
            - **hids** [tuple of](layer, bsize, src_dim)
            
        Outputs: 
            what_states containing
            - **obj_logits** (bsize, tlen, output_cls_size)
            - **rnn_outs**   (bsize, tlen, tgt_dim, height, width)

            - **bgfs**       (bsize, tlen, bgf_dim, height, width)
            - **hids**       [tuple of](layer, bsize, tgt_dim, height, width)

            - **att_ctx**    (bsize, tlen, src_dim (+ emb_dim))
            - **att_wei**    (bsize, tlen, slen)
        """
        #################################################################
        # Conv RNN
        #################################################################
        prev_bgfs, prev_fgfs, prev_hids = prev_states
        enc_rfts, enc_embs, enc_msks, enc_hids = encoder_states 


        bsize, tlen, bgf_dim, gh, gw = prev_bgfs.size()
        fgf_dim = self.cfg.output_cls_size
        
        # Initialize the foreground onehot representation if necessary
        if self.cfg.use_fg_to_pred > 0: 
            if prev_fgfs is None:
                fgf_dim = self.cfg.output_cls_size
                prev_fgfs = torch.zeros(bsize, tlen, fgf_dim).float()
                start_inds = self.cfg.SOS_idx * torch.ones(bsize, tlen, 1).long()
                prev_fgfs.scatter_(2, start_inds, 1.0)
                if self.cfg.cuda:
                    prev_fgfs = prev_fgfs.cuda()
            if self.cfg.use_fg_to_pred == 1:
                prev_fgfs = prev_fgfs.view(bsize, tlen, fgf_dim, 1, 1)
                prev_fgfs = prev_fgfs.expand(bsize, tlen, fgf_dim, gh, gw)
            
        # Initialize the hidden states if necessary
        if prev_hids is None:
            prev_hids = self.init_state(enc_hids)

        # print('prev_hids ', prev_hids.size())
        # If use fgf as input for the conv rnn
        prev_feats = prev_bgfs
        if self.cfg.use_fg_to_pred == 1:
            prev_feats = torch.cat([prev_feats, prev_fgfs], 2)
        # print('prev_feats ', prev_feats.size())

        
        seg_outs = self.segment_forward(prev_feats, prev_hids)
        rnn_outs, nxt_hids, att_imgs, att_rnns = seg_outs
        # print('rnn_outs ', rnn_outs.size())
        # print('nxt_hids ', nxt_hids.size())
        # print('att_imgs ', att_imgs.size())
        # print('att_rnns ', att_rnns.size())

        att_src = att_rnns
        if self.cfg.use_fg_to_pred == 2:
            att_src = torch.cat([att_src, prev_fgfs], -1)
        # print('att_src ', att_src.size())

        #################################################################
        # Attention
        #################################################################
        att_ctx, att_wei = None, None
        if self.cfg.what_attn:
            encoder_feats = enc_rfts
            if self.cfg.attn_emb:
                encoder_feats = torch.cat([encoder_feats, enc_embs], -1)
            # print('encoder_feats ', encoder_feats.size())
            # encoder_msks = encoder_states['msks']
            # print('encoder_msks ', encoder_msks.size())
            att_ctx, att_wei = self.attention(att_src, encoder_feats, enc_msks)
            # print('att_ctx ', att_ctx.size())
            # print('att_wei ', att_wei.size())

        #################################################################
        # Combine features
        #################################################################
        combined = att_src
        if self.cfg.what_attn:
            combined = torch.cat((combined, att_ctx), dim=2)
        if self.cfg.use_bg_to_pred:
            combined = torch.cat((combined, att_imgs), dim=2)
        # print('combined ', combined.size())

        #################################################################
        # Classifer
        #################################################################
        logits = self.decoder(combined)
        # print('logits: ', logits.size())
        obj_logits = F.softmax(logits,  -1)

        # what_outs = {}
        # what_outs['obj_logits'] = obj_logits 
        # # what_outs['pose_logits'] = pose_logits
        # # what_outs['expr_logits'] = expr_logits
        # what_outs['rnn_outs'] = rnn_outs
        # what_outs['hids'] = nxt_hids
        # what_outs['bgfs'] = prev_bgfs
        # if self.cfg.what_attn:
        #     what_outs['attn_ctx'] = att_ctx
        #     what_outs['attn_wei'] = att_wei

        
        return obj_logits, rnn_outs, nxt_hids, prev_bgfs, att_ctx, att_wei

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

        #################################################################
        # Dimensions
        #################################################################
        # If the encoder is bidirectional, double the size of the hidden state
        factor = 2 if config.bidirectional else 1
        src_dim = factor * config.n_src_hidden
        tgt_dim = factor * config.n_tgt_hidden
        emb_dim = config.n_embed
        bgf_dim = config.n_conv_hidden
        fgf_dim = config.output_cls_size 

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
            # print('in_dim',  in_dim)
            # print('out_dim', out_dim)

            if self.cfg.where_attn_2d:
                in_dim_2d = out_dim + tgt_dim
                if self.cfg.use_bn:
                    self.spatial_attn = nn.Sequential(
                        nn.Conv2d(in_dim_2d, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(tgt_dim//2),
                        nn.ReLU(True),
                        nn.Conv2d(tgt_dim//2, 1, kernel_size=3, stride=1, padding=1),
                    )
                else:
                    self.spatial_attn = nn.Sequential(
                        nn.Conv2d(in_dim_2d, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(tgt_dim//2),
                        nn.ReLU(True),
                        nn.Conv2d(tgt_dim//2, 1, kernel_size=3, stride=1, padding=1),
                    )

        #################################################################
        # Location decoder
        #################################################################
        if self.cfg.what_attn:
            input_dim = tgt_dim + fgf_dim + src_dim
            if self.cfg.attn_emb:
                input_dim += emb_dim
        else:
            input_dim = tgt_dim + fgf_dim
        # input_dim = tgt_dim + fgf_dim + src_dim
        # if self.cfg.attn_emb:
        #     input_dim += emb_dim
        if self.cfg.use_bg_to_locate:
            input_dim += bgf_dim
        # print('input_dim',  input_dim)

        output_dim = 1 + self.cfg.num_scales + self.cfg.num_ratios

        if config.use_bn:
            self.decoder = nn.Sequential(
                nn.Conv2d(input_dim, tgt_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(tgt_dim),
                nn.ReLU(True),
                nn.Conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(tgt_dim//2),
                nn.ReLU(True),
                nn.Conv2d(tgt_dim//2, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(tgt_dim//2),
                nn.ReLU(True),
                nn.Conv2d(tgt_dim//2, output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(input_dim, tgt_dim, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(tgt_dim),
                nn.ReLU(True),
                nn.Conv2d(tgt_dim, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(tgt_dim//2),
                nn.ReLU(True),
                nn.Conv2d(tgt_dim//2, tgt_dim//2, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(tgt_dim//2),
                nn.ReLU(True),
                nn.Conv2d(tgt_dim//2, output_dim, kernel_size=3, stride=1, padding=1),
            )

        # self.init_weights()

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
            - **rnn_outs** (bsize, tlen, tgt_dim, height, width)
            - **bgfs**     (bsize, tlen, bgf_dim, height, width)
            - **fgfs**     (bsize, tlen, output_cls_size)

            encoder_states containing
            - **rfts** (bsize, slen, src_dim)
            - **embs** (bsize, slen, emb_dim)
            - **msks** (bsize, slen)
            - **hids** [tuple of](layer, bsize, src_dim)
            
        Outputs: 
            where_outs containing
            - **coord_logits** (bsize, tgt_len,         grid_dim)
            - **attri_logits** (bsize, tgt_len, sr_dim, grid_dim)
            - **att_ctx**  (bsize, tlen, src_dim (+ emb_dim))
            - **att_wei**  (bsize, tlen, slen)
        """

        rnn_outs, curr_fgfs, prev_bgfs, what_ctx = what_states
        enc_rfts, enc_embs, enc_msks, enc_hids = encoder_states

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
                # print('encoder_feats ', encoder_feats.size())
                # encoder_msks = encoder_states['msks']
                # print('encoder_msks ', encoder_msks.size())
                if self.cfg.where_attn == 1:
                    query = curr_fgfs
                else:
                    query = torch.cat([curr_fgfs, what_ctx], -1)
                # print('query', query.size())    
                att_ctx, att_wei = self.attention(query, encoder_feats, enc_msks)
            
            bsize, tlen, att_dim = att_ctx.size()
            # Replicate  
            ctx_2d = att_ctx.view(bsize, tlen, att_dim, 1, 1)
            ctx_2d = ctx_2d.expand(bsize, tlen, att_dim, gh, gw)
            # print('ctx_2d ', ctx_2d.size())

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

                # print('attn_map', attn_map.size())

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
        # print('fg_2d ', fg_2d.size())

        #################################################################
        # Concatenate with fg features
        #################################################################
        if self.cfg.what_attn:
            combined = torch.cat([attn_rnn_outs, fg_2d, ctx_2d], dim=2)
        else:
            combined = torch.cat([attn_rnn_outs, fg_2d], dim=2)
        # combined = torch.cat([attn_rnn_outs, fg_2d, ctx_2d], dim=2)
        if self.cfg.use_bg_to_locate:
            combined = torch.cat([combined, prev_bgfs], dim=2)
        # print('combined ', combined.size())

        #################################################################
        # Classifer
        #################################################################
        bsize, tlen, fsize, gh, gw = combined.size()
        combined = combined.view(bsize * tlen, fsize, gh, gw)
        logits = self.decoder(combined)
        nsize, fsize, gh, gw = logits.size()
        assert(nsize == bsize * tlen)
        logits = logits.view(bsize, tlen, fsize, gh, gw)
        # print('logits ', logits.size())

        #################################################################
        # Reshape the outputs
        #################################################################
        logits = logits.view(bsize, tlen, fsize, -1)
        coord_logits = logits[:,:,0,:]
        scale_logits = logits[:,:,1:(self.cfg.num_scales+1), :]
        ratio_logits = logits[:,:,(self.cfg.num_scales+1):, :]

        coord_logits = F.softmax(coord_logits, dim=-1)
        scale_logits = F.softmax(scale_logits, dim=-2)
        ratio_logits = F.softmax(ratio_logits, dim=-2)

        attri_logits  = torch.cat([scale_logits, ratio_logits], 2)

        # print('coord_logits ', coord_logits.size())
        # print('scale_logits ', scale_logits.size())
        # print('ratio_logits ', ratio_logits.size())
        # print('attri_logits ', attri_logits.size())

        # where_outs = {}
        # where_outs['coord_logits'] = coord_logits
        # where_outs['attri_logits'] = attri_logits
        # if self.cfg.where_attn > 0:
        #     where_outs['attn_ctx'] = att_ctx
        #     where_outs['attn_wei'] = att_wei

        return coord_logits, attri_logits, att_ctx, att_wei

