#!/usr/bin/env python

import math, cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from layout_utils import conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class TextEncoder(nn.Module):
    def __init__(self, db):
        super(TextEncoder, self).__init__()
        self.db = db
        self.cfg = db.cfg
        self.embedding = nn.Embedding(self.cfg.input_vocab_size, self.cfg.n_embed)
        if self.cfg.emb_dropout_p > 0:
            self.embedding_dropout = nn.Dropout(p=self.cfg.emb_dropout_p)

        rnn_cell = self.cfg.rnn_cell.lower()
        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.rnn = self.rnn_cell(self.cfg.n_embed, self.cfg.n_src_hidden, 
            self.cfg.n_rnn_layers, batch_first=True, 
            bidirectional=self.cfg.bidirectional, 
            dropout=self.cfg.rnn_dropout_p)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

        self.embedding.weight.data.copy_(self.db.lang_vocab.vectors)

    def init_hidden(self, bsize):
        num_layers = self.cfg.n_rnn_layers
        hsize = self.cfg.n_src_hidden
        num_directions = 2 if self.cfg.bidirectional else 1

        hs = torch.zeros(num_layers * num_directions, bsize, hsize)
        if self.cfg.cuda:
            hs = hs.cuda()

        if self.cfg.rnn_cell.lower() == 'lstm':
            cs = torch.zeros(num_layers * num_directions, bsize, hsize)
            if self.cfg.cuda:
                cs = cs.cuda()
            return (hs, cs)
        
        return hs

    def forward(self, input_inds, input_lens):
        """
        Args:
            - **input_inds**  (bsize, slen) or (bsize, 3, slen)
            - **input_msks**  (bsize, slen) or (bsize, 3, slen)
        Returns: dict containing
            - **output_feats**   (bsize, tlen, hsize)
            - **output_embed**   (bsize, tlen, esize)
            - **output_msks**    (bsize, tlen)
            - **output_hiddens** [list of](num_layers * num_directions, bsize, hsize)
        """
        bsize, slen = input_inds.size()
        out_embs, out_rfts, out_msks = [], [], []
        out_hids, out_cels = [], []

        factor = 2 if self.cfg.bidirectional else 1
        hsize  = factor * self.cfg.n_src_hidden
        pad_rft = torch.zeros(1, 1, hsize)
        pad_emb = torch.zeros(1, 1, self.cfg.n_embed)
        if self.cfg.cuda:
            pad_rft = pad_rft.cuda()
            pad_emb = pad_emb.cuda()

        for i in range(bsize):
            curr_len  = input_lens[i].data.item()
            curr_inds = input_inds[i].view(-1)
            curr_inds = curr_inds[:curr_len]
            curr_inds = curr_inds.view(1, curr_len)

            inst_embs = self.embedding(curr_inds)
            if self.cfg.emb_dropout_p > 0:
                inst_embs = self.embedding_dropout(inst_embs)
            inst_rfts, inst_hids = self.rnn(inst_embs)
            tlen  = inst_rfts.size(1)
            n_pad = slen - tlen

            # Pad mask
            inst_msks = [1.0] * tlen
            if n_pad > 0:
                inst_msks = inst_msks + [0.0] * n_pad
            inst_msks = np.array(inst_msks)
            inst_msks = torch.from_numpy(inst_msks).float()
            if self.cfg.cuda:
                inst_msks = inst_msks.cuda()

            if n_pad > 0:
                # Pad features 
                inst_rfts = torch.cat([inst_rfts, pad_rft.expand(1, n_pad, hsize)], 1)
                inst_embs = torch.cat([inst_embs, pad_emb.expand(1, n_pad, self.cfg.n_embed)], 1)
            

            out_msks.append(inst_msks)
            out_rfts.append(inst_rfts)
            out_embs.append(inst_embs)

            # Average hiddens
            if isinstance(inst_hids, tuple):
                hs = inst_hids[0]
                cs = inst_hids[1]
                # print('hs.size(), cs.size(): ', hs.size(), cs.size())
                out_hids.append(hs)
                out_cels.append(cs)
            else:
                hs = inst_hids
                out_hids.append(hs)

        out_rfts = torch.cat(out_rfts, 0).contiguous()
        out_embs = torch.cat(out_embs, 0).contiguous()
        out_hids = torch.cat(out_hids, 1).contiguous()
        out_msks = torch.stack(out_msks, 0).contiguous()

        # print('out_rfts: ', out_rfts.size())
        # print('out_embs: ', out_embs.size())
        # print('out_hids: ', out_hids.size())
        # print('out_msks: ', out_msks.size())

        out = {}
        out['rfts'] = out_rfts
        out['embs'] = out_embs
        out['msks'] = out_msks

        if len(out_cels) > 0:
            out_cels = torch.cat(out_cels, 1).contiguous()
            out['hids'] = (out_hids, out_cels)
        else:
            out['hids'] = out_hids
    
        return out['rfts'], out['embs'], out['msks'], out['hids']


class VolumeEncoder(nn.Module):
    def __init__(self, config):
        self.inplanes = 128
        super(VolumeEncoder, self).__init__()
        self.conv1 = nn.Conv2d(83, 128, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 128, 1)
        # self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 256, 1, stride=2)
        self.upsample = nn.Upsample(size=(config.grid_size[1], config.grid_size[0]), mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, stack_vols):
        bsize, slen, fsize, height, width = stack_vols.size()
        inputs = stack_vols.view(bsize * slen, fsize, height, width)

        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.upsample(x)
        
        nsize, fsize, gh, gw = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize, gh, gw)

        return x

