#!/usr/bin/env python

import math, cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from modules.separable_convolution import same_padding_size
from modules.separable_convolution import separable_conv2d


from composites_utils import *


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.cfg = config

        original_model = models.resnet152(pretrained=True)
        self.conv1   = original_model.conv1
        self.bn1     = original_model.bn1
        self.relu    = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1  = original_model.layer1
        self.layer2  = original_model.layer2
        self.layer3  = original_model.layer3
        self.layer4  = original_model.layer4
        self.avgpool = original_model.avgpool

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, db):
        super(TextEncoder, self).__init__()
        self.db = db
        self.cfg = db.cfg

        # Embedding module
        self.embedding = nn.Embedding(self.cfg.input_vocab_size, self.cfg.n_embed)
        if self.cfg.emb_dropout_p > 0:
            self.embedding_dropout = nn.Dropout(p=self.cfg.emb_dropout_p)

        # RNN type
        rnn_cell = self.cfg.rnn_cell.lower()
        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        # RNN module
        self.rnn = self.rnn_cell(self.cfg.n_embed, self.cfg.n_src_hidden,
            self.cfg.n_rnn_layers, batch_first=True,
            bidirectional=self.cfg.bidirectional,
            dropout=self.cfg.rnn_dropout_p)

        # Weight initialization
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
        vhs = Variable(hs)

        if self.cfg.rnn_cell.lower() == 'lstm':
            cs = torch.zeros(num_layers * num_directions, bsize, hsize)
            if self.cfg.cuda:
                cs = cs.cuda()
            vcs = Variable(cs)
            return (vhs, vcs)

        return vhs

    def forward(self, input_inds, input_lens):
        """
        Args:
            - **input_inds**  (bsize, slen)
            - **input_lens**  (bsize, )

        Returns: dict containing
            - **output_feats**   (bsize, tlen, hsize)
            - **output_embed**   (bsize, tlen, esize)
            - **output_msks**    (bsize, tlen)
            - **output_hiddens** [list of](num_layers * num_directions, bsize, hsize)
        """
        out_embeddings = []
        out_rnn_features, out_rnn_masks = [], []
        out_hidden_states, out_cell_states = [], []

        bsize, slen = input_inds.size()
        factor = 2 if self.cfg.bidirectional else 1
        hsize  = factor * self.cfg.n_src_hidden
        pad_rnn_features = torch.zeros(1, 1, hsize)
        pad_embedding = torch.zeros(1, 1, self.cfg.n_embed)
        if self.cfg.cuda:
            pad_rnn_features = pad_rnn_features.cuda()
            pad_embedding = pad_embedding.cuda()

        for i in range(bsize):
            curr_len  = input_lens[i].data.item()
            curr_inds = input_inds[i].view(-1)
            curr_inds = curr_inds[:curr_len]
            curr_inds = curr_inds.view(1, curr_len)

            curr_embedding = self.embedding(curr_inds)
            if self.cfg.emb_dropout_p > 0:
                curr_embedding = self.embedding_dropout(curr_embedding)
            self.rnn.flatten_parameters()
            curr_rnn_features, curr_hiddens = self.rnn(curr_embedding)
            tlen  = curr_rnn_features.size(1)
            n_pad = slen - tlen

            # Pad mask
            curr_rnn_mask = [1.0] * tlen
            if n_pad > 0:
                curr_rnn_mask = curr_rnn_mask + [0.0] * n_pad
            curr_rnn_mask = np.array(curr_rnn_mask)
            curr_rnn_mask = torch.from_numpy(curr_rnn_mask).float()
            if self.cfg.cuda:
                curr_rnn_mask = curr_rnn_mask.cuda()

            if n_pad > 0:
                # Pad features
                curr_rnn_features = torch.cat([curr_rnn_features, pad_rnn_features.expand(1, n_pad, hsize)], 1)
                curr_embedding = torch.cat([curr_embedding, pad_embedding.expand(1, n_pad, self.cfg.n_embed)], 1)


            out_rnn_masks.append(curr_rnn_mask)
            out_rnn_features.append(curr_rnn_features)
            out_embeddings.append(curr_embedding)

            if isinstance(curr_hiddens, tuple):
                hs = curr_hiddens[0]
                cs = curr_hiddens[1]
                out_hidden_states.append(hs)
                out_cell_states.append(cs)
            else:
                hs = curr_hiddens
                out_hidden_states.append(hs)

        out_embeddings = torch.cat(out_embeddings, 0).contiguous()
        out_rnn_features = torch.cat(out_rnn_features, 0).contiguous()
        out_rnn_masks = torch.stack(out_rnn_masks, 0).contiguous()
        out_hidden_states = torch.cat(out_hidden_states, 1).contiguous()

        out = {}
        out['embeddings'] = out_embeddings
        out['rnn_features'] = out_rnn_features
        out['rnn_masks'] = out_rnn_masks

        if len(out_cell_states) > 0:
            out_cell_states = torch.cat(out_cell_states, 1).contiguous()
            out['rnn_hiddens'] = (out_hidden_states, out_cell_states)
        else:
            out['rnn_hiddens'] = out_hidden_states

        return out['embeddings'], out['rnn_features'], out['rnn_masks'], out['rnn_hiddens']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        p = same_padding_size(3, 2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                         padding=p, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None, use_normalization=False, normalization_res=None):
        super(SimpleBlock, self).__init__()
        self.use_normalization = use_normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.stride = stride
        self.downsample = downsample
        if use_normalization:
            self.norm1 = nn.LayerNorm([out_channels, normalization_res[0]//stride, normalization_res[1]//stride])
            self.norm2 = nn.LayerNorm([out_channels, normalization_res[0]//stride, normalization_res[1]//stride])

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_normalization:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.use_normalization:
            out = self.norm2(out)
        out = self.activation(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.cfg = config

        original_model = models.resnet50(pretrained=True)
        self.conv1   = original_model.conv1
        self.bn1     = original_model.bn1
        self.relu    = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1  = original_model.layer1
        self.layer2  = original_model.layer2
        self.layer3  = original_model.layer3
        # self.layer3  = self._make_layer(BasicBlock, 256, 2, stride=1)
        # self.layer3.load_state_dict(original_model.layer3.state_dict())
        self.layer4  = original_model.layer4
        self.upsample = nn.Upsample(size=(self.cfg.grid_size[1], self.cfg.grid_size[0]), mode='bilinear', align_corners=True)

        # for param in self.parameters():
        #     param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(128, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(128, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, stack_imgs):
        if self.cfg.finetune_lr == 0:
            self.eval()
        # if self.cfg.teacher_forcing:
        #     bsize, slen, fsize, height, width = stack_imgs.size()
        #     inputs = stack_imgs.view(bsize * slen, fsize, height, width)
        # else:
        #     bsize, fsize, height, width = stack_imgs.size()
        #     slen = 1
        #     inputs = stack_imgs
        bsize, slen, fsize, height, width = stack_imgs.size()
        inputs = stack_imgs.view(bsize * slen, fsize, height, width)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.upsample(x)
        nsize, fsize, gh, gw = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize, gh, gw)
        return x


class VolumeEncoder(nn.Module):
    def __init__(self, config):
        super(VolumeEncoder, self).__init__()
        self.cfg = config

        if self.cfg.use_color_volume:
            if self.cfg.use_normalization:
                self.block1 = nn.Sequential(
                    nn.Conv2d(3 * self.cfg.output_vocab_size, self.cfg.output_vocab_size,
                        kernel_size=7, stride=2, padding=3, groups=self.cfg.output_vocab_size),
                    nn.LayerNorm([self.cfg.output_vocab_size, self.cfg.input_image_size[0]//2, self.cfg.input_image_size[1]//2]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                self.block1 = nn.Sequential(
                    nn.Conv2d(3 * self.cfg.output_vocab_size, self.cfg.output_vocab_size,
                        kernel_size=7, stride=2, padding=3, groups=self.cfg.output_vocab_size),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.block2 = self._make_layer(SimpleBlock, self.cfg.output_vocab_size, 2*self.cfg.output_vocab_size, 1)
        else:
            if self.cfg.use_normalization:
                self.block1 = nn.Sequential(
                    nn.Conv2d(self.cfg.output_vocab_size+4, 256, kernel_size=7, stride=2, padding=3),
                    nn.LayerNorm([256, self.cfg.input_image_size[0]//2, self.cfg.input_image_size[1]//2]),
                    nn.LeakyReLU(0.2, inplace=True)
                ) 
            else:
                self.block1 = nn.Sequential(
                    nn.Conv2d(self.cfg.output_vocab_size+4, 256, kernel_size=7, stride=2, padding=3),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            self.block2 = self._make_layer(SimpleBlock, 256, 2*self.cfg.output_vocab_size, 1)
        self.block3 = self._make_layer(SimpleBlock, 2*self.cfg.output_vocab_size, 3*self.cfg.output_vocab_size, 2)
        self.block4 = self._make_layer(SimpleBlock, 3*self.cfg.output_vocab_size, 4*self.cfg.output_vocab_size, 1)
        # self.block5 = self._make_layer(SimpleBlock, 4*self.cfg.output_vocab_size, 5*self.cfg.output_vocab_size, 1)

    def _make_layer(self, block, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        layers = []
        # layers.append(block(in_channels,  out_channels, stride, downsample))
        # layers.append(block(out_channels, out_channels, 1, None))
        layers.append(block(in_channels, in_channels, 1, None))
        layers.append(block(in_channels, out_channels, stride, downsample))

        return nn.Sequential(*layers)

    def inference(self, color_vols):
        stack_vols = (color_vols.float()-128.0).permute(0,1,4,2,3)

        bsize, slen, fsize, height, width = stack_vols.size()
        inputs = stack_vols.view(bsize * slen, fsize, height, width)

        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        nsize, fsize, gh, gw = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize, gh, gw)
        return x

    def forward(self, stack_vols):
        if self.cfg.use_color_volume:
            stack_vols = sequence_color_volumn_preprocess(stack_vols, self.cfg.output_vocab_size)
        else:
            stack_vols = sequence_onehot_volumn_preprocess(stack_vols, self.cfg.output_vocab_size)
        stack_vols = (stack_vols-128.0).permute(0,1,4,2,3)

        bsize, slen, fsize, height, width = stack_vols.size()
        inputs = stack_vols.view(bsize * slen, fsize, height, width)

        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        nsize, fsize, gh, gw = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize, gh, gw)
        return x


class ShapeEncoder(nn.Module):
    def __init__(self, config):
        super(ShapeEncoder, self).__init__()
        self.cfg = config

        if self.cfg.use_separable_convolution:
            conv2d = nn.Conv2d
        else:
            conv2d = separable_conv2d

        hidden_size = self.cfg.n_shape_hidden
        if self.cfg.use_normalization:
            input_res = self.cfg.input_patch_size
            self.main = nn.Sequential(
                nn.Conv2d(4 + self.cfg.output_vocab_size, hidden_size,
                    kernel_size=2, stride=2, padding=0),
                nn.LayerNorm([hidden_size, input_res[0]//2, input_res[1]//2]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LayerNorm([hidden_size, input_res[0]//4, input_res[1]//4]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LayerNorm([hidden_size, input_res[0]//8, input_res[1]//8]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LayerNorm([hidden_size, input_res[0]//16, input_res[1]//16]),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LayerNorm([hidden_size, input_res[0]//32, input_res[1]//32]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(4 + self.cfg.output_vocab_size, hidden_size,
                    kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )

        in_dim = self.cfg.n_shape_hidden
        if self.cfg.use_resnet:
            in_dim = self.cfg.n_shape_hidden + 2048

        self.merge = nn.Sequential(
            nn.Linear(in_dim, self.cfg.n_shape_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.cfg.n_shape_hidden, self.cfg.n_patch_features))

    def batch_forward(self, input_vols, input_features):
        vols = batch_onehot_volumn_preprocess(input_vols, self.cfg.output_vocab_size)
        vols = (vols-128.0).permute(0,3,1,2)
        x = self.main(vols).squeeze()
        if self.cfg.use_resnet: 
            x = torch.cat((x, input_features), -1)
        x = self.merge(x)
        x = F.normalize(x, dim=-1)
        return x

    def forward(self, stack_vols, stack_features):
        stack_vols = sequence_onehot_volumn_preprocess(stack_vols, self.cfg.output_vocab_size)
        stack_vols = (stack_vols-128.0).permute(0,1,4,2,3)

        bsize, slen, fsize, height, width = stack_vols.size()
        inputs = stack_vols.view(bsize * slen, fsize, height, width)

        x = self.main(inputs).squeeze()
        nsize, fsize = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize)
        if self.cfg.use_resnet:
            x = torch.cat((x, stack_features), -1)
        x = self.merge(x)
        x = F.normalize(x, dim=-1)
        return x


class SynthesisEncoder(nn.Module):
    def __init__(self, config):
        super(SynthesisEncoder, self).__init__()
        self.cfg = config
        h, w = config.output_image_size

        # if self.cfg.use_color_volume:
        #     in_channels = 3 * config.output_vocab_size
        # else:
        #     in_channels = config.output_vocab_size + 4
        in_channels = config.output_vocab_size + 4

        self.cfg = config
        self.block1 = nn.Sequential(self.make_layers(in_channels, [256, 256], config.use_normalization, [h,     w]))
        self.block2 = nn.Sequential(self.make_layers(256, [256, 256],  config.use_normalization, [h//2,  w//2]))
        self.block3 = nn.Sequential(self.make_layers(256, [256, 256, 256],  config.use_normalization, [h//4,  w//4]))
        self.block4 = nn.Sequential(self.make_layers(256, [512, 512, 512],  config.use_normalization, [h//8,  w//8]))
        self.block5 = nn.Sequential(self.make_layers(512, [512, 512, 512],  config.use_normalization, [h//16, w//16]))
        self.block6 = nn.Sequential(self.make_layers(512, [512, 512, 512],  config.use_normalization, [h//32, w//32]))

    def make_layers(self, inplace, places, use_layer_norm=False, resolution=None):
        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        layers = []
        in_channels = inplace

        for v in places:
            if use_layer_norm:
                current_conv2d = conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                current_lnorm = nn.LayerNorm([v, resolution[0], resolution[1]])
                layers.extend([current_conv2d, current_lnorm])
            else:
                current_conv2d = conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                layers.append(current_conv2d)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = v
        layers.append(nn.AvgPool2d(3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, inputs):

        # if self.cfg.use_color_volume:
        #     x0 = batch_color_volumn_preprocess(inputs, self.cfg.output_vocab_size)
        # else:
        #     x0 = batch_onehot_volumn_preprocess(inputs, self.cfg.output_vocab_size)
        x0 = batch_onehot_volumn_preprocess(inputs, self.cfg.output_vocab_size)
        x0 = (x0-128.0).permute(0,3,1,2)
        h, w = self.cfg.output_image_size

        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        return (x0, x1, x2, x3, x4, x5, x6)


sp = 256
in_channels = 24
class ImageAndLayoutEncoder(nn.Module):
    def __init__(self, config):
        super(ImageAndLayoutEncoder, self).__init__()
        self.cfg = config

        self.block1 = nn.Sequential(self.make_layers(in_channels, [64, 64], True, [sp,     2*sp]))
        self.block2 = nn.Sequential(self.make_layers(64,  [128, 128],       True, [sp//2,  sp]))
        self.block3 = nn.Sequential(self.make_layers(128, [256, 256, 256],  True, [sp//4,  sp//2]))
        self.block4 = nn.Sequential(self.make_layers(256, [512, 512, 512],  True, [sp//8,  sp//4]))
        self.block5 = nn.Sequential(self.make_layers(512, [512, 512, 512],  True, [sp//16, sp//8]))
        self.block6 = nn.Sequential(self.make_layers(512, [512, 512, 512],  True, [sp//32, sp//16]))

    def make_layers(self, inplace, places, use_layer_norm=False, resolution=None):
        if self.cfg.use_separable_convolution:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        layers = []
        in_channels = inplace

        for v in places:
            if use_layer_norm:
                current_conv2d = conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                current_lnorm = nn.LayerNorm([v, resolution[0], resolution[1]])
                layers.extend([current_conv2d, current_lnorm])
            else:
                current_conv2d = conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                layers.append(current_conv2d)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = v
        layers.append(nn.AvgPool2d(3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, inputs):

        x0 = inputs #F.interpolate(inputs, size=[sp, 2*sp], mode='bilinear', align_corners=True)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        return (x0, x1, x2, x3, x4, x5, x6)
