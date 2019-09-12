#!/usr/bin/env python
# codes modified from
#   https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import os, sys, cv2, json
import math, copy, random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from modules.separable_convolution import separable_conv2d


USE_SEPARABLE_CONVOLUTION = False


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()

        self.hidden_size = hidden_size

        if USE_SEPARABLE_CONVOLUTION:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        self.conv1 = conv2d(
            in_channels=(input_size + hidden_size),
            out_channels=(2 * hidden_size),
            kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2),
            bias=bias)

        self.conv2 = conv2d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2),
            bias=bias)

        self.conv3 = conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2),
            bias=bias)


    def forward(self, x, h):
        # x: [batch, input_size, height, width]
        # h: [batch, hidden_size, height, width]
        combined = torch.cat((x, h), dim=1)
        A = self.conv1(combined)
        (az, ar) = torch.split(A, self.hidden_size, dim=1)
        z = torch.sigmoid(az)
        r = torch.sigmoid(ar)

        ag = self.conv2(x) + r * self.conv3(h)
        g = torch.tanh(ag)

        new_h = z * h + (1.0 - z) * g
        return new_h


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        if USE_SEPARABLE_CONVOLUTION:
            conv2d = separable_conv2d
        else:
            conv2d = nn.Conv2d

        self.conv = conv2d(
            in_channels=(input_size + hidden_size),
            out_channels=(4 * hidden_size),
            kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2),
            bias=bias)


    def forward(self, x, h, c):
        # x: [batch, input_size, height, width]
        # h, c: [batch, hidden_size, height, width]
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.hidden_size, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, bias=True, dropout=0.0):
        super(ConvGRU, self).__init__()

        self.input_sizes = [input_size] + [hidden_size] * (num_layers-1)
        self.hidden_sizes = [hidden_size] * num_layers
        self.num_layers = num_layers
        self.dropout_p = dropout

        for i in range(num_layers):
            cell = ConvGRUCell(self.input_sizes[i], self.hidden_sizes[i], kernel_size, bias)
            setattr(self, 'cell%02d'%i, cell)

        if self.dropout_p > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_p)

        self.init_weights()

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_var, prev_hidden):
        # Inputs
        #   input_var: (#batch, #sequence, #input_size, #height, #width)
        #   prev_hidden: (#layers, #batch, #hidden_size, #height, #width)
        # Outputs
        #   last_layer_hiddens: (#batch, #sequence, #hidden_size, #height, #width)
        #   last_step_hiddens: (#layers, #batch, #hidden_size, #height, #width)
        #   all_hiddens: (#layers, #batch, #sequence, #hidden_size, #height, #width)


        all_hiddens_list = []
        current_layer_input = input_var
        for layer in range(self.num_layers):
            layer_output_list = []
            h = prev_hidden[layer]
            for step in range(current_layer_input.size(1)):
                x = current_layer_input[:, step, :, :, :]
                h = getattr(self, 'cell%02d'%layer)(x, h)
                if self.dropout_p > 0:
                    h = self.dropout(h)
                layer_output_list.append(h)
            layer_output = torch.stack(layer_output_list, dim=1)
            current_layer_input = layer_output
            all_hiddens_list.append(layer_output)
        last_layer_hiddens = all_hiddens_list[-1]
        all_hiddens = torch.stack(all_hiddens_list, dim=0)
        last_step_hiddens = all_hiddens[:, :, -1, :, :, :]

        return last_layer_hiddens, last_step_hiddens, all_hiddens


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, bias=True, dropout=0.0):
        super(ConvLSTM, self).__init__()

        self.input_sizes = [input_size] + [hidden_size] * (num_layers-1)
        self.hidden_sizes = [hidden_size] * num_layers
        self.num_layers = num_layers
        self.dropout_p = dropout

        for i in range(num_layers):
            cell = ConvLSTMCell(self.input_sizes[i], self.hidden_sizes[i], kernel_size, bias)
            setattr(self, 'cell%02d'%i, cell)

        if self.dropout_p > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_p)

        self.init_weights()

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_var, prev_hidden):
        # Inputs
        #   input_var: (#batch, #sequence, #input_size, #height, #width)
        #   prev_hidden: tuple of (#layers, #batch, #hidden_size, #height, #width)
        # Outputs
        #   last_layer_hiddens: (#batch, #sequence, #hidden_size, #height, #width)
        #   last_step_*: tuple (#layers, #batch, #hidden_size, #height, #width)
        #   all_*: tuple (#layers, #batch, #sequence, #hidden_size, #height, #width)

        all_hiddens_list = []
        all_cells_list = []
        current_layer_input = input_var
        (prev_h, prev_c) = prev_hidden
        for layer in range(self.num_layers):
            layer_hidden_list = []
            layer_cell_list = []
            h = prev_h[layer]
            c = prev_c[layer]
            for step in range(current_layer_input.size(1)):
                x = current_layer_input[:, step, :, :, :]
                h, c = getattr(self, 'cell%02d'%layer)(x, h, c)
                if self.dropout_p > 0:
                    h = self.dropout(h)
                    c = self.dropout(c)
                layer_hidden_list.append(h)
                layer_cell_list.append(c)
            layer_hidden = torch.stack(layer_hidden_list, dim=1)
            layer_cell = torch.stack(layer_cell_list, dim=1)
            current_layer_input = layer_hidden
            all_hiddens_list.append(layer_hidden)
            all_cells_list.append(layer_cell)

        last_layer_hiddens = all_hiddens_list[-1]
        all_hiddens = torch.stack(all_hiddens_list, dim=0)
        last_step_hiddens = all_hiddens[:, :, -1, :, :, :]

        all_cells = torch.stack(all_cells_list, dim=0)
        last_step_cells = all_cells[:, :, -1, :, :, :]

        return last_layer_hiddens, (last_step_hiddens, last_step_cells), (all_hiddens, all_cells)
