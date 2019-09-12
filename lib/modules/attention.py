#!/usr/bin/env python
# codes modified from
#   https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
# and
#   https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py


import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Inputs: h_t, h_s, m_s
        - **h_t** (batch, tgt_len, tgt_dim):
            tensor containing output hidden features from the decoder.
        - **h_s** (batch, src_len, src_dim):
            tensor containing output hidden features from the encoder.
        - **m_s** (batch, src_len):
            tensor containing the padding mask of the encoded input sequence.

    Outputs: context, scores
        - **context** (batch, tgt_len, src_dim):
            tensor containing the attended output features
        - **scores** (batch, tgt_len, src_len):
            tensor containing attention weights.

    Examples::

         >>> attention = Attention('general', 256, 128)
         >>> h_s = torch.randn(5, 3, 256)
         >>> h_t = torch.randn(5, 5, 128)
         >>> m_s = torch.randn(5, 3).random_(0, 2)
         >>> context, scores = attention(h_t, h_s, m_s)

    """
    def __init__(self, attn_type, src_dim, tgt_dim):
        super(Attention, self).__init__()

        self.attn_type = attn_type.lower()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        # Luong Attention
        if self.attn_type == 'general':
            # general: :math:`score(H_j, q) = H_j^T W_a q`
            self.linear_in = nn.Linear(src_dim, tgt_dim, bias=False)
        # Bahdanau Attention
        elif self.attn_type == 'mlp':
            # mlp: :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
            self.linear_context = nn.Linear(src_dim, tgt_dim, bias=False)
            self.linear_query = nn.Linear(tgt_dim, tgt_dim, bias=True)
            self.v_a = nn.Linear(tgt_dim, 1, bias=False)
        elif self.attn_type == 'dot':
            # dot product scoring requires the encoder and decoder hidden states have the same dimension
            assert(src_dim == tgt_dim)
        else:
            raise ValueError("Unsupported attention mechanism: {0}".format(attn_type))

        # mlp wants it with bias
        # out_bias = self.attn_type == "mlp"
        # self.linear_out = nn.Linear(src_dim+tgt_dim, tgt_dim, bias=out_bias)

        # self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def score(self, h_t, h_s):
        # h_t: (batch, tgt_len, tgt_dim)
        # h_s: (batch, src_len, src_dim)
        # scores: (batch, tgt_len, src_len)

        if self.attn_type == 'dot':
            scores = torch.bmm(h_t, h_s.transpose(1, 2))
        elif self.attn_type == 'general':
            energy = self.linear_in(h_s)
            scores = torch.bmm(h_t, energy.transpose(1, 2))
        elif self.attn_type == 'mlp':
            src_batch, src_len, src_dim = h_s.size()
            tgt_batch, tgt_len, tgt_dim = h_t.size()
            assert(src_batch == tgt_batch)

            wq = self.linear_query(h_t)
            wq = wq.view(tgt_batch, tgt_len, 1, tgt_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, tgt_dim)

            uh = self.linear_context(h_s)
            uh = uh.view(src_batch, 1, src_len, tgt_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, tgt_dim)

            wquh = self.tanh(wq + uh)

            scores = self.v_a(wquh).view(src_batch, tgt_len, src_len)

        return scores

    def forward(self, h_t, h_s, m_s):
        src_batch, src_len, src_dim = h_s.size()
        scores = self.score(h_t, h_s)
        src_mask = m_s.view(src_batch, 1, src_len)
        scores = scores * src_mask
        scores = scores - 1e11 * (1.0 - src_mask)
        scores = self.softmax(scores.clamp(min=-1e10))
        # (batch, tgt_len, src_len) * (batch, src_len, src_dim) -> (batch, tgt_len, src_dim)
        context = torch.bmm(scores, h_s)
        # concat -> (batch, tgt_len, src_dim+tgt_dim)
        # combined = torch.cat((context, h_t), dim=-1)
        # # output -> (batch, tgt_len, tgt_dim)
        # output = self.linear_out(combined)
        # if self.attn_type in ["general", "dot"]:
        #     output = self.tanh(output)

        return context, scores
