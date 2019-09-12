#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import OrderedDict


from composites_utils import *
from composites_config import get_config

from datasets.composites_coco import composites_coco
from datasets.composites_loader import sequence_loader
from modules.attention import Attention
from modules.composites_encoder import TextEncoder, ImageEncoder
from modules.composites_encoder import VolumeEncoder, ShapeEncoder
from modules.composites_encoder import ImageAndLayoutEncoder
from modules.composites_decoder import WhatDecoder, WhereDecoder
from modules.puzzle_model import PuzzleModel
from modules.conv_rnn import ConvGRU, ConvLSTM

# from modules.perceptual_loss import VGG19LossNetwork
# from modules.crn_decoder import CRNDecoder
# from modules.image_synthesis_model import ImageSynthesisModel
from nntable import AllCategoriesTables

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def test_attention(config):
    attention = Attention(config.attn_type, 256, 128)
    h_s = torch.randn(5, 6, 256)
    h_t = torch.randn(5, 5, 128)
    src_mask = torch.randn(5, 6).random_(0, 2)
    context, scores = attention(h_t, h_s, src_mask)

    print('context.size()', context.size())
    print('scores.size()', scores.size())


def test_txt_encoder_coco(config):
    db = composites_coco(config, 'train', '2017')
    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)
    sequence_db = sequence_loader(db, all_tables)
    net = TextEncoder(db)

    loader = DataLoader(sequence_db,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        input_inds = batched['word_inds'].long()
        input_lens = batched['word_lens'].long()

        print('Checking the output shapes')
        out = net(input_inds, input_lens)
        out_embs, out_rfts, out_msks, out_hids = out
        print(out_rfts.size(), out_embs.size(), out_msks.size())
        if isinstance(out_hids, tuple):
            print(out_hids[0].size())
        else:
            print(out_hids.size())
        print('m: ', out_msks[-1])


        print('Checking the embedding')
        embeded = net.embedding(input_inds)
        v1 = embeded[0, 0]
        idx = input_inds[0,0].data.item()
        v2 = db.lang_vocab.vectors[idx]
        diff = v2 - v1
        print('Diff: (should be zero)', torch.sum(diff.abs_()))

        break


def test_img_encoder(config):
    # transformer = image_normalize('background')
    # db = coco(config, 'val', transform=transformer)
    # # pca_table = AllCategoriesTables(db)
    # # pca_table.run_PCAs_and_build_nntables_in_feature_space()

    img_encoder = ImageEncoder(config)
    print(get_n_params(img_encoder))

    # loader = DataLoader(db,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers)
    #
    # for cnt, batched in enumerate(loader):
    #     x = batched['background'].float()
    #     y = img_encoder(x)
    #     print('y.size()', y.size())
    #     break


def test_vol_encoder(config):
    db = composites_coco(config, 'train', '2017')

    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)
    sequence_db = sequence_loader(db, all_tables)

    img_encoder = VolumeEncoder(config)
    print(get_n_params(img_encoder))
    # print(img_encoder)

    loader = DataLoader(sequence_db,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        x = batched['background'].float()
        y = img_encoder(x)
        print('y.size()', y.size())
        break


def test_shape_encoder(config):
    db = coco(config, 'train', '2017')
    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)
    sequence_db = sequence_loader(db, all_tables)

    img_encoder = ShapeEncoder(config)
    print(get_n_params(img_encoder))

    loader = DataLoader(sequence_db,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        x = batched['foreground'].float()
        y = batched['foreground_resnets'].float()
        y = img_encoder(x, y)
        print('y.size()', y.size())
        print('y max', torch.max(y))
        print('y min', torch.min(y))
        print('y norm', torch.norm(y, dim=-1)[0,0])
        break


def test_conv_gru(config):
    net = ConvGRU(512, config.n_tgt_hidden, config.n_rnn_layers, 3, dropout=0.1)

    input_var = np.random.randn(2, 3, 512, 14, 14)
    input_msk = np.array([[1,1,0],[1,0,0]])
    prev_hidden = np.random.randn(config.n_rnn_layers, 2, config.n_tgt_hidden, 14, 14)

    input_var_th = torch.from_numpy(input_var).float()
    input_msk_th = torch.from_numpy(input_msk).long()
    prev_hidden_th = torch.from_numpy(prev_hidden).float()

    last_layer_hiddens, last_step_hiddens, all_hiddens = net(input_var_th, prev_hidden_th)
    print('last_layer_hiddens.size()', last_layer_hiddens.size())
    print('last_step_hiddens.size()', last_step_hiddens.size())
    print('all_hiddens.size()', all_hiddens.size())


def test_conv_lstm(config):
    net = ConvLSTM(512, config.n_tgt_hidden, config.n_rnn_layers, 3, dropout=0.1)

    input_var = np.random.randn(2, 3, 512, 14, 14)
    input_msk = np.array([[1,1,0],[1,0,0]])
    hs = np.random.randn(config.n_rnn_layers, 2, config.n_tgt_hidden, 14, 14)
    cs = np.random.randn(config.n_rnn_layers, 2, config.n_tgt_hidden, 14, 14)

    input_var_th = torch.from_numpy(input_var).float()
    input_msk_th = torch.from_numpy(input_msk).long()
    hs_th = torch.from_numpy(hs).float()
    cs_th = torch.from_numpy(cs).float()

    a, (b,c), (d,e) = net(input_var_th, (hs_th, cs_th))
    print(a.size(), b.size(), c.size(), d.size(), e.size())


def test_coco_decoder(config):
    db = composites_coco(config, 'train', '2017')
    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)
    sequence_db = sequence_loader(db, all_tables)

    text_encoder  = TextEncoder(db)
    img_encoder   = VolumeEncoder(config)
    what_decoder  = WhatDecoder(config)
    where_decoder = WhereDecoder(config)

    print('txt_encoder', get_n_params(text_encoder))
    print('img_encoder', get_n_params(img_encoder))
    print('what_decoder', get_n_params(what_decoder))
    print('where_decoder', get_n_params(where_decoder))

    loader = DataLoader(sequence_db,
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        word_inds = batched['word_inds'].long()
        word_lens = batched['word_lens'].long()
        bg_imgs   = batched['background'].float()

        encoder_states = text_encoder(word_inds, word_lens)
        bg_feats = img_encoder(bg_imgs)
        prev_bgfs = bg_feats[:,0].unsqueeze(1)
        what_outs = what_decoder((prev_bgfs, None, None), encoder_states)
        obj_logits, rnn_feats_2d, nxt_hids_2d, prev_bgfs, att_ctx, att_wei = what_outs
        print('------------------------------------------')
        print('obj_logits', obj_logits.size())
        print('rnn_feats_2d', rnn_feats_2d.size())
        print('nxt_hids_2d', nxt_hids_2d.size())
        print('prev_bgfs', prev_bgfs.size())
        print('att_ctx', att_ctx.size())
        print('att_wei', att_wei.size())
        print('------------------------------------------')

        _, obj_inds = torch.max(obj_logits + 1.0, dim=-1)
        curr_fgfs = indices2onehots(obj_inds.cpu().data, config.output_vocab_size)
        # curr_fgfs = curr_fgfs.unsqueeze(1)
        if config.cuda:
            curr_fgfs = curr_fgfs.cuda()

        where_outs = where_decoder((rnn_feats_2d, curr_fgfs, prev_bgfs, att_ctx), encoder_states)
        coord_logits, attri_logits, patch_vectors, where_ctx, where_wei = where_outs
        print('coord_logits ', coord_logits.size())
        print('attri_logits ', attri_logits.size())
        print('patch_vectors', patch_vectors.size())
        # print('att_ctx', where_ctx.size())
        # print('att_wei', where_wei.size())
        break


def test_perceptual_loss_network(config):
    transformer = image_normalize('background')
    db = coco(config, 'val', transform=transformer)

    img_encoder = VGG19LossNetwork(config).eval()
    print(get_n_params(img_encoder))

    loader = DataLoader(db,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        x = batched['background'].float()
        bsize, slen, c, h, w = x.size()
        x = x.view(bsize*slen, c, h, w)
        y = img_encoder(x)
        for z in y:
            print(z.size())
        # print(y)
        # print('y.size()', y.size())
        break


def test_syn_encoder(config):
    transformer = image_normalize('background')
    db = coco(config, 'val', transform=transformer)

    img_encoder = ImageAndLayoutEncoder(config)
    print(get_n_params(img_encoder))

    loader = DataLoader(db,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        x = batched['background'].float()
        bsize, slen, c, h, w = x.size()
        x = x.view(bsize*slen, c, h, w)
        x_fake = torch.randn(1, 24, 512, 1024)
        y = img_encoder(x_fake)
        for z in y:
            print(z.size())
        break


def test_syn_decoder(config):
    img_encoder = ImageAndLayoutEncoder(config)
    img_decoder = CRNDecoder(config)
    print(get_n_params(img_encoder))
    print(get_n_params(img_decoder))

    x_fake = torch.randn(1, 24, 512, 1024)
    x0, x1, x2, x3, x4, x5, x6 = img_encoder(x_fake)
    inputs = (x0, x1, x2, x3, x4, x5, x6)
    image, label = img_decoder(inputs)
    print(image.size(), label.size())


def test_syn_model(config):
    synthesizer = ImageSynthesisModel(config)
    print(get_n_params(synthesizer))

    x = torch.randn(1, 24, 512, 1024)
    image, label, features, _ = synthesizer(x, True)

    print(image.size(), label.size())
    for z in features:
        print(z.size())


def test_puzzle_model(config):
    output_dir = osp.join(config.model_dir, 'test_puzzle_model')
    maybe_create(output_dir)
    plt.switch_backend('agg')

    db = composites_coco(config, 'train', '2017')
    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)
    sequence_db = sequence_loader(db, all_tables)
    loader = DataLoader(sequence_db,
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers)

    net = PuzzleModel(db)

    net.eval()
    for cnt, batched in enumerate(loader):
        word_inds = batched['word_inds'].long()
        word_lens = batched['word_lens'].long()
        bg_images = batched['background'].float()
        fg_images = batched['foreground'].float()
        neg_images = batched['negative'].float()

        fg_resnets = batched['foreground_resnets'].float()
        neg_resnets = batched['negative_resnets'].float()

        fg_inds = batched['fg_inds'].long()
        gt_inds = batched['out_inds'].long()
        gt_msks = batched['out_msks'].float()

        fg_onehots = indices2onehots(fg_inds, config.output_vocab_size)

        inf_outs, _, positive_feats, negative_feats = net((word_inds, word_lens, bg_images, fg_onehots, fg_images, neg_images, fg_resnets, neg_resnets))
        obj_logits, coord_logits, attri_logits, patch_vectors, enc_msks, what_wei, where_wei = inf_outs
        print('teacher forcing')
        print('obj_logits ', obj_logits.size())
        print('coord_logits ', coord_logits.size())
        print('attri_logits ', attri_logits.size())
        print('patch_vectors ', patch_vectors.size())
        print('patch_vectors max:', torch.max(patch_vectors))
        print('patch_vectors min:', torch.min(patch_vectors))
        print('patch_vectors norm:', torch.norm(patch_vectors, dim=-2)[0,0,0])
        print('positive_feats ', positive_feats.size())
        print('negative_feats ', negative_feats.size())
        if config.what_attn:
            print('what_att_logits ', what_wei.size())
        if config.where_attn > 0:
            print('where_att_logits ', where_wei.size())
        print('----------------------')

        _, pred_vecs = net.collect_logits_and_vectors(inf_outs, gt_inds)
        print('pred_vecs', pred_vecs.size())
        print('*******************')

        # # inf_outs, env = net.inference(word_inds, word_lens, -1, 0.0, 0, gt_inds, gt_vecs)
        # inf_outs, env = net.inference(word_inds, word_lens, -1, 2.0, 0, None, None, all_tables)
        # obj_logits, coord_logits, attri_logits, patch_vectors, enc_msks, what_wei, where_wei = inf_outs
        # print('scheduled sampling')
        # print('obj_logits ', obj_logits.size())
        # print('coord_logits ', coord_logits.size())
        # print('attri_logits ', attri_logits.size())
        # print('patch_vectors ', patch_vectors.size())
        # if config.what_attn:
        #     print('what_att_logits ', what_wei.size())
        # if config.where_attn > 0:
        #     print('where_att_logits ', where_wei.size())
        # print('----------------------')
        
        
        # sequences = env.batch_redraw(True)
        # for i in range(len(sequences)):
        #     sequence = sequences[i]
        #     image_idx = batched['image_index'][i]
        #     name = '%03d_'%i + str(image_idx).zfill(12)
        #     out_path = osp.join(output_dir, name+'.png')
        #     color = cv2.imread(batched['color_path'][i], cv2.IMREAD_COLOR)
        #     color, _, _ = create_squared_image(color)
        
        #     fig = plt.figure(figsize=(32, 16))
        #     plt.suptitle(batched['sentence'][i], fontsize=30)
        
        #     for j in range(min(len(sequence), 14)):
        #         plt.subplot(3, 5, j+1)
        #         partially_completed_img = clamp_array(sequence[j][:,:,-3:], 0, 255).astype(np.uint8)
        #         partially_completed_img = partially_completed_img[:,:,::-1]
        #         plt.imshow(partially_completed_img)
        #         plt.axis('off')
        
        #     plt.subplot(3, 5, 15)
        #     plt.imshow(color[:,:,::-1])
        #     plt.axis('off')
        
        #     fig.savefig(out_path, bbox_inches='tight')
        #     plt.close(fig)

        break


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # test_attention(config)
    # test_conv_gru(config)
    # test_conv_lstm(config)
    # test_txt_encoder_coco(config)
    # test_img_encoder(config)
    # test_vol_encoder(config)
    # test_shape_encoder(config)
    # test_coco_decoder(config)
    test_puzzle_model(config)
    # test_perceptual_loss_network(config)
    # test_syn_encoder(config)
    # test_syn_decoder(config)
    # test_syn_model(config)
