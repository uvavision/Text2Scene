#!/usr/bin/env python

import _init_paths
import math, cv2, random
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

from modules.layout_encoder import TextEncoder, VolumeEncoder
from modules.layout_decoder import WhatDecoder, WhereDecoder
from modules.layout_evaluator import evaluator, eval_info, scene_graph
from modules.layout_simulator import simulator
from modules.layout_model import DrawModel

from datasets.layout_coco import layout_coco
from layout_utils import *
from layout_config import get_config


def test_txt_encoder_coco(config):
    transformer = volume_normalize('background')
    db = layout_coco(config, 'train', transform=transformer)
    net = TextEncoder(db)

    loader = DataLoader(db, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        input_inds = batched['word_inds'].long()
        input_lens = batched['word_lens'].long()
        
        print('Checking the output shapes')
        out = net(input_inds, input_lens)
        out_rfts, out_embs, out_msks, out_hids = out
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


def test_vol_encoder(config):    
    transformer = volume_normalize('background')
    db = layout_coco(config, 'test', transform=transformer)
    
    vol_encoder = VolumeEncoder(config)
    print(get_n_params(vol_encoder))

    loader = DataLoader(db, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        x = batched['background'].float()
        y = vol_encoder(x)
        print(y.size())
        break


def test_coco_decoder(config):
    transformer = volume_normalize('background')
    db = layout_coco(config, 'val', transform=transformer)

    text_encoder  = TextEncoder(db)
    img_encoder   = VolumeEncoder(config)
    what_decoder  = WhatDecoder(config)
    where_decoder = WhereDecoder(config)

    print('txt_encoder', get_n_params(text_encoder))
    print('vol_encoder', get_n_params(img_encoder))
    print('what_decoder', get_n_params(what_decoder))
    print('where_decoder', get_n_params(where_decoder))


    loader = DataLoader(db, 
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
        obj_logits, rnn_outs, nxt_hids, prev_bgfs, att_ctx, att_wei = what_outs
        print('------------------------------------------')
        print('obj_logits', obj_logits.size())
        print('rnn_outs', rnn_outs.size())
        print('nxt_hids', nxt_hids.size())
        print('prev_bgfs', prev_bgfs.size())
        # print('att_ctx', att_ctx.size())
        # print('att_wei', att_wei.size())

        _, obj_inds = torch.max(obj_logits + 1.0, dim=-1)
        curr_fgfs = indices2onehots(obj_inds.cpu().data, config.output_cls_size).float()
        # curr_fgfs = curr_fgfs.unsqueeze(1)
        if config.cuda:
            curr_fgfs = curr_fgfs.cuda()


        where_outs = where_decoder((rnn_outs, curr_fgfs, prev_bgfs, att_ctx), encoder_states)
        coord_logits, attri_logits, where_ctx, where_wei = where_outs
        print('coord_logits ', coord_logits.size())
        print('attri_logits ', attri_logits.size())
        # print('att_ctx', where_ctx.size())
        # print('att_wei', where_wei.size())
        break


def visualize_unigram(config, img, unigrams, color):
    for i in range(len(unigrams)):
        vec = unigrams[i].copy()
        xy  = vec[-2:]
        bb  = vec[-6:-2]
        iw, ih = config.draw_size
        xy  = (xy * np.array([iw, ih])).astype(np.int32)
        bb  = (bb * np.array([iw, ih, iw, ih])).astype(np.int32)
        cv2.circle(img, (xy[0], xy[1]), 12, color, -1)
        # cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
    return img


def visualize_bigram(config, img, bigrams, color):
    for i in range(len(bigrams)):
        vec = bigrams[i].copy()
        txy = (vec[4:6] * np.array(config.draw_size)).astype(np.int32)
        sxy = txy + (vec[6:8] * np.array(config.draw_size)).astype(np.int32)
        cv2.line(img, (sxy[0], sxy[1]), (txy[0], txy[1]), color, 3)
    return img


def test_evaluator(config):
    transformer = volume_normalize('background')
    db = layout_coco(config, 'val', transform=transformer)
    output_dir = osp.join(db.cfg.model_dir, 'test_evaluator_coco')
    maybe_create(output_dir)

    ev = evaluator(db)
    for i in range(0, len(db), 2):
        # print('--------------------------------------')
        entry_1 = db[i]
        entry_2 = db[i+1]
        scene_1 = db.scenedb[i]
        scene_2 = db.scenedb[i+1]
        name_1 = osp.splitext(osp.basename(entry_1['color_path']))[0]
        name_2 = osp.splitext(osp.basename(entry_2['color_path']))[0]

        graph_1 = scene_graph(db, scene_1, entry_1['out_inds'], True)
        graph_2 = scene_graph(db, scene_2, entry_2['out_inds'], False)

        color_1 = cv2.imread(entry_1['color_path'], cv2.IMREAD_COLOR)
        color_2 = cv2.imread(entry_2['color_path'], cv2.IMREAD_COLOR)
        color_1, _, _ = create_squared_image(color_1)
        color_2, _, _ = create_squared_image(color_2)
        color_1 = cv2.resize(color_1, (config.draw_size[0], config.draw_size[1]))
        color_2 = cv2.resize(color_2, (config.draw_size[0], config.draw_size[1]))

        color_1 = visualize_unigram(config, color_1, graph_1.unigrams, (225, 0, 0))
        color_2 = visualize_unigram(config, color_2, graph_2.unigrams, (225, 0, 0))
        color_1 = visualize_bigram(config, color_1, graph_1.bigrams, (0, 255, 255))
        color_2 = visualize_bigram(config, color_2, graph_2.bigrams, (0, 255, 255))

        scores = ev.evaluate_graph(graph_1, graph_2)

        color_1 = visualize_unigram(config, color_1, ev.common_pred_unigrams, (0, 225, 0))
        color_2 = visualize_unigram(config, color_2, ev.common_gt_unigrams,   (0, 225, 0))
        color_1 = visualize_bigram(config, color_1, ev.common_pred_bigrams, (255, 255, 0))
        color_2 = visualize_bigram(config, color_2, ev.common_gt_bigrams, (255, 255, 0))

        info = eval_info(config, scores[None, ...])

        plt.switch_backend('agg')
        fig = plt.figure(figsize=(16, 10))
        title = entry_1['sentence'] + '\n' + entry_2['sentence'] + '\n'
        title += 'unigram f3: %f, bigram f3: %f, bigram sim: %f\n'%(info.unigram_F3()[0], info.bigram_F3()[0], info.bigram_coord()[0])
        title += 'scale: %f, ratio: %f, coord: %f \n'%(info.scale()[0], info.ratio()[0], info.unigram_coord()[0])

        
        plt.suptitle(title)
        plt.subplot(1, 2, 1); plt.imshow(color_1[:,:,::-1]); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(color_2[:,:,::-1]); plt.axis('off')

        out_path = osp.join(output_dir, name_1+'_'+name_2+'.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

        if i > 40:
            break


def test_simulator(config):
    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'simulator')
    maybe_create(output_dir)
    
    transformer = volume_normalize('background')
    db = layout_coco(config, 'val', transform=transformer)

    loader = DataLoader(db, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers)

    env = simulator(db, config.batch_size)
    env.reset()

    for cnt, batched in enumerate(loader):
        out_inds = batched['out_inds'].numpy()
        gt_paths = batched['color_path']
        img_inds = batched['image_idx']
        sents    = batched['sentence']

        sequences = []
        for i in range(out_inds.shape[1]):
            frames = env.batch_render_to_pytorch(out_inds[:, i, :])
            sequences.append(frames)
        seqs1 = torch.stack(sequences, dim=1)
        print('seqs1', seqs1.size())
        seqs2 = env.batch_redraw(return_sequence=True)

        seqs = seqs2
        for i in range(len(seqs)):
            imgs = seqs[i]
            image_idx = img_inds[i]
            name = '%03d_'%i + str(image_idx.item()).zfill(9)
            out_path = osp.join(output_dir, name+'.png')
            color = cv2.imread(gt_paths[i], cv2.IMREAD_COLOR)
            color, _, _ = create_squared_image(color)
            
            fig = plt.figure(figsize=(32, 16))
            plt.suptitle(sents[i])

            for j in range(len(imgs)):
                plt.subplot(3, 5, j+1)
                plt.imshow(imgs[j])
                plt.axis('off')

            plt.subplot(3, 5, 15)
            plt.imshow(color[:,:,::-1])
            plt.axis('off')
            
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        break


def test_model(config):
    transformer = volume_normalize('background')
    db = layout_coco(config, 'val', transform=transformer)
    net = DrawModel(db)

    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'test_model_coco')
    maybe_create(output_dir)

    pretrained_path = osp.join('../data/caches/layout_ckpts/supervised_coco_top1.pkl')
    assert osp.exists(pretrained_path)
    if config.cuda:
        states = torch.load(pretrained_path) 
    else:
        states = torch.load(pretrained_path, map_location=lambda storage, loc: storage) 
    net.load_state_dict(states)

    loader = DataLoader(db, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers)

    net.eval()
    for cnt, batched in enumerate(loader):
        word_inds = batched['word_inds'].long()
        word_lens = batched['word_lens'].long()
        bg_images = batched['background'].float()

        fg_inds = batched['fg_inds'].long()
        gt_inds = batched['out_inds'].long()
        gt_msks = batched['out_msks'].float()

        fg_onehots = indices2onehots(fg_inds, config.output_cls_size)


        # inf_outs, _ = net((word_inds, word_lens, bg_images, fg_onehots))
        # obj_logits, coord_logits, attri_logits, enc_msks, what_wei, where_wei = inf_outs
        # print('teacher forcing')
        # print('obj_logits ', obj_logits.size())
        # print('coord_logits ', coord_logits.size())
        # print('attri_logits ', attri_logits.size())
        # if config.what_attn:
        #     print('what_att_logits ', what_wei.size())
        # if config.where_attn > 0:
        #     print('where_att_logits ', where_wei.size())
        # print('----------------------')
        # inf_outs, env = net.inference(word_inds, word_lens, -1, 0, 0, gt_inds)
        inf_outs, env = net.inference(word_inds, word_lens, -1, 2.0, 0, None)
        obj_logits, coord_logits, attri_logits, enc_msks, what_wei, where_wei = inf_outs
        print('scheduled sampling')
        print('obj_logits ', obj_logits.size())
        print('coord_logits ', coord_logits.size())
        print('attri_logits ', attri_logits.size())
        if config.what_attn:
            print('what_att_logits ', what_wei.size())
        if config.where_attn > 0:
            print('where_att_logits ', where_wei.size())
        print('----------------------')


        pred_out_inds, pred_out_msks = env.get_batch_inds_and_masks()
        print('pred_out_inds', pred_out_inds[0, 0], pred_out_inds.shape)
        print('gt_inds', gt_inds[0, 0], gt_inds.size())
        print('pred_out_msks', pred_out_msks[0, 0], pred_out_msks.shape)
        print('gt_msks', gt_msks[0, 0], gt_msks.size())



        batch_frames = env.batch_redraw(True)
        scene_inds = batched['scene_idx']
        for i in range(len(scene_inds)):
            sid = scene_inds[i]
            entry = db[sid]
            name = osp.splitext(osp.basename(entry['color_path']))[0]
            imgs = batch_frames[i]
            out_path = osp.join(output_dir, name+'.png')

            fig = plt.figure(figsize=(60, 30))
            plt.suptitle(entry['sentence'], fontsize=50)
            for j in range(len(imgs)):
                plt.subplot(4, 3, j+1)
                plt.imshow(imgs[j,:,:,::-1].astype(np.uint8))
                plt.axis('off')

            target = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
            plt.subplot(4, 3, 12)
            plt.imshow(target[:,:,::-1])
            plt.axis('off')
            
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        break


def test_topk(config):
    import os.path as osp
    from coco import coco
    from utils import volume_normalize, maybe_create, indices2onehots
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    
    transformer = volume_normalize('background')
    db = coco(config, 'val', transform=transformer)
    net = DrawModel(db)

    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'test_topk_coco')
    maybe_create(output_dir)

    pretrained_path = osp.join('data/caches/supervised_coco.pkl')
    assert osp.exists(pretrained_path)
    if config.cuda:
        states = torch.load(pretrained_path) 
    else:
        states = torch.load(pretrained_path, map_location=lambda storage, loc: storage) 
    net.load_state_dict(states)

    plt.switch_backend('agg')

    for i in range(len(db)):
        entry = db[i]
        scene = db.scenedb[i]
        
        input_inds_np = np.array(entry['word_inds'])
        input_lens_np = np.array(entry['word_lens'])

        input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
        input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
        if config.cuda:
            input_inds = input_inds.cuda()
            input_lens = input_lens.cuda()

        net.eval()
        with torch.no_grad():
            env = net.topk_inference(input_inds, input_lens, config.beam_size, -1)
        frames = env.batch_redraw(return_sequence=True)
        gt_img = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
        for j in range(len(frames)):
            fig = plt.figure(figsize=(60, 30))
            title = entry['sentence']
            # title = title + '\n reward: %f, scores: %f, %f, %f, %f, %f'%(rews[j], *(scores[j]))
            plt.suptitle(title, fontsize=50)
            imgs = frames[j]
            for k in range(len(imgs)):
                plt.subplot(3, 4, k+1)
                plt.imshow(imgs[k, :, :, ::-1])
                plt.axis('off')
            plt.subplot(3, 4, 12)
            plt.imshow(gt_img[:, :, ::-1])
            plt.axis('off')
            output_path = osp.join(output_dir, '%03d_%03d.png'%(i, j))
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)

        
        if i > 10:
            break


if __name__ == '__main__':
    cv2.setNumThreads(0)

    config, unparsed = get_config()
    config = layout_arguments(config)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)


    # test_txt_encoder_coco(config)
    # test_vol_encoder(config)
    # test_coco_decoder(config)
    # test_evaluator(config)
    # test_simulator(config)
    test_model(config)

