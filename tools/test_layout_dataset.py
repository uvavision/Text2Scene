#!/usr/bin/env python

import _init_paths
import os, sys, cv2, math, PIL, cairo, random
import numpy as np
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

import torch, torchtext
from torch.utils.data import Dataset

from layout_config import get_config
from layout_utils import *
from datasets.layout_coco import layout_coco


def test_dataset(config):
    traindb = layout_coco(config, 'train')
    valdb = layout_coco(config, 'val')
    db = layout_coco(config, 'test')
    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'test_dataset')
    maybe_create(output_dir)

    indices = np.random.permutation(range(len(db)))
    indices = indices[:config.n_samples]

    for i in indices:
        entry = db[i]
        layouts = db.render_indices_as_output(entry)
        image_idx = entry['image_idx']
        name = '%03d_'%i + str(image_idx).zfill(12)
        out_path = osp.join(output_dir, name+'.png')
        color = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
        color, _, _ = create_squared_image(color)
        
        fig = plt.figure(figsize=(32, 16))
        plt.suptitle(entry['sentence'], fontsize=30)

        for j in range(len(layouts)):
            plt.subplot(3, 5, j+1)
            plt.imshow(layouts[j])
            plt.axis('off')

        plt.subplot(3, 5, 15)
        plt.imshow(color[:,:,::-1])
        plt.axis('off')
        
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)


def test_loader(config):
    from torch.utils.data import DataLoader
    transformer = volume_normalize('background')
    db = layout_coco(config, 'test', transform=transformer)
    
    loader = DataLoader(db, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers)
    for cnt, batched in enumerate(loader):
        # print(batched['background'].size())
        print(batched['word_inds'].size())
        print(batched['word_lens'].size())
        # print(batched['word_inds'][1])
        # print(batched['word_lens'][1])
        print(batched['out_inds'].size())
        print(batched['out_msks'].size())
        print(batched['out_inds'][0])
        print(batched['out_msks'][0])
        # print(batched['trans_inds'].size())
        # print(batched['cls_mask'].size())
        # print(batched['pos_mask'].size())
        # cls_inds = batched['cls_inds']
        # fg_onehots = batched['foreground_onehots']
        # foo = np.argmax(fg_onehots, axis=-1)
        # assert((cls_inds == foo).all())
        # print(cls_inds, foo)
        # print(batched['word_vecs'].shape)
        # A = batched['output_clip_indices']
        # B = batched['output_clip_onehots']
        # C = np.argmax(B, axis=-1)
        # assert((A==C).all())
        # print(A[0], C[0])
        break


def test_lang_vocab(config):
    train_db = coco(config, 'train')
    val_db = coco(config, 'val')

    scenedb = train_db.scenedb
    lang_vocab = val_db.lang_vocab

    sent_lens = []
    for i in range(len(scenedb)):
        group_sents = scenedb[i]['captions']
        for j in range(len(group_sents)):
            sentence = group_sents[j]
            tokens = word_tokenize(sentence.lower())
            tokens = further_token_process(tokens)
            word_inds = [lang_vocab.word_to_index(w) for w in tokens]
            # word_inds = [wi for wi in word_inds if wi > config.EOS_idx] 
            sent_lens.append(len(word_inds))

    
    print('sent len: ', np.median(sent_lens), np.amax(sent_lens), np.amin(sent_lens))
    ps = [80.0, 90.0, 95.0, 99.0]
    for p in ps:
        print('p %d/100: '%(int(p)), np.percentile(sent_lens, p))
    # 10.0 50 6

    print("vocab size: ", len(lang_vocab.index2word))
    print("vocab: ", lang_vocab.index2word[:10])

    obj_lens = []
    for i in range(len(scenedb)):
        clses = scenedb[i]['clses']
        obj_lens.append(len(clses))
    print('obj len: ', np.median(obj_lens), np.amax(obj_lens), np.amin(obj_lens))
    for p in ps:
        print('p %d/100: '%(int(p)), np.percentile(obj_lens, p))
    # 4.0 38 2


def test_overlay_boxes(config):
    db = coco(config, 'test')
    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'test_overlay_boxes')
    maybe_create(output_dir)
    indices = np.random.permutation(range(len(db)))
    indices = indices[:config.n_samples]
    for i in indices:
        entry = db[i]
        layouts = db.overlay_boxes(entry)
        image_idx = entry['image_idx']
        name = '%03d_'%i + str(image_idx).zfill(12)
        out_path = osp.join(output_dir, name+'.png')
        color = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
        color, _, _ = create_squared_image(color)
        
        fig = plt.figure(figsize=(32, 16))
        plt.suptitle(entry['sentence'], fontsize=30)

        for j in range(len(layouts)):
            plt.subplot(3, 5, j+1)
            plt.imshow(layouts[j])
            plt.axis('off')

        plt.subplot(3, 5, 15)
        plt.imshow(color[:,:,::-1])
        plt.axis('off')
        
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)


def bbox_statistic(config):
    train_db = coco(config, 'train')
    val_db  = coco(config, 'val')
    scenedb = train_db.scenedb + val_db.scenedb

    all_boxes = []
    for i in range(len(scenedb)):
        boxes = scenedb[i]['boxes']
        all_boxes.append(boxes)
    all_boxes = np.concatenate(all_boxes, axis=0)
    print(all_boxes.shape)
    print('--------')
    print('boxes: ', np.median(all_boxes, axis=0), np.amax(all_boxes, axis=0), np.amin(all_boxes, axis=0))


def statistic(config):
    train_db  = coco(config, 'train')
    val_db    = coco(config, 'val')
    test_db   = coco(config, 'test')

    print(len(train_db))
    print(len(val_db))
    print(len(test_db))


    # scenedb    = train_db.scenedb
    # lang_vocab = val_db.lang_vocab

    # sent_lens = []
    # areas = []
    # for i in range(len(scenedb)):
    #     group_sents = scenedb[i]['captions']
    #     for j in range(len(group_sents)):
    #         sentence = group_sents[j]
    #         tokens = word_tokenize(sentence.lower())
    #         tokens = further_token_process(tokens)
    #         word_inds = [lang_vocab.word_to_index(w) for w in tokens]
    #         # word_inds = [wi for wi in word_inds if wi > config.EOS_idx] 
    #         sent_lens.append(len(word_inds))

    #     curr_areas = deepcopy(scenedb[i]['areas'])
    #     areas.extend(curr_areas.tolist())

    # areas = np.array(areas)
    
    # print('sent len: ', np.median(sent_lens), np.amax(sent_lens), np.amin(sent_lens))
    # ps = [80.0, 90.0, 95.0, 99.0]
    # for p in ps:
    #     print('p %d/100: '%(int(p)), np.percentile(sent_lens, p))

    # print('areas: ', np.median(areas), np.amax(areas), np.amin(areas))
    # ps = [80.0, 90.0, 95.0, 99.0]
    # for p in ps:
    #     print('p %d/100: '%(int(p)), np.percentile(-areas, p))

    # print("vocab size: ", len(lang_vocab.index2word))
    # print("vocab: ", lang_vocab.index2word[:10])

    # obj_lens = []
    # for i in range(len(scenedb)):
    #     clses = scenedb[i]['clses']
    #     obj_lens.append(len(clses))
    # print('obj len: ', np.median(obj_lens), np.amax(obj_lens), np.amin(obj_lens))
    # for p in ps:
    #     print('p %d/100: '%(int(p)), np.percentile(obj_lens, p))
    # # 4.0 38 2


if __name__ == '__main__':
    config, unparsed = get_config()
    config = layout_arguments(config)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    
    # test_dataset(config)
    test_loader(config)
    # test_lang_vocab(config)
    # test_overlay_boxes(config)
    # bbox_statistic(config)
    # statistic(config)
