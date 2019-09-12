#!/usr/bin/env python

import _init_paths
import os, sys, cv2, math, PIL, cairo, random
import numpy as np
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from nltk.tokenize import word_tokenize
from collections import OrderedDict
# from evaluator import scene_graph, eval_info, evaluator
from abstract_config import get_config
from abstract_utils import *

import torch, torchtext
from torch.utils.data import Dataset
from datasets.abstract_scene import abstract_scene


def test_dataset(config):
    db = abstract_scene(config, 'train')
    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'test_abstract_scene')
    maybe_create(output_dir)

    indices = np.random.permutation(range(len(db)))
    indices = indices[:config.n_samples]

    for i in indices:
        entry = db[i]
        scene = db.scenedb[i]
        # print('cls_inds: ', scene['cls_inds'])
        imgs = db.render_scene_as_input(scene, True, True)
        name = osp.splitext(osp.basename(entry['color_path']))[0]
        out_path = osp.join(output_dir, name+'.png')
        fig = plt.figure(figsize=(16, 8))
        plt.suptitle(entry['sentence'])

        for j in range(min(len(imgs), 11)):
            plt.subplot(4, 3, j+1)
            plt.imshow(imgs[j,:,:,::-1].astype(np.uint8))
            plt.axis('off')

        target = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
        plt.subplot(4, 3, 12)
        plt.imshow(target[:,:,::-1])
        plt.axis('off')
        
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)


def test_loader(config):
    from torch.utils.data import DataLoader
    transformer = image_normalize('background')

    # traindb = abstract_scene(config, 'train', transform=transformer)
    # print('traindb', len(traindb))
    # valdb   = abstract_scene(config, 'val', transform=transformer)
    # print('valdb', len(valdb))
    # testdb  = abstract_scene(config, 'test', transform=transformer)
    # print('testdb', len(testdb))

    # print(testdb.scenedb[0]['scene_idx'], testdb.scenedb[-1]['scene_idx'])

    db = abstract_scene(config, 'val', transform=transformer)
    
    loader = DataLoader(db, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers)
    for cnt, batched in enumerate(loader):
        print(batched['scene_idx'].size())
        print(batched['word_inds'].size())
        print(batched['word_lens'].size())
        print(batched['word_inds'][0])
        print(batched['word_lens'][0])
        print(batched['background'].size())
        print(batched['out_inds'].size())
        print(batched['out_msks'].size())
        print(batched['fg_inds'].size())
        print(batched['hmaps'].size())
        print(batched['out_inds'][0,:,0])
        print(batched['fg_inds'][0])
        break


    # db = abstract_scene(config, 'val')
    # loader = DataLoader(db, 
    #     batch_size=config.batch_size, 
    #     shuffle=True, 
    #     num_workers=config.num_workers)
    # output_dir = osp.join(config.model_dir, 'test_loader')
    # maybe_create(output_dir)
    # for cnt, batched in enumerate(loader):
    #     bgs = batched['background'].numpy()
    #     for i in range(len(bgs)):
    #         imgs = bgs[i]
    #         fig = plt.figure(figsize=(16, 8))
    #         for j in range(len(imgs)):
    #             plt.subplot(4, 4, j+1)
    #             plt.imshow(imgs[j,:,:,::-1].astype(np.uint8))
    #             plt.axis('off')  
    #         out_path = osp.join(output_dir, '%09d_%09d.png'%(cnt, i))      
    #         fig.savefig(out_path, bbox_inches='tight')
    #         plt.close(fig)
    #     break


def test_lang_vocab(config):
    db = abstract_scene(config, 'train')
    scenedb = db.scenedb
    lang_vocab = db.lang_vocab

    sent_lens = []
    for i in range(len(scenedb)):
        group_sents = scenedb[i]['scene_sentences']
        for j in range(len(group_sents)):
            triplet = group_sents[j]
            for k in range(len(triplet)):
                sentence = triplet[k]
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
        clses = scenedb[i]['clips']
        obj_lens.append(len(clses))
    print('obj len: ', np.median(obj_lens), np.amax(obj_lens), np.amin(obj_lens))
    for p in ps:
        print('p %d/100: '%(int(p)), np.percentile(obj_lens, p))
    # 4.0 38 2


def test_clip_and_triplet(config):
    db = abstract_scene(config)
    clip_vocab = db.clip_vocab

    for i in range(len(clip_vocab.index2word)):
        o, p, e = db.clip_to_triplet(i)
        w = clip_vocab.index2word[i]
        print(i, o, p, e, w)
        j = db.triplet_to_clip([o, p, e])
        assert(i == j)


def test_indices(config):
    plt.switch_backend('agg')
    output_dir = osp.join(config.model_dir, 'test_indices')
    maybe_create(output_dir)

    db = abstract_scene(config, 'test')
    # print('db', len(db))
    
    indices = np.random.permutation(range(len(db)))
    indices = indices[:config.n_samples]

    for i in indices:
        entry = db[i]
        scene = db.scenedb[i]

        out_inds, out_msks = db.scene_to_output_inds(scene)
        new_scene = db.output_inds_to_scene(out_inds)
        imgs = db.render_scene_as_input(new_scene, True, True)

        name = osp.splitext(osp.basename(entry['color_path']))[0]
        out_path = osp.join(output_dir, name+'.png')
        fig = plt.figure(figsize=(16, 8))
        plt.suptitle(entry['sentence'])

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

        print('out_inds: ', out_inds)
        print('out_msks: ', out_msks)
        # print('cls_inds: ', entry['cls_inds'])
        # print('pos_inds: ', entry['pos_inds'])
        # print('trans_inds: ', entry['trans_inds'])
        break


if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    config = abstract_arguments(config)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # test_dataset(config)
    test_loader(config)
    # test_lang_vocab(config)
    # test_clip_and_triplet(config)
    # test_indices(config)