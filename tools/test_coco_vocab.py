#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from config import get_config
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from utils import *

from datasets.coco import coco
import torch, torchtext
from torch.utils.data import Dataset


def test_coco_vocab(config):
    db = coco(config, 'train', '2017')
    output_dir = osp.join(config.model_dir, 'test_coco_vocab')
    maybe_create(output_dir)

    print('scenedb', len(db.scenedb))

    scenedb = db.scenedb
    lang_vocab = db.lang_vocab
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
    # 10.0 50 5

    print("vocab size: ", len(lang_vocab.index2word)) # 10041
    print("vocab: ", lang_vocab.index2word[:10])


    obj_areas = []
    obj_lens = []
    for i in range(len(scenedb)):
        clses = scenedb[i]['clses']
        areas = scenedb[i]['areas']
        width = scenedb[i]['width']
        height = scenedb[i]['height']
        areas = areas/float(width*height)
        obj_lens.append(len(clses))
        obj_areas.extend(areas.tolist())

    obj_areas = -np.array(obj_areas)

    print('obj len: ', np.median(obj_lens), np.amax(obj_lens), np.amin(obj_lens))
    for p in ps:
        print('p %d/100: '%(int(p)), np.percentile(obj_lens, p))
    # 4.0 90 0
    print('obj area: ', np.median(obj_areas), np.amax(obj_areas), np.amin(obj_areas))
    for p in ps:
        print('p %d/100: '%(int(p)), np.percentile(obj_areas, p))


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_coco_vocab(config)
