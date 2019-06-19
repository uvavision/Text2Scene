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
from utils import *

from datasets.coco import coco
from datasets.coco_loader import synthesis_loader
from modules.synthesis_model import SynthesisModel
from modules.synthesis_trainer import SynthesisTrainer

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def train_synthesis_model(config):
    traindb = coco(config, 'train', '2017')
    # valdb   = coco(config, 'val',   '2017')
    testdb  = coco(config, 'test',  '2017')

    trainer = SynthesisTrainer(config)
    # we use the official validation set as test set
    trainer.train(traindb, testdb, testdb)


def overfit_synthesis_model(config):
    config.log_per_steps = 1
    # traindb = coco(config, 'train', '2017')
    testdb  = coco(config, 'test',  '2017')

    # traindb.scenedb = traindb.scenedb[:config.batch_size*3]
    testdb.scenedb = testdb.scenedb[:config.batch_size*3]
    # testdb.scenedb = testdb.scenedb[:config.batch_size]
    # print('build pca table')
    # pca_table = AllCategoriesTables(traindb)
    # pca_table.run_PCAs_and_build_nntables_in_feature_space()
    # print('create trainer')
    trainer = SynthesisTrainer(config)
    # print('start training')
    # trainer.train(traindb, traindb, traindb)
    trainer.train(testdb, testdb, testdb)


def test_synthesis_model(config):
    traindb = coco(config, 'train', '2017')
    # valdb   = coco(config, 'val',   '2017')
    testdb  = coco(config, 'aux',  '2017')

    trainer = SynthesisTrainer(config)
    # we use the official validation set as test set
    trainer.sample_for_eval(testdb)
    

if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # overfit_synthesis_model(config)
    # train_synthesis_model(config)
    test_synthesis_model(config)
