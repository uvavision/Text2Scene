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
from modules.embedding_model import EmbeddingModel
from modules.embedding_trainer import EmbeddingTrainer
# from nntable import AllCategoriesTables

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def train_embedding_model(config):
    traindb = coco(config, 'train')
    valdb   = coco(config, 'val')
    testdb  = coco(config, 'test')
    trainer = EmbeddingTrainer(traindb)
    # trainer.load_pretrained_net('epoch_06')
    # we use the official validation set as test set
    trainer.train(traindb, testdb, valdb)


def overfit_embedding_model(config):
    config.log_per_steps = 1
    # traindb = coco(config, 'train')
    valdb = coco(config, 'val')
    # testdb  = coco(config, 'test')
    # traindb.scenedb = traindb.scenedb[:config.batch_size*3]
    valdb.scenedb = valdb.scenedb[:config.batch_size*3]
    # testdb.scenedb = testdb.scenedb[:config.batch_size]
    # print('build pca table')
    # pca_table = AllCategoriesTables(traindb)
    # pca_table.run_PCAs_and_build_nntables_in_feature_space()
    # print('create trainer')
    trainer = EmbeddingTrainer(valdb)
    # print('start training')
    # trainer.train(traindb, traindb, traindb)
    trainer.train(valdb, valdb, valdb)



if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # overfit_embedding_model(config)
    train_embedding_model(config)
