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
from modules.scene_model import SceneModel
from modules.supervised_trainer import SupervisedTrainer
from nntable import AllCategoriesTables

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def train_scene_model(config):
    transformer = image_normalize('background')
    traindb = coco(config, 'train', transform=transformer)
    valdb = coco(config, 'val', transform=transformer)
    testdb = coco(config, 'test', transform=transformer)
    pca_table = AllCategoriesTables(traindb)
    pca_table.run_PCAs_and_build_nntables_in_feature_space()

    trainer = SupervisedTrainer(traindb)
    # we use the official validation set as test set
    trainer.train(traindb, testdb, valdb)


def overfit_scene_model(config):
    config.log_per_steps = 1
    transformer = image_normalize('background')
    traindb = coco(config, 'train', transform=transformer)
    valdb   = coco(config, 'val', transform=transformer)
    testdb  = coco(config, 'test', transform=transformer)
    traindb.scenedb = traindb.scenedb[:config.batch_size]
    valdb.scenedb = valdb.scenedb[:config.batch_size]
    testdb.scenedb = testdb.scenedb[:config.batch_size]
    print('build pca table')
    pca_table = AllCategoriesTables(traindb)
    pca_table.run_PCAs_and_build_nntables_in_feature_space()
    print('create trainer')
    trainer = SupervisedTrainer(traindb)
    print('start training')
    # trainer.train(traindb, traindb, traindb)
    trainer.train(traindb, valdb, testdb)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # overfit_scene_model(config)
    train_scene_model(config)
