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
from modules.puzzle_model import PuzzleModel
from modules.puzzle_trainer import PuzzleTrainer
# from nntable import AllCategoriesTables

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nntable import AllCategoriesTables


def puzzle_model_inference_preparation(config):
    traindb = coco(config, 'train', '2017')
    testdb = coco(config, 'test', '2017')
    trainer = PuzzleTrainer(traindb)
    t0 = time()
    trainer.dump_shape_vectors(traindb)
    trainer.dump_shape_vectors(testdb)
    print("Dump shape vectors completes (time %.2fs)" % (time() - t0))
    t0 = time()
    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(False)
    print("NN completes (time %.2fs)" % (time() - t0))
    # trainer.sample(0, valdb, len(valdb), random_or_not=False)


def puzzle_model_inference(config):
    traindb = coco(config, 'train', '2017')
    valdb = coco(config, 'val', '2017')
    auxdb = coco(config, 'aux', '2017')
    trainer = PuzzleTrainer(traindb)
    t0 = time()
    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(True)
    print("NN completes (time %.2fs)" % (time() - t0))
    t0 = time()
    if config.for_visualization:
        trainer.sample_for_vis(0, valdb, len(valdb.scenedb), nn_table=all_tables)
        trainer.sample_for_vis(0, auxdb, len(auxdb.scenedb), nn_table=all_tables)
    else:
        trainer.sample_for_eval(valdb, nn_table=all_tables)
        # trainer.sample_for_eval(auxdb, nn_table=all_tables)
    print("Sampling completes (time %.2fs)" % (time() - t0))
    
    
if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    puzzle_model_inference_preparation(config)
    # puzzle_model_inference(config)