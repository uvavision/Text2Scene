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

from nntable import AllCategoriesTables


def embedding_model_inference_preparation(config):
    traindb = coco(config, 'train')
    valdb   = coco(config, 'val')
    trainer = EmbeddingTrainer(traindb)
    t0 = time()
    trainer.dump_shape_vectors(traindb)
    print("Dump shape vectors completes (time %.2fs)" % (time() - t0))
    t0 = time()
    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(False)
    print("NN completes (time %.2fs)" % (time() - t0))
    # trainer.sample(0, valdb, len(valdb), random_or_not=False)


def embedding_model_inference(config):
    traindb = coco(config, 'train')
    valdb   = coco(config, 'val')
    trainer = EmbeddingTrainer(traindb)
    t0 = time()
    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(True)
    print("NN completes (time %.2fs)" % (time() - t0))
    t0 = time()
    trainer.sample(0, valdb, 50, random_or_not=False)
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

    embedding_model_inference(config)
