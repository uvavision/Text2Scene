#!/usr/bin/env python

import _init_paths
import math, cv2, random
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
from datasets.abstract_scene import abstract_scene
from modules.abstract_trainer import SupervisedTrainer

from abstract_utils import *
from abstract_config import get_config


def train_model(config):
    transformer = image_normalize('background')
    train_db = abstract_scene(config, split='train', transform=transformer)   
    val_db   = abstract_scene(config, split='val',   transform=transformer) 
    test_db  = abstract_scene(config, split='test',  transform=transformer) 

    trainer = SupervisedTrainer(train_db)
    trainer.train(train_db, val_db, test_db) 


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

    train_model(config)