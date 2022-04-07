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


def abstract_eval(config):
    transformer = image_normalize('background')
    train_db = abstract_scene(config, split='train', transform=transformer)
    test_db  = abstract_scene(config, split='test',  transform=transformer)   
    trainer = SupervisedTrainer(train_db)

    val_loss, val_accu, val_infos = trainer.validate_epoch(test_db) 
    out_path = osp.join(config.model_dir, config.exp_name+'.json')  
    log_scores(val_infos, out_path) 


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

    abstract_eval(config)
