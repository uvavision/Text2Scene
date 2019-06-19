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

from datasets.cityscapes_syn import cityscapes_syn, cityscapes_tranformation
from modules.image_synthesis_model import ImageSynthesisModel
from modules.image_synthesis_trainer import ImageSynthesisTrainer

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def train_synthesis_model(config):
    tranformer = cityscapes_tranformation()
    traindb = cityscapes_syn(config, 'train', tranformer)
    valdb = cityscapes_syn(config, 'val', tranformer)

    trainer = ImageSynthesisTrainer(config)
    trainer.train(traindb, valdb)


def overfit_synthesis_model(config):
    config.log_per_steps = 1
    tranformer = cityscapes_tranformation()
    traindb = cityscapes_syn(config, 'train', tranformer)
    # valdb = cityscapes_syn(config, 'val', tranformer)
    traindb.image_index = traindb.image_index[:config.batch_size]
    # valdb.image_index = valdb.image_index[:config.batch_size]

    trainer = ImageSynthesisTrainer(config)
    trainer.train(traindb, traindb)
    

if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # overfit_synthesis_model(config)
    train_synthesis_model(config)
