#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt

from composites_utils import *
from composites_config import get_config
from datasets.composites_coco import composites_coco
from datasets.composites_loader import sequence_loader, synthesis_loader
from modules.synthesis_model import SynthesisModel
from modules.synthesis_trainer import SynthesisTrainer

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def test_synthesis_model(config):
    testdb  = composites_coco(config, 'test',  '2017')
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

    test_synthesis_model(config)
