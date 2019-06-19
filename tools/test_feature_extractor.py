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
from modules.encoder import FeatureExtractor
import torch, torchtext
from torch.utils.data import Dataset


def test_feature_extractor(config):

    fake_images_np = np.ones((4, 3, 224, 224), dtype=np.float32)
    fake_images_th = torch.from_numpy(fake_images_np).float()

    net = FeatureExtractor(config)

    features = net(fake_images_th)
    print(features.size())


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_feature_extractor(config)