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
from annoy import AnnoyIndex
from utils import *
from tsne_wrapper import *
from nntable import AllCategoriesTables

from datasets.coco import coco
import torch, torchtext
from torch.utils.data import Dataset
from pycocotools import mask as COCOmask


def test_nntable(config):
    db = coco(config, 'test')
    output_dir = osp.join(config.model_dir, 'test_nntable')
    maybe_create(output_dir)

    t0 = time()
    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(False)
    print("NN completes (time %.2fs)" % (time() - t0))


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_nntable(config)
