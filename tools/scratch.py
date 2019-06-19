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

from html_writer import HTML


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    # prepare_directories(config)

    image_folder_name = '0919_epoch_02'

    config_html = HTML()
    config_table = config_html.table(border='1')
    img_paths = sorted(glob("%s/*.png"%image_folder_name))
    # r = config_table.tr
    # r.th('%s'%self.classes[i])
    for j in range(len(img_paths)):
        c = config_table.tr
        c.img(src='%s'%(img_paths[j]),width='800',height='800')
        # c.img(src='%s'%(img_paths[j]))
    html_file = open('%s.html'%image_folder_name, 'w')
    print(config_table, file=html_file)
    html_file.close()
