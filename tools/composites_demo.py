#!/usr/bin/env python

import _init_paths
import cv2, random
import numpy as np
import os.path as osp
from time import time
import matplotlib.pyplot as plt

from composites_utils import *
from composites_config import get_config
from datasets.composites_coco import composites_coco

from modules.puzzle_model import PuzzleModel
from modules.puzzle_trainer import PuzzleTrainer

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nntable import AllCategoriesTables


def puzzle_model_inference_preparation(config):
    traindb = composites_coco(config, 'train', '2017')
    testdb  = composites_coco(config, 'test', '2017')
    trainer = PuzzleTrainer(traindb)
    t0 = time()
    trainer.dump_shape_vectors(traindb)
    trainer.dump_shape_vectors(testdb)
    print("Dump shape vectors completes (time %.2fs)" % (time() - t0))
    t0 = time()
    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(False)
    print("NN completes (time %.2fs)" % (time() - t0))


def puzzle_model_inference(config):
    traindb = composites_coco(config, 'train', '2017')
    valdb   = composites_coco(config, 'test', '2017')
    auxdb   = composites_coco(config, 'aux', '2017')
    trainer = PuzzleTrainer(traindb)
    t0 = time()

    patch_dir_name = 'patch_feature_'+'with_bg' if config.use_patch_background else 'without_bg'
    if not osp.exists(osp.join(traindb.root_dir, patch_dir_name)):
        trainer.dump_shape_vectors(traindb)

    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(True)
    print("NN completes (time %.2fs)" % (time() - t0))
    t0 = time()
    if config.for_visualization:
        trainer.sample_for_vis(0, testdb, len(valdb.scenedb), nn_table=all_tables)
        trainer.sample_for_vis(0, auxdb, len(auxdb.scenedb), nn_table=all_tables)
    else:
        trainer.sample_for_eval(testdb, nn_table=all_tables)
        trainer.sample_for_eval(auxdb, nn_table=all_tables)
    print("Sampling completes (time %.2fs)" % (time() - t0))


def composites_demo(config):
    traindb = composites_coco(config, 'train', '2017')
    trainer = PuzzleTrainer(traindb)

    t0 = time()
    patch_dir_name = 'patch_feature_'+'with_bg' if config.use_patch_background else 'without_bg'
    if not osp.exists(osp.join(traindb.root_dir, patch_dir_name)):
        trainer.dump_shape_vectors(traindb)

    all_tables = AllCategoriesTables(traindb)
    all_tables.build_nntables_for_all_categories(True)
    print("NN completes (time %.2fs)" % (time() - t0))
    t0 = time()
    input_sentences = json_load('examples/composites_samples.json')
    trainer.sample_demo(input_sentences, all_tables)   
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

    # puzzle_model_inference_preparation(config)
    # puzzle_model_inference(config)
    composites_demo(config)