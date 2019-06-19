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
from simulator import simulator

from datasets.coco import coco
from datasets.coco_loader import sequence_loader
import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools import mask as COCOmask

###############################################################
# Draw
###############################################################
def test_step_by_step(config):
    db = coco(config, 'train', '2017')
    output_dir = osp.join(config.model_dir, 'test_step_by_step')
    maybe_create(output_dir)

    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)

    seq_db = sequence_loader(db, all_tables)
    env = simulator(db, config.batch_size, all_tables)
    env.reset()

    loader = DataLoader(seq_db,
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        out_inds = batched['out_inds'].long().numpy()
        out_vecs = batched['out_vecs'].float().numpy()

        sequences = []
        for i in range(out_inds.shape[1]):
            frames = env.batch_render_to_pytorch(out_inds[:, i], out_vecs[:, i])
            sequences.append(frames)
        sequences = torch.stack(sequences, dim=1)
        # sequences = [tensors_to_vols(x) for x in sequences]

        for i in range(len(sequences)):
            sequence = sequences[i]
            image_idx = batched['image_index'][i]
            name = '%03d_'%i + str(image_idx).zfill(12)
            out_path = osp.join(output_dir, name+'.png')
            color = cv2.imread(batched['image_path'][i], cv2.IMREAD_COLOR)
            color, _, _ = create_squared_image(color)

            fig = plt.figure(figsize=(32, 32))
            plt.suptitle(batched['sentence'][i], fontsize=30)

            for j in range(min(len(sequence), 14)):
                plt.subplot(4, 4, j+1)
                seq_np = sequence[j].cpu().data.numpy()
                if config.use_color_volume:
                    partially_completed_img, _ = heuristic_collage(seq_np, 83)
                else:
                    partially_completed_img = seq_np[:,:,-3:]
                partially_completed_img = clamp_array(partially_completed_img, 0, 255).astype(np.uint8)
                partially_completed_img = partially_completed_img[:,:,::-1]
                plt.imshow(partially_completed_img)
                plt.axis('off')

            plt.subplot(4, 4, 16)
            plt.imshow(color[:,:,::-1])
            plt.axis('off')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        break


def test_redraw(config):
    transformer = image_normalize('background')
    db = coco(config, 'train', transform=transformer)
    output_dir = osp.join(config.model_dir, 'test_redraw')
    maybe_create(output_dir)

    env = simulator(db, config.batch_size)
    env.reset()

    loader = DataLoader(db,
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    for cnt, batched in enumerate(loader):
        out_inds = batched['out_inds'].long().numpy()
        out_vecs = batched['out_vecs'].float().numpy()

        # sequences = []
        for i in range(out_inds.shape[1]):
            frames = env.batch_render_to_pytorch(out_inds[:, i], out_vecs[:, i])
            # sequences.append(frames)
        # sequences = torch.stack(sequences, dim=1)
        sequences = env.batch_redraw(True)
        # sequences = [tensors_to_imgs(x) for x in sequences]

        for i in range(len(sequences)):
            sequence = sequences[i]
            image_idx = batched['image_index'][i]
            name = '%03d_'%i + str(image_idx).zfill(12)
            out_path = osp.join(output_dir, name+'.png')
            color = cv2.imread(batched['color_path'][i], cv2.IMREAD_COLOR)
            color, _, _ = create_squared_image(color)

            fig = plt.figure(figsize=(32, 16))
            plt.suptitle(batched['sentence'][i], fontsize=30)

            for j in range(min(len(sequence), 14)):
                plt.subplot(3, 5, j+1)
                partially_completed_img = clamp_array(sequence[j], 0, 255).astype(np.uint8)
                partially_completed_img = partially_completed_img[:,:,::-1]
                plt.imshow(partially_completed_img)
                plt.axis('off')

            plt.subplot(3, 5, 15)
            plt.imshow(color[:,:,::-1])
            plt.axis('off')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        break


if __name__ == '__main__':
    from config import get_config
    from utils import prepare_directories
    import random

    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_step_by_step(config)
    # test_redraw(config)
