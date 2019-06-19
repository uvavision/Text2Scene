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

from datasets.cityscapes_syn import cityscapes_syn, cityscapes_tranformation
import torch, torchtext
from torch.utils.data import Dataset


def test_cityscapes_syn_dataloader(config):
    from torch.utils.data import DataLoader
    tranformer = cityscapes_tranformation()
    db = cityscapes_syn(config, 'train', tranformer)
    output_dir = osp.join(config.model_dir, 'test_cityscapes_syn_dataloader')
    maybe_create(output_dir)

    loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=0)

    for cnt, batched in enumerate(loader):
        print('proposal_image', batched['proposal_image'].size())
        print('proposal_label', batched['proposal_label'].size())
        print('proposal_mask', batched['proposal_mask'].size())
        print('gt_image', batched['gt_image'].size())
        print('gt_label', batched['gt_label'].size())

        plt.switch_backend('agg')
        gt_images       = batched['gt_image'].cpu().data.numpy()
        proposal_images = batched['proposal_image'].cpu().data.numpy()
        proposal_masks  = batched['proposal_mask'].cpu().data.numpy()
        proposal_labels = batched['proposal_label'].cpu().data.numpy()
        gt_labels = batched['gt_label'].cpu().data.numpy()
        # gt_images = gt_images.transpose((0, 2, 3, 1))
        # proposal_images = proposal_images.transpose((0, 2, 3, 1))

        for i in range(gt_images.shape[0]):
            gt_img = unnormalize(gt_images[i])
            proposal_img = unnormalize(proposal_images[i])
            proposal_label = db.decode_semantic_map(proposal_labels[i])
            gt_label = db.decode_semantic_map(gt_labels[i])
            proposal_msk = proposal_masks[i].squeeze()
            image_idx = batched['image_index'][i]
            name = '%03d_'%i + str(image_idx).zfill(8)
            out_path = osp.join(output_dir, name+'.png')

            fig = plt.figure(figsize=(32, 32))
            plt.subplot(3, 2, 1)
            plt.imshow(gt_img[:,:,::-1].astype(np.uint8))
            plt.axis('off')

            plt.subplot(3, 2, 2)
            plt.imshow(gt_label.astype(np.uint8))
            plt.axis('off')

            plt.subplot(3, 2, 3)
            plt.imshow(proposal_img[:,:,::-1].astype(np.uint8))
            plt.axis('off')

            plt.subplot(3, 2, 4)
            plt.imshow(proposal_label.astype(np.uint8))
            plt.axis('off')

            plt.subplot(3, 2, 5)
            plt.imshow(clamp_array(proposal_msk*255, 0, 255).astype(np.uint8))
            plt.axis('off')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        if cnt > 100:
            break


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_cityscapes_syn_dataloader(config)
