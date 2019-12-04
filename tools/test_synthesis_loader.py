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
from nltk.tokenize import word_tokenize
from collections import OrderedDict

from composites_utils import *
from composites_config import get_config

from datasets.composites_coco import composites_coco
from datasets.composites_loader import synthesis_loader, proposal_loader

import torch, torchtext
from torch.utils.data import Dataset, DataLoader

from nntable import AllCategoriesTables


def test_syn_dataloader(config):
    db = composites_coco(config, 'train', '2017')

    syn_loader = synthesis_loader(db)
    output_dir = osp.join(config.model_dir, 'test_syn_dataloader')
    maybe_create(output_dir)

    loader = DataLoader(syn_loader,
        batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers)

    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['input_vol'].float()
        y = batched['gt_image'].float()
        z = batched['gt_label'].float()

        if config.use_color_volume:
            x = batch_color_volumn_preprocess(x, len(db.classes))
        else:
            x = batch_onehot_volumn_preprocess(x, len(db.classes))
        print('input_vol', x.size())
        print('gt_image',  y.size())
        print('gt_label',  z.size())

        # cv2.imwrite('mask0.png', x[0,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask1.png', x[1,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask2.png', x[2,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask3.png', x[3,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('label0.png', x[0,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label1.png', x[1,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label2.png', x[2,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label3.png', x[3,:,:,3].cpu().data.numpy())
        # cv2.imwrite('color0.png', x[0,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color1.png', x[1,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color2.png', x[2,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color3.png', x[3,:,:,-3:].cpu().data.numpy())
        
        x = (x-128.0).permute(0,3,1,2)


        plt.switch_backend('agg')
        x = tensors_to_vols(x)
        for i in range(x.shape[0]):
            image_idx = batched['image_index'][i]
            name = '%03d_'%i + str(image_idx).zfill(12)
            out_path = osp.join(output_dir, name+'.png')

            if config.use_color_volume:
                proposal = x[i, :, :, 12:15]
                mask = x[i, :, :, :3]
                person = x[i, :, :, 9:12]
                other = x[i, :, :, 15:18]
                gt_color = y[i]
                gt_label = z[i]
                gt_label = np.repeat(gt_label[..., None], 3, -1) 
            else:
                proposal = x[i, :, :, -3:]
                mask = x[i, :, :, -4]
                mask = np.repeat(mask[..., None], 3, -1)
                person = x[i, :, :, 3]
                person = np.repeat(person[..., None], 3, -1)
                other = x[i, :, :, 5]
                other = np.repeat(other[..., None], 3, -1)
                gt_color = y[i]
                gt_label = z[i]
                gt_label = np.repeat(gt_label[..., None], 3, -1)

            r1 = np.concatenate((proposal, mask, person), 1)
            r2 = np.concatenate((gt_color, gt_label, other), 1)
            out = np.concatenate((r1, r2), 0).astype(np.uint8)

            fig = plt.figure(figsize=(32, 32))
            plt.imshow(out[:,:,:])
            plt.axis('off')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        if cnt == 1:
            break
    print("Time", time() - start)


def test_proposal_dataloader(config):
    db = coco(config, 'val', '2017')
    proposal_db = proposal_loader(db)
    output_dir = osp.join(config.model_dir, 'test_proposal_dataloader')
    maybe_create(output_dir)

    loader = DataLoader(proposal_db,
        batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers)

    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['input_vol'].float()
        y = batched['image_index'].long()
        z = batched['box'].long()

        if config.use_color_volume:
            x = batch_color_volumn_preprocess(x, len(db.classes))
        else:
            x = batch_onehot_volumn_preprocess(x, len(db.classes))
        print('input_vol', x.size())

        # cv2.imwrite('mask0.png', x[0,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask1.png', x[1,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask2.png', x[2,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask3.png', x[3,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('label0.png', x[0,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label1.png', x[1,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label2.png', x[2,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label3.png', x[3,:,:,3].cpu().data.numpy())
        # cv2.imwrite('color0.png', x[0,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color1.png', x[1,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color2.png', x[2,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color3.png', x[3,:,:,-3:].cpu().data.numpy())
        
        x = (x-128.0).permute(0,3,1,2)


        plt.switch_backend('agg')
        x = tensors_to_vols(x)
        for i in range(x.shape[0]):
            image_idx = y[i].item()
            name = '%03d_'%i + str(image_idx).zfill(12)
            out_path = osp.join(output_dir, name+'.png')

            if config.use_color_volume:
                proposal = x[i, :, :, 12:15]
                mask = x[i, :, :, :3]
                person = x[i, :, :, 9:12]
                other = x[i, :, :, 15:18]
            else:
                proposal = x[i, :, :, -3:]
                mask = x[i, :, :, -4]
                mask = np.repeat(mask[..., None], 3, -1)
                person = x[i, :, :, 3]
                person = np.repeat(person[..., None], 3, -1)
                other = x[i, :, :, 5]
                other = np.repeat(other[..., None], 3, -1)

            r1 = np.concatenate((proposal, mask), 1)
            r2 = np.concatenate((person, other), 1)
            out = np.concatenate((r1, r2), 0).astype(np.uint8)

            fig = plt.figure(figsize=(32, 32))
            plt.imshow(out)
            plt.axis('off')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        if cnt == 1:
            break
    print("Time", time() - start)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_syn_dataloader(config)
    # test_proposal_dataloader(config)
