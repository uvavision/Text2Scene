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

from composites_config import get_config
from composites_utils import *
from datasets.composites_coco import composites_coco

from datasets.composites_loader import sequence_loader
import torch, torchtext
from torch.utils.data import Dataset, DataLoader

from nntable import AllCategoriesTables


def test_coco_dataset(config):
    db = composites_coco(config, 'train')
    # with open('category_ind_to_class_ind.json', 'w') as fp:
    #     json.dump(db.category_ind_to_class_ind, fp, indent=4, sort_keys=True)
    valdb = composites_coco(config, 'val')
    testdb = composites_coco(config, 'test')
    auxdb = composites_coco(config, 'aux')
    print(len(db.scenedb), len(valdb.scenedb), len(testdb.scenedb), len(auxdb.scenedb))
    # print(len(db.classes))
    # print(config.output_vocab_size)
    # print(len(db.patchdb), len(valdb.patchdb), len(testdb.patchdb))
    # output_dir = osp.join(config.model_dir, 'test_coco_dataset')
    # maybe_create(output_dir)

    # indices = np.random.permutation(range(len(db.scenedb)))
    # indices = indices[:config.n_samples]

    # plt.switch_backend('agg')
    # for i in indices:
    #     entry = db.scenedb[i]
    #     comps = db.render_reference_scene(entry)
    #     image_idx = entry['image_index']
    #     name = '%03d_'%i + str(image_idx).zfill(12)
    #     out_path = osp.join(output_dir, name+'.png')
    #     color = cv2.imread(db.color_path_from_index(image_idx), cv2.IMREAD_COLOR)
    #     color, _, _ = create_squared_image(color)

    #     fig = plt.figure(figsize=(32, 16))
    #     plt.suptitle(entry['captions'][0], fontsize=30)

    #     for j in range(min(len(comps), 16)):
    #         plt.subplot(4, 5, j+1)
    #         partially_completed_img = clamp_array(comps[j], 0, 255).astype(np.uint8)
    #         partially_completed_img = partially_completed_img[:,:,::-1]
    #         plt.imshow(partially_completed_img)
    #         plt.axis('off')

    #     plt.subplot(4, 5, 20)
    #     plt.imshow(color[:,:,::-1])
    #     plt.axis('off')

    #     fig.savefig(out_path, bbox_inches='tight')
    #     plt.close(fig)


def test_coco_dataloader(config):
    db = composites_coco(config, 'train')

    all_tables = AllCategoriesTables(db)
    all_tables.build_nntables_for_all_categories(True)

    sequence_db = sequence_loader(db, all_tables)
    output_dir = osp.join(config.model_dir, 'test_coco_dataloader')
    maybe_create(output_dir)

    loader = DataLoader(sequence_db,
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    start = time()
    for cnt, batched in enumerate(loader):
        x = batched['background'].float()
        y = batched['foreground'].float()
        z = batched['negative'].float()

        x = sequence_color_volumn_preprocess(x, len(db.classes))
        y = sequence_onehot_volumn_preprocess(y, len(db.classes))
        z = sequence_onehot_volumn_preprocess(z, len(db.classes))

        # cv2.imwrite('mask0.png', y[0,2,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask1.png', y[1,2,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask2.png', y[2,2,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('mask3.png', y[3,2,:,:,-4].cpu().data.numpy())
        # cv2.imwrite('label0.png', y[0,2,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label1.png', y[1,2,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label2.png', y[2,2,:,:,3].cpu().data.numpy())
        # cv2.imwrite('label3.png', y[3,2,:,:,3].cpu().data.numpy())
        # cv2.imwrite('color0.png', y[0,2,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color1.png', y[1,2,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color2.png', y[2,2,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('color3.png', y[3,2,:,:,-3:].cpu().data.numpy())
        # cv2.imwrite('bg0.png', x[0,3,:,:,9:12].cpu().data.numpy())
        # cv2.imwrite('bg1.png', x[1,3,:,:,9:12].cpu().data.numpy())
        # cv2.imwrite('bg2.png', x[2,3,:,:,9:12].cpu().data.numpy())
        # cv2.imwrite('bg3.png', x[3,3,:,:,9:12].cpu().data.numpy())

        x = (x-128.0).permute(0,1,4,2,3)
        y = (y-128.0).permute(0,1,4,2,3)
        z = (z-128.0).permute(0,1,4,2,3)

        print('background', x.size())
        print('foreground', y.size())
        print('negative', z.size())
        print('word_inds', batched['word_inds'].size())
        print('word_lens', batched['word_lens'].size())
        print('fg_inds',   batched['fg_inds'].size())
        print('patch_inds', batched['patch_inds'].size())
        print('out_inds', batched['out_inds'].size())
        print('out_msks', batched['out_msks'].size())
        print('foreground_resnets', batched['foreground_resnets'].size())
        print('negative_resnets', batched['negative_resnets'].size())

        print('foreground_resnets', batched['foreground_resnets'][0, 0])
        print('negative_resnets', batched['negative_resnets'][0, 0])
        print('out_msks', batched['out_msks'][0])
        print('patch_inds', batched['patch_inds'][0])

        plt.switch_backend('agg')
        bg_images  = x
        fg_images  = y
        neg_images = z

        bsize, ssize, n, h, w = bg_images.size()
        bg_images = bg_images.view(bsize*ssize, n, h, w)
        bg_images = tensors_to_vols(bg_images)
        bg_images = bg_images.reshape(bsize, ssize, h, w, n)

        bsize, ssize, n, h, w = fg_images.size()
        fg_images = fg_images.view(bsize*ssize, n, h, w)
        fg_images = tensors_to_vols(fg_images)
        fg_images = fg_images.reshape(bsize, ssize, h, w, n)

        bsize, ssize, n, h, w = neg_images.size()
        neg_images = neg_images.view(bsize*ssize, n, h, w)
        neg_images = tensors_to_vols(neg_images)
        neg_images = neg_images.reshape(bsize, ssize, h, w, n)

        for i in range(bsize):
            bg_seq = bg_images[i]
            fg_seq = fg_images[i]
            neg_seq = neg_images[i]
            image_idx = batched['image_index'][i]
            fg_inds = batched['fg_inds'][i]
            name = '%03d_'%i + str(image_idx).zfill(12)
            out_path = osp.join(output_dir, name+'.png')
            color = cv2.imread(batched['image_path'][i], cv2.IMREAD_COLOR)
            color, _, _ = create_squared_image(color)

            fig = plt.figure(figsize=(48, 32))
            plt.suptitle(batched['sentence'][i], fontsize=30)

            for j in range(min(len(bg_seq), 15)):
                bg, _ = heuristic_collage(bg_seq[j], 83)
                bg_mask = 255 * np.ones((bg.shape[1], bg.shape[0]))
                row, col = np.where(np.sum(np.absolute(bg), -1) == 0)
                bg_mask[row, col] = 0
                # bg = bg_seq[j][:,:,-3:]
                # bg_mask = bg_seq[j][:,:,-4]
                bg_mask = np.repeat(bg_mask[...,None], 3, -1)
                fg_color = fg_seq[j][:,:,-3:]
                # fg_mask = fg_seq[j][:,:,fg_inds[j+1]]
                fg_mask = fg_seq[j][:,:,-4]
                neg_color = neg_seq[j][:,:,-3:]
                # neg_mask = neg_seq[j][:,:,fg_inds[j+1]]
                neg_mask = neg_seq[j][:,:,-4]

                color_pair = np.concatenate((fg_color,neg_color), 1)
                mask_pair = np.concatenate((fg_mask, neg_mask), 1)
                mask_pair = np.repeat(mask_pair[...,None], 3, -1)
                patch = np.concatenate((color_pair,mask_pair), 0)
                patch = cv2.resize(patch, (bg.shape[1], bg.shape[0]))

                partially_completed_img = np.concatenate((bg, bg_mask, patch), 1)
                partially_completed_img = clamp_array(partially_completed_img, 0, 255).astype(np.uint8)
                partially_completed_img = partially_completed_img[:,:,::-1]
                plt.subplot(4, 4, j+1)
                plt.imshow(partially_completed_img)
                plt.axis('off')

            plt.subplot(4, 4, 16)
            plt.imshow(color[:,:,::-1])
            plt.axis('off')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        if cnt == 3:
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

    # test_coco_dataset(config)
    test_coco_dataloader(config)
