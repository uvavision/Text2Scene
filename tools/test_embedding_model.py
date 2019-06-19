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
from modules.embedding_model import EmbeddingModel
# from nntable import AllCategoriesTables

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def test_embedding_model(config):
    output_dir = osp.join(config.model_dir, 'test_embedding_model')
    maybe_create(output_dir)
    plt.switch_backend('agg')

    transformer = vol_transform(['background', 'foreground', 'negative'])
    db = coco(config, 'val', transform=transformer)
    # pca_table = AllCategoriesTables(db)
    # pca_table.run_PCAs_and_build_nntables_in_feature_space()

    loader = DataLoader(db,
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers)

    net = EmbeddingModel(db)

    net.eval()
    for cnt, batched in enumerate(loader):
        word_inds = batched['word_inds'].long()
        word_lens = batched['word_lens'].long()
        bg_images = batched['background'].float()
        fg_images = batched['foreground'].float()
        neg_images = batched['negative'].float()

        fg_inds = batched['fg_inds'].long()
        gt_inds = batched['out_inds'].long()
        gt_vecs = batched['out_vecs'].float()
        gt_msks = batched['out_msks'].float()

        fg_onehots = indices2onehots(fg_inds, config.output_vocab_size)

        inf_outs, _, positive_feats, negative_feats = net((word_inds, word_lens, bg_images, fg_onehots, fg_images, neg_images))
        obj_logits, coord_logits, attri_logits, patch_vectors, enc_msks, what_wei, where_wei = inf_outs
        print('teacher forcing')
        print('obj_logits ', obj_logits.size())
        print('coord_logits ', coord_logits.size())
        print('attri_logits ', attri_logits.size())
        print('patch_vectors ', patch_vectors.size())
        print('patch_vectors max:', torch.max(patch_vectors))
        print('patch_vectors min:', torch.min(patch_vectors))
        print('patch_vectors norm:', torch.norm(patch_vectors, dim=-2)[0,0,0])
        print('positive_feats ', positive_feats.size())
        print('negative_feats ', negative_feats.size())
        if config.what_attn:
            print('what_att_logits ', what_wei.size())
        if config.where_attn > 0:
            print('where_att_logits ', where_wei.size())
        print('----------------------')

        _, pred_vecs = net.collect_logits_and_vectors(inf_outs, gt_inds)
        print('pred_vecs', pred_vecs.size())
        print('*******************')

        # inf_outs, env = net.inference(word_inds, word_lens, -1, 0, 0, gt_inds, gt_vecs)
        # # inf_outs, env = net.inference(word_inds, word_lens, -1, 2.0, 0, None, None)
        # obj_logits, coord_logits, attri_logits, patch_vectors, enc_msks, what_wei, where_wei = inf_outs
        # print('scheduled sampling')
        # print('obj_logits ', obj_logits.size())
        # print('coord_logits ', coord_logits.size())
        # print('attri_logits ', attri_logits.size())
        # print('patch_vectors ', patch_vectors.size())
        # if config.what_attn:
        #     print('what_att_logits ', what_wei.size())
        # if config.where_attn > 0:
        #     print('where_att_logits ', where_wei.size())
        # print('----------------------')
        #
        #
        # sequences = env.batch_redraw(True)
        # for i in range(len(sequences)):
        #     sequence = sequences[i]
        #     image_idx = batched['image_index'][i]
        #     name = '%03d_'%i + str(image_idx).zfill(12)
        #     out_path = osp.join(output_dir, name+'.png')
        #     color = cv2.imread(batched['color_path'][i], cv2.IMREAD_COLOR)
        #     color, _, _ = create_squared_image(color)
        #
        #     fig = plt.figure(figsize=(32, 16))
        #     plt.suptitle(batched['sentence'][i], fontsize=30)
        #
        #     for j in range(min(len(sequence), 14)):
        #         plt.subplot(3, 5, j+1)
        #         partially_completed_img = clamp_array(sequence[j], 0, 255).astype(np.uint8)
        #         partially_completed_img = partially_completed_img[:,:,::-1]
        #         plt.imshow(partially_completed_img)
        #         plt.axis('off')
        #
        #     plt.subplot(3, 5, 15)
        #     plt.imshow(color[:,:,::-1])
        #     plt.axis('off')
        #
        #     fig.savefig(out_path, bbox_inches='tight')
        #     plt.close(fig)

        break


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_embedding_model(config)
