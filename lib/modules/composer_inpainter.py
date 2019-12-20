#!/usr/bin/env python

import os, sys, cv2, math
import random, json, logz
import numpy as np
import pickle, shutil
import os.path as osp
from copy import deepcopy

import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from composites_config import get_config
from composites_utils import *
from optim import Optimizer

from datasets.composites_loader import sequence_loader, patch_vol_loader
from nntable import AllCategoriesTables

from modules.puzzle_model import PuzzleModel
from modules.synthesis_model import SynthesisModel


class ComposerInpainter(object):
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg
        self.composer  = PuzzleModel(db)
        self.inpainter = SynthesisModel(self.cfg)
        if self.cfg.cuda:
            self.composer = self.composer.cuda()
            self.inpainter = self.inpainter.cuda()
        self.composer = self.load_pretrained_net(self.composer, 'composites_ckpts', self.cfg.composer_pretrained)
        self.inpainter = self.load_pretrained_net(self.inpainter, 'synthesis_ckpts', self.cfg.inpainter_pretrained)
    
    def load_pretrained_net(self, net, prefix, pretrained_name):
        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, prefix, pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(states['state_dict'])
        return net

    def decode_attention(self, word_inds, word_lens, att_logits):
        _, att_inds  = torch.topk(att_logits, 3, -1)
        att_inds  = att_inds.cpu().data.numpy()

        if len(word_inds.shape) > 1:
            lin_inds = []
            for i in range(word_inds.shape[0]):
                lin_inds.extend(word_inds[i, : word_lens[i]].tolist())
            vlen = len(lin_inds)
            npad = self.cfg.max_input_length * 3 - vlen
            lin_inds = lin_inds + [0] * npad
            # print(lin_inds)
            lin_inds = np.array(lin_inds).astype(np.int32)
        else:
            lin_inds = word_inds.copy()

        slen, _ = att_inds.shape
        attn_words = []
        for i in range(slen):
            w_inds = [lin_inds[x] for x in att_inds[i]]
            w_strs = [self.db.lang_vocab.index2word[x] for x in w_inds]
            attn_words = attn_words + [w_strs]

        return attn_words

    def sample_demo(self, input_sentences, nn_table):
        output_dir = osp.join(self.cfg.model_dir, 'inpainted_samples')
        maybe_create(output_dir)
        plt.switch_backend('agg')
        num_sents = len(input_sentences)
        out_h, out_w = self.cfg.output_image_size[1], self.cfg.output_image_size[0]
        for i in range(num_sents):
            sentence = input_sentences[i]
            ##############################################################
            # Inputs
            ##############################################################
            word_inds, word_lens = self.db.encode_sentence(sentence)
            input_inds_np = np.array(word_inds)
            input_lens_np = np.array(word_lens)
            input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
            input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
            if self.cfg.cuda:
                input_inds = input_inds.cuda()
                input_lens = input_lens.cuda()
            ##############################################################
            # Inference
            ##############################################################
            self.composer.eval()
            self.inpainter.eval()
            with torch.no_grad():
                inf_outs, env = self.composer.inference(input_inds, input_lens, -1, 1.0, 0, None, None, nn_table)
            frames, noises, masks, labels, env_info = env.batch_redraw(return_sequence=True)
            in_proposal = frames[0][-1]; noise = noises[0][-1]; in_mask = 255 * masks[0][-1]; in_label = labels[0][-1]
            in_proposal, _ = heuristic_collage(in_proposal, 83)
            # bounding box of the mask
            nonzero_pixels = cv2.findNonZero(in_mask.astype(np.uint8))
            x,y,w,h = cv2.boundingRect(nonzero_pixels)
            xyxy = np.array([int(x), int(y), int(x+w), int(y+h)])
            in_mask  = in_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            in_label = in_label[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] 
            in_proposal = in_proposal[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] 
            # print('in_proposal', in_proposal.shape)
            in_proposal = cv2.resize(in_proposal, (out_h, out_w), interpolation = cv2.INTER_CUBIC)
            in_mask = cv2.resize(in_mask, (out_h, out_w), interpolation = cv2.INTER_NEAREST)
            in_label = cv2.resize(in_label, (out_h, out_w), interpolation = cv2.INTER_NEAREST)
            in_mask = in_mask.astype(np.float32)
            in_mask = 255.0 - in_mask
            in_vol = np.concatenate((in_label[..., None], in_mask[..., None], in_proposal[:,:,::-1].copy()), -1)

            in_vol = torch.from_numpy(in_vol).unsqueeze(0)
            out, _, _, _ = self.inpainter(in_vol, False, None)
            out = out[0].cpu().data.numpy().transpose((1,2,0))
            out = clamp_array(out, 0, 255).astype(np.uint8)
            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(32, 32))
            plt.suptitle(sentence, fontsize=40)
            for j in range(len(frames[0])):
                plt.subplot(4, 4, j+1)
                # plt.title(subtitle, fontsize=30)
                if self.cfg.use_color_volume:
                    vis_img, _ = heuristic_collage(frames[0][j], 83)
                else:
                    vis_img = frames[0][j][:,:,-3:]
                vis_img = clamp_array(vis_img[ :, :, ::-1], 0, 255).astype(np.uint8)
                plt.imshow(vis_img)
                plt.axis('off')
            plt.subplot(4, 4, 16)
            plt.imshow(out)
            plt.axis('off')

            out_path = osp.join(output_dir, '%09d.png'%i)
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        
