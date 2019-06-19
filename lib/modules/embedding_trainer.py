#!/usr/bin/env python

import os, sys, cv2, math
import random, json, logz
import numpy as np
import pickle
import os.path as osp
from copy import deepcopy
from config import get_config
import matplotlib.pyplot as plt
from glob import glob
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.embedding_model import EmbeddingModel
from optim import Optimizer

from datasets.coco_shape import coco_shape


class EmbeddingTrainer(object):
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg
        self.net = EmbeddingModel(db)
        if self.cfg.pretrained is not None:
            self.load_pretrained_net(self.cfg.pretrained)
        if self.cfg.cuda:
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def load_pretrained_net(self, pretrained_name):
        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, 'embedding_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(states)

    def batch_data(self, entry):
        ################################################
        # Inputs
        ################################################
        input_inds = entry['word_inds'].long()
        input_lens = entry['word_lens'].long()
        fg_inds = entry['fg_inds'].long()
        bg_imgs = entry['background']
        fg_imgs = entry['foreground']
        neg_imgs = entry['negative']
        fg_onehots = indices2onehots(fg_inds, self.cfg.output_vocab_size)

        ################################################
        # Outputs
        ################################################
        gt_inds = entry['out_inds'].long()
        # gt_vecs = entry['out_vecs'].float()
        gt_msks = entry['out_msks'].float()
        gt_scene_inds = entry['scene_index'].long().numpy()

        if self.cfg.cuda:
            input_inds = input_inds.cuda(non_blocking=True)
            input_lens = input_lens.cuda(non_blocking=True)
            fg_onehots = fg_onehots.cuda(non_blocking=True)
            bg_imgs    = bg_imgs.cuda(non_blocking=True)
            fg_imgs    = fg_imgs.cuda(non_blocking=True)
            neg_imgs   = neg_imgs.cuda(non_blocking=True)
            gt_inds = gt_inds.cuda(non_blocking=True)
            # gt_vecs = gt_vecs.cuda()
            gt_msks = gt_msks.cuda(non_blocking=True)

        # bg_imgs = sequence_color_volumn_preprocess(bg_imgs, len(self.db.classes))
        # bg_imgs = (bg_imgs-128.0).permute(0,1,4,2,3)
        # fg_imgs = sequence_onehot_volumn_preprocess(fg_imgs, len(self.db.classes))
        # fg_imgs = (fg_imgs-128.0).permute(0,1,4,2,3)
        # neg_imgs = sequence_onehot_volumn_preprocess(neg_imgs, len(self.db.classes))
        # neg_imgs = (neg_imgs-128.0).permute(0,1,4,2,3)

        return input_inds, input_lens, fg_onehots, bg_imgs, fg_imgs, neg_imgs, gt_inds, gt_msks, gt_scene_inds

    def dump_shape_vectors(self, train_db):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        db = coco_shape(train_db)
        loader = DataLoader(db,
            batch_size=128, shuffle=False,
            num_workers=1, pin_memory=True)

        for cnt, batched in enumerate(loader):
            start = time()
            patch_inds = batched['patch_ind']
            patch_vols = batched['patch_vol']
            if self.cfg.cuda:
                patch_vols = patch_vols.cuda(non_blocking=True)
            patch_features = net.shape_encoder.batch_forward(patch_vols)
            patch_features = patch_features.cpu().data.numpy()
            for i in range(patch_vols.size(0)):
                patch = db.patchdb[patch_inds[i]]
                patch_feature_path = patch['features_path']
                patch_feature_dir = osp.dirname(patch_feature_path)
                maybe_create(patch_feature_dir)
                features = patch_features[i].flatten()
                with open(patch_feature_path, 'wb') as fid:
                    pickle.dump(features, fid, pickle.HIGHEST_PROTOCOL)
            print('current_ind: %d, time consumed: %f'%(cnt, time() - start))
            
    def evaluate(self, inf_outs, pos_vecs, neg_vecs, ref_inds, ref_msks):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        ####################################################################
        # Prediction loss
        ####################################################################
        _, _, _, _, enc_msks, what_wei, where_wei = inf_outs
        _, pred_vecs = net.collect_logits_and_vectors(inf_outs, ref_inds)

        bsize, slen, vsize = pred_vecs.size()

        ####################################################################
        # Regression loss
        ####################################################################
        embed_metric = nn.TripletMarginLoss(
            margin=0.5,
            p=2,
            eps=1e-06,
            swap=False,
            size_average=None,
            reduction='none')

        embed_loss = embed_metric(
            pred_vecs.view(bsize*slen, vsize),
            pos_vecs.view(bsize*slen, vsize),
            neg_vecs.view(bsize*slen, vsize)).view(bsize, slen)
        embed_mask = ref_msks[:,:,1]
        embed_loss = torch.sum(embed_loss * embed_mask)/(torch.sum(embed_mask) + self.cfg.eps)
        # print('embed_loss', embed_loss)

        ####################################################################
        # doubly stochastic attn loss
        ####################################################################
        attn_loss = 0
        encoder_msks = enc_msks
        if self.cfg.what_attn:
            obj_msks = ref_msks[:,:,0].unsqueeze(-1)
            what_att_logits = what_wei
            raw_obj_att_loss = torch.mul(what_att_logits, obj_msks)
            raw_obj_att_loss = torch.sum(raw_obj_att_loss, dim=1)
            obj_att_loss = raw_obj_att_loss - encoder_msks
            obj_att_loss = torch.sum(obj_att_loss ** 2, dim=-1)
            obj_att_loss = torch.mean(obj_att_loss)
            attn_loss = attn_loss + obj_att_loss
        attn_loss = self.cfg.attn_loss_weight * attn_loss

        return embed_loss, attn_loss

    def train(self, train_db, val_db, test_db):
        ##################################################################
        ## Optimizer
        ##################################################################
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        image_encoder_trainable_paras = \
            filter(lambda p: p.requires_grad, net.image_encoder.parameters())
        raw_optimizer = optim.Adam([
                {'params': image_encoder_trainable_paras},
                {'params': net.text_encoder.embedding.parameters(), 'lr': self.cfg.finetune_lr},
                {'params': net.text_encoder.rnn.parameters()},
                {'params': net.what_decoder.parameters()},
                {'params': net.where_decoder.parameters()},
                {'params': net.shape_encoder.parameters()},
            ], lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, factor=0.8, patience=3)
        # scheduler = optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=3, gamma=0.8)
        # optimizer.set_scheduler(scheduler)

        ##################################################################
        ## LOG
        ##################################################################
        logz.configure_output_dir(self.cfg.model_dir)
        logz.save_config(self.cfg)

        ##################################################################
        ## Main loop
        ##################################################################
        start = time()
        min_val_loss = 100000000
        for epoch in range(self.cfg.n_epochs):
            ##################################################################
            ## Training
            ##################################################################
            torch.cuda.empty_cache()
            train_loss = self.train_epoch(train_db, optimizer, epoch)

            ##################################################################
            ## Validation
            ##################################################################
            torch.cuda.empty_cache()
            val_loss = self.validate_epoch(val_db, epoch)

            ##################################################################
            ## Logging
            ##################################################################

            # update optim scheduler
            current_val_loss = np.mean(val_loss[:,0])
            # optimizer.update(current_val_loss, epoch)
            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)
            logz.log_tabular("AverageLoss",         np.mean(train_loss[:,0]))
            logz.log_tabular("AverageEmbedLoss",    np.mean(train_loss[:,1]))
            logz.log_tabular("AverageAttnLoss",     np.mean(train_loss[:,2]))
            logz.log_tabular("ValAverageLoss",      np.mean(val_loss[:,0]))
            logz.log_tabular("ValAverageEmbedLoss", np.mean(val_loss[:,1]))
            logz.log_tabular("ValAverageAttnLoss",  np.mean(val_loss[:,2]))
            logz.dump_tabular()

            ##################################################################
            ## Checkpoint
            ##################################################################
            if min_val_loss > current_val_loss:
                min_val_loss = current_val_loss
                # log_info = [np.mean(val_loss), np.mean(val_accu)]
                # self.save_checkpoint(epoch, log_info)
                self.save_best_checkpoint()
                torch.cuda.empty_cache()

    def train_epoch(self, train_db, optimizer, epoch):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        train_db.cfg.sent_group = -1
        train_loader = DataLoader(train_db,
            batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True)

        all_losses = []

        for cnt, batched in enumerate(train_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            input_inds, input_lens, fg_onehots, bg_imgs, fg_imgs, neg_imgs, gt_inds, gt_msks, gt_scene_inds = \
                self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.train()
            self.net.zero_grad()

            inputs = (input_inds, input_lens, bg_imgs, fg_onehots, fg_imgs, neg_imgs)
            inf_outs, _, pos_vecs, neg_vecs = self.net(inputs)
            embed_loss, attn_loss = self.evaluate(inf_outs, pos_vecs, neg_vecs, gt_inds, gt_msks)

            loss = embed_loss + attn_loss
            loss.backward()
            optimizer.step()

            ##################################################################
            ## Collect info
            ##################################################################
            all_losses.append(np.array([loss.cpu().data.item(), embed_loss.cpu().data.item(), attn_loss.cpu().data.item()]))

            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Epoch %03d, iter %07d:'%(epoch, cnt))
                tmp_losses = np.stack(all_losses, 0)
                print('losses: ', np.mean(tmp_losses[:,0]), np.mean(tmp_losses[:,1]), np.mean(tmp_losses[:,2]))
                print('-------------------------')

        all_losses = np.stack(all_losses, 0)

        return all_losses

    def validate_epoch(self, val_db, epoch):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        all_losses = []
        # initial experiment, just use one group of sentence
        for G in range(1):
            val_db.cfg.sent_group = G
            val_loader = DataLoader(val_db,
                batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, pin_memory=True)

            for cnt, batched in enumerate(val_loader):
                ##################################################################
                ## Batched data
                ##################################################################
                input_inds, input_lens, fg_onehots, bg_imgs, fg_imgs, neg_imgs, gt_inds, gt_msks, gt_scene_inds = \
                    self.batch_data(batched)

                ##################################################################
                ## Validate one step
                ##################################################################
                self.net.eval()
                with torch.no_grad():
                    inputs = (input_inds, input_lens, bg_imgs, fg_onehots, fg_imgs, neg_imgs)
                    inf_outs, _, pos_vecs, neg_vecs = self.net(inputs)
                    embed_loss, attn_loss = self.evaluate(inf_outs, pos_vecs, neg_vecs, gt_inds, gt_msks)

                loss = embed_loss + attn_loss

                all_losses.append(np.array([loss.cpu().data.item(), embed_loss.cpu().data.item(), attn_loss.cpu().data.item()]))
                print(epoch, G, cnt)
        all_losses = np.stack(all_losses, 0)
        return all_losses

    def save_checkpoint(self, epoch, log):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'embedding_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_name = "ckpt-%03d-%.4f-%.4f.pkl" % (epoch, log[0], log[1])
        torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))

    def save_best_checkpoint(self):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'embedding_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_name = "ckpt-best.pkl"
        torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))

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

    def sample(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'vis')
        maybe_create(output_dir)

        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if random_or_not:
            indices = np.random.permutation(range(len(test_db)))
        else:
            indices = range(len(test_db))
        indices = indices[:min(N, len(test_db))]

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for i in indices:
            entry    = test_db[i]
            gt_scene = test_db.scenedb[i]

            gt_img = cv2.imread(entry['image_path'], cv2.IMREAD_COLOR)
            gt_img, _, _ = create_squared_image(gt_img)
            gt_img = cv2.resize(gt_img, (self.cfg.input_image_size[0], self.cfg.input_image_size[1]))

            ##############################################################
            # Inputs
            ##############################################################
            input_inds_np = np.array(entry['word_inds'])
            input_lens_np = np.array(entry['word_lens'])
            gt_inds_np = np.array(entry['out_inds'])


            input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
            input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
            gt_inds = torch.from_numpy(gt_inds_np).long().unsqueeze(0)

            if self.cfg.cuda:
                input_inds = input_inds.cuda()
                input_lens = input_lens.cuda()
                gt_inds = gt_inds.cuda()

            ##############################################################
            # Inference
            ##############################################################
            self.net.eval()
            with torch.no_grad():
                inf_outs, env = net.inference(input_inds, input_lens, 1000, 0.0, 0, gt_inds, None)
            frames = env.batch_redraw(return_sequence=True)[0]
            _, _, _, _, _, what_wei, where_wei = inf_outs

            if self.cfg.what_attn:
                what_attn_words = self.decode_attention(
                    input_inds_np, input_lens_np, what_wei.squeeze(0))
            if self.cfg.where_attn > 0:
                where_attn_words = self.decode_attention(
                    input_inds_np, input_lens_np, where_wei.squeeze(0))

            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(32, 32))
            plt.suptitle(entry['sentence'], fontsize=40)
            for j in range(len(frames)):
                subtitle = ''
                if self.cfg.what_attn:
                    subtitle = subtitle + ' '.join(what_attn_words[j])
                if self.cfg.where_attn > 0:
                    subtitle = subtitle + '\n' + ' '.join(where_attn_words[j])

                plt.subplot(4, 4, j+1)
                plt.title(subtitle, fontsize=30)
                vis_img, _ = self.db.heuristic_collage(frames[j])
                vis_img = clamp_array(vis_img[ :, :, ::-1], 0, 255).astype(np.uint8)
                plt.imshow(vis_img)
                plt.axis('off')
            plt.subplot(4, 4, 16)
            plt.imshow(gt_img[:, :, ::-1])
            plt.axis('off')

            name = osp.splitext(osp.basename(entry['image_path']))[0]
            out_path = osp.join(output_dir, name+'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print('sampling: %d, %d'%(epoch, i))