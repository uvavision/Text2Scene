#!/usr/bin/env python

import os, sys, cv2, math
import random, json, logz
import numpy as np
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

from modules.scene_model import SceneModel
from optim import Optimizer


class SupervisedTrainer(object):
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg
        net = SceneModel(db)
        if self.cfg.cuda:
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            net = net.cuda()
        self.net = net
        if self.cfg.pretrained is not None:
            self.load_pretrained_net(self.cfg.pretrained)

    def load_pretrained_net(self, pretrained_name):
        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, 'scene_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(states)

    def batch_data(self, entry):
        ################################################
        # Inputs
        ################################################
        input_inds = entry['word_inds'].long()
        input_lens = entry['word_lens'].long()
        fg_inds = entry['fg_inds'].long()
        bg_imgs = entry['background'].float()
        fg_onehots = indices2onehots(fg_inds, self.cfg.output_vocab_size)

        ################################################
        # Outputs
        ################################################
        gt_inds = entry['out_inds'].long()
        gt_vecs = entry['out_vecs'].float()
        gt_msks = entry['out_msks'].float()
        gt_scene_inds = entry['scene_index'].long().numpy()

        if self.cfg.cuda:
            input_inds = input_inds.cuda()
            input_lens = input_lens.cuda()
            fg_onehots = fg_onehots.cuda()
            bg_imgs = bg_imgs.cuda()
            gt_inds = gt_inds.cuda()
            gt_vecs = gt_vecs.cuda()
            gt_msks = gt_msks.cuda()

        return input_inds, input_lens, bg_imgs, fg_onehots, gt_inds, gt_vecs, gt_msks, gt_scene_inds

    def evaluate(self, inf_outs, ref_inds, ref_vecs, ref_msks):

        ####################################################################
        # Prediction loss
        ####################################################################
        _, _, _, _, enc_msks, what_wei, where_wei = inf_outs

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        logits, pca_vectors = net.collect_logits_and_vectors(inf_outs, ref_inds)
        bsize, slen, _ = logits.size()
        loss_wei = [
            self.cfg.obj_loss_weight, \
            self.cfg.coord_loss_weight, \
            self.cfg.scale_loss_weight, \
            self.cfg.ratio_loss_weight
        ]
        loss_wei = torch.from_numpy(np.array(loss_wei)).float()
        if self.cfg.cuda:
            loss_wei = loss_wei.cuda()
        loss_wei = loss_wei.view(1,1,4)
        loss_wei = loss_wei.expand(bsize, slen, 4)

        pred_loss = -torch.log(logits.clamp(min=self.cfg.eps)) * loss_wei * ref_msks
        pred_loss = torch.sum(pred_loss)/(torch.sum(ref_msks) + self.cfg.eps)

        ####################################################################
        # Regression loss
        ####################################################################
        regression_mask = ref_msks[:,:,1]
        regression_loss = torch.sum(torch.abs(pca_vectors - ref_vecs), -1) * regression_mask
        regression_loss = torch.sum(regression_loss)/(torch.sum(regression_mask) + self.cfg.eps)
        regression_loss = self.cfg.regression_loss_weight * regression_loss
        # print('regression_loss', regression_loss)

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

        # if self.cfg.where_attn > 0:
        #     pos_msks = ref_msks[:,:,1].unsqueeze(-1)
        #     where_att_logits = where_wei
        #     raw_pos_att_loss = torch.mul(where_att_logits, pos_msks)
        #     raw_pos_att_loss = torch.sum(raw_pos_att_loss, dim=1)
        #     pos_att_loss = raw_pos_att_loss - encoder_msks
        #     pos_att_loss = torch.sum(pos_att_loss ** 2, dim=-1)
        #     pos_att_loss = torch.mean(pos_att_loss)
        #     attn_loss = attn_loss + pos_att_loss

        attn_loss = self.cfg.attn_loss_weight * attn_loss

        # eos_loss = 0
        # if self.cfg.what_attn and self.cfg.eos_loss_weight > 0:
        #     # print('-------------------')
        #     # print('obj_msks: ', obj_msks.size())
        #     inds_1 = torch.sum(obj_msks, 1, keepdim=True) - 1
        #     # print('inds_1: ', inds_1.size())
        #     bsize, tlen, slen = what_att_logits.size()
        #     # print('what_att_logits: ', what_att_logits.size())
        #     inds_1 = inds_1.expand(bsize, 1, slen).long()
        #     local_eos_probs = torch.gather(what_att_logits, 1, inds_1).squeeze(1)
        #     # print('local_eos_probs: ', local_eos_probs.size())
        #     # print('encoder_msks: ', encoder_msks.size())
        #     inds_2 = torch.sum(encoder_msks, 1, keepdim=True) - 1
        #     # print('inds_2: ', inds_2.size())
        #     eos_probs  = torch.gather(local_eos_probs, 1, inds_2.long())
        #     norm_probs = torch.gather(raw_obj_att_loss, 1, inds_2.long())
        #     # print('norm_probs:', norm_probs.size())
        #     # print('eos_probs: ', eos_probs.size())
        #     eos_loss = -torch.log(eos_probs.clamp(min=self.cfg.eps))
        #     eos_loss = torch.mean(eos_loss)
        #     diff = torch.sum(norm_probs) - 1.0
        #     norm_loss = diff * diff
        #     # print('obj_att_loss: ', att_loss)
        #     # print('eos_loss: ', eos_loss)
        #     # print('norm_loss: ', norm_loss)
        # eos_loss = self.cfg.eos_loss_weight * eos_loss
        #

        ####################################################################
        # Accuracies
        ####################################################################
        pred_accu = net.collect_accuracies(inf_outs, ref_inds)
        pred_accu = pred_accu * ref_msks
        comp_accu = torch.sum(torch.sum(pred_accu, 0), 0)
        comp_msks = torch.sum(torch.sum(ref_msks, 0), 0)
        pred_accu = comp_accu/(comp_msks + self.cfg.eps)

        return pred_loss, regression_loss, attn_loss, pred_accu

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
                {'params': net.text_encoder.embedding.parameters(), 'lr': self.cfg.finetune_lr},
                {'params': image_encoder_trainable_paras, 'lr': self.cfg.finetune_lr},
                {'params': net.text_encoder.rnn.parameters()},
                {'params': net.what_decoder.parameters()},
                {'params': net.where_decoder.parameters()}
            ], lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, factor=0.8, patience=3)
        scheduler = optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=3, gamma=0.8)
        optimizer.set_scheduler(scheduler)

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
            train_pred_loss, train_regression_loss, train_attn_loss, train_accu = \
                self.train_epoch(train_db, optimizer, epoch)

            ##################################################################
            ## Validation
            ##################################################################
            torch.cuda.empty_cache()
            val_loss, val_accu = self.validate(val_db)

            ##################################################################
            ## Sample
            ##################################################################
            torch.cuda.empty_cache()
            self.sample(epoch, test_db, self.cfg.n_samples)
            torch.cuda.empty_cache()
            ##################################################################
            ## Logging
            ##################################################################

            # update optim scheduler
            current_val_loss = np.mean(val_loss)
            optimizer.update(current_val_loss, epoch)

            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)

            logz.log_tabular("AveragePredLoss", np.mean(train_pred_loss))
            logz.log_tabular("AverageRegressLoss", np.mean(train_regression_loss))
            logz.log_tabular("AverageAccu", np.mean(train_accu))
            logz.log_tabular("ValAverageError", np.mean(val_loss))
            logz.log_tabular("ValAverageAccu", np.mean(val_accu))
            logz.log_tabular("ValAverageObjAccu",  np.mean(val_accu[:, 0]))
            logz.log_tabular("ValAverageCoordAccu", np.mean(val_accu[:, 1]))
            logz.log_tabular("ValAverageScaleAccu", np.mean(val_accu[:, 2]))
            logz.log_tabular("ValAverageRatioAccu", np.mean(val_accu[:, 3]))

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
        train_db.cfg.sent_group = -1
        train_loader = DataLoader(train_db,
            batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers)

        train_pred_loss, train_regression_loss, train_attn_loss, train_accu = [], [], [], []
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for cnt, batched in enumerate(train_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            input_inds, input_lens, bg_imgs, fg_onehots, \
            gt_inds, gt_vecs, gt_msks, gt_scene_inds = \
                self.batch_data(batched)
            # gt_scenes = [deepcopy(train_db.scenedb[x]) for x in gt_scene_inds]

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.train()
            self.net.zero_grad()

            if self.cfg.teacher_forcing:
                inf_outs, _ = self.net((input_inds, input_lens, bg_imgs, fg_onehots))
            else:
                inf_outs, _ = net.inference(input_inds, input_lens, -1, -0.1, 0, gt_inds, gt_vecs)


            pred_loss, regression_loss, attn_loss, pred_accu = self.evaluate(inf_outs, gt_inds, gt_vecs, gt_msks)
            loss = pred_loss + regression_loss + attn_loss
            loss.backward()
            optimizer.step()

            ##################################################################
            ## Collect info
            ##################################################################
            train_pred_loss.append(pred_loss.cpu().data.item())
            train_regression_loss.append(regression_loss.cpu().data.item())
            if attn_loss == 0:
                attn_loss_np = 0
            else:
                attn_loss_np = attn_loss.cpu().data.item()
            train_attn_loss.append(attn_loss_np)
            train_accu.append(pred_accu.cpu().data.numpy())


            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Epoch %03d, iter %07d:'%(epoch, cnt))
                print('loss: ', np.mean(train_pred_loss), np.mean(train_regression_loss), np.mean(train_attn_loss))
                print('accu: ', np.mean(np.array(train_accu), 0))
                print('-------------------------')

        return train_pred_loss, train_regression_loss, train_attn_loss, train_accu

    def validate(self, val_db):
        val_loss, val_accu = [], []
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        # initial experiment, just use one group of sentence
        for G in range(1):
            val_db.cfg.sent_group = G
            val_loader = DataLoader(val_db,
                batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers)

            for cnt, batched in enumerate(val_loader):
                ##################################################################
                ## Batched data
                ##################################################################
                input_inds, input_lens, bg_imgs, fg_onehots, \
                gt_inds, gt_vecs, gt_msks, gt_scene_inds = \
                    self.batch_data(batched)
                # gt_scenes = [deepcopy(val_db.scenedb[x]) for x in gt_scene_inds]

                ##################################################################
                ## Validate one step
                ##################################################################
                self.net.eval()
                with torch.no_grad():
                    inf_outs, _ = net.teacher_forcing(input_inds, input_lens, bg_imgs, fg_onehots)
                    pred_loss, regression_loss, attn_loss, pred_accu = self.evaluate(inf_outs, gt_inds, gt_vecs, gt_msks)

                loss = pred_loss + regression_loss

                val_loss.append(loss.cpu().data.item())
                val_accu.append(pred_accu.cpu().data.numpy())
                print(G, cnt)

        val_loss = np.array(val_loss)
        val_accu = np.stack(val_accu, 0)

        return val_loss, val_accu

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

            gt_img = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
            gt_img, _, _ = create_squared_image(gt_img)
            gt_img = cv2.resize(gt_img, (self.cfg.input_image_size[0], self.cfg.input_image_size[1]))

            ##############################################################
            # Inputs
            ##############################################################
            input_inds_np = np.array(entry['word_inds'])
            input_lens_np = np.array(entry['word_lens'])

            input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
            input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
            if self.cfg.cuda:
                input_inds = input_inds.cuda()
                input_lens = input_lens.cuda()

            ##############################################################
            # Inference
            ##############################################################
            self.net.eval()
            with torch.no_grad():
                inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None, None)
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

                plt.subplot(4, 5, j+1)
                plt.title(subtitle, fontsize=30)
                vis_img = frames[j][ :, :, ::-1]
                vis_img = clamp_array(vis_img, 0, 255).astype(np.uint8)
                plt.imshow(vis_img)
                plt.axis('off')
            plt.subplot(4, 5, 20)
            plt.imshow(gt_img[:, :, ::-1])
            plt.axis('off')

            name = osp.splitext(osp.basename(entry['color_path']))[0]
            out_path = osp.join(output_dir, name+'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print('sampling: %d, %d'%(epoch, i))

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

    def save_checkpoint(self, epoch, log):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'supervised_ckpts')
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
        checkpoint_dir = osp.join(self.cfg.model_dir, 'supervised_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_name = "ckpt-best.pkl"
        torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))
