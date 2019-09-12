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

from modules.puzzle_model import PuzzleModel
from composites_config import get_config
from composites_utils import *
from optim import Optimizer

from datasets.composites_loader import sequence_loader, patch_vol_loader
from nntable import AllCategoriesTables


class PuzzleTrainer(object):
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg
        self.net = PuzzleModel(db)
        if self.cfg.cuda:
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

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
        self.optimizer = optimizer
        self.epoch = 0

        if self.cfg.pretrained is not None:
            self.load_pretrained_net(self.cfg.pretrained)
        
    def load_pretrained_net(self, pretrained_name):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, 'composites_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        # states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(states['state_dict'])
        self.optimizer.optimizer.load_state_dict(states['optimizer'])
        self.epoch = states['epoch']

    def batch_data(self, entry):
        ################################################
        # Inputs
        ################################################
        input_inds = entry['word_inds'].long()
        input_lens = entry['word_lens'].long()
        fg_inds    = entry['fg_inds'].long()
        bg_imgs    = entry['background'].float()
        fg_imgs    = entry['foreground'].float()
        neg_imgs   = entry['negative'].float()
        fg_resnets  = entry['foreground_resnets'].float()
        neg_resnets = entry['negative_resnets'].float()
        fg_onehots = indices2onehots(fg_inds, self.cfg.output_vocab_size)

        ################################################
        # Outputs
        ################################################
        gt_inds = entry['out_inds'].long()
        # gt_vecs = entry['out_vecs'].float()
        gt_msks = entry['out_msks'].float()
        patch_inds = entry['patch_inds'].long().numpy()

        if self.cfg.cuda:
            input_inds = input_inds.cuda(non_blocking=True)
            input_lens = input_lens.cuda(non_blocking=True)
            fg_onehots = fg_onehots.cuda(non_blocking=True)
            bg_imgs    = bg_imgs.cuda(non_blocking=True)
            fg_imgs    = fg_imgs.cuda(non_blocking=True)
            neg_imgs   = neg_imgs.cuda(non_blocking=True)
            fg_resnets = fg_resnets.cuda(non_blocking=True)
            neg_resnets = neg_resnets.cuda(non_blocking=True)
            gt_inds = gt_inds.cuda(non_blocking=True)
            gt_msks = gt_msks.cuda(non_blocking=True)

        return input_inds, input_lens, fg_onehots, bg_imgs, fg_imgs, neg_imgs, fg_resnets, neg_resnets, gt_inds, gt_msks, patch_inds

    def dump_shape_vectors(self, train_db):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        db = patch_vol_loader(train_db)
        loader = DataLoader(db,
            batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)

        for cnt, batched in enumerate(loader):
            start = time()
            patch_inds = batched['patch_ind'].long()
            patch_vols = batched['patch_vol'].float()
            patch_resnet = batched['patch_resnet'].float()
            if self.cfg.cuda:
                patch_vols = patch_vols.cuda(non_blocking=True)
                patch_resnet = patch_resnet.cuda(non_blocking=True)
            patch_features = net.shape_encoder.batch_forward(patch_vols, patch_resnet)
            patch_features = patch_features.cpu().data.numpy()
            for i in range(patch_vols.size(0)):
                patch = train_db.patchdb[patch_inds[i]]
                image_index = patch['image_index']
                instance_ind = patch['instance_ind']
                patch_feature_path = train_db.patch_path_from_indices(image_index, instance_ind, 'patch_feature', 'pkl', self.cfg.use_patch_background)
                patch_feature_dir = osp.dirname(patch_feature_path)
                maybe_create(patch_feature_dir)
                features = patch_features[i].flatten()
                with open(patch_feature_path, 'wb') as fid:
                    pickle.dump(features, fid, pickle.HIGHEST_PROTOCOL)
            print('%s, current_ind: %d, time consumed: %f'%(train_db.split, cnt, time() - start))
            
    def evaluate(self, inf_outs, pos_vecs, neg_vecs, ref_inds, ref_msks, db=None, patch_inds=None):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        _, _, _, _, enc_msks, what_wei, where_wei = inf_outs
        logits, pred_vecs = net.collect_logits_and_vectors(inf_outs, ref_inds)

        ####################################################################
        # Prediction loss
        ####################################################################
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
        # Embedding loss
        ####################################################################
        bsize, slen, vsize = pred_vecs.size()
        embed_metric = nn.TripletMarginLoss(margin=self.cfg.margin, p=2, eps=1e-06, swap=False,
            size_average=None, reduction='none')
        embed_loss = embed_metric(
            pred_vecs.view(bsize*slen, vsize),
            pos_vecs.view(bsize*slen, vsize),
            neg_vecs.view(bsize*slen, vsize)).view(bsize, slen)
        embed_mask = ref_msks[:,:,1]
        embed_loss = torch.sum(embed_loss * embed_mask)/(torch.sum(embed_mask) + self.cfg.eps)
        embed_loss = embed_loss * self.cfg.embed_loss_weight
        # print('embed_loss', embed_loss)

        ####################################################################
        # doubly stochastic attn loss
        ####################################################################
        attn_loss = logits.new_zeros(size=(1,))
        encoder_msks = enc_msks
        if self.cfg.what_attn:
            obj_msks = ref_msks[:,:,0].unsqueeze(-1)
            what_att_logits  = what_wei
            raw_obj_att_loss = torch.mul(what_att_logits, obj_msks)
            raw_obj_att_loss = torch.sum(raw_obj_att_loss, dim=1)
            obj_att_loss = raw_obj_att_loss - encoder_msks
            obj_att_loss = torch.sum(obj_att_loss ** 2, dim=-1)
            obj_att_loss = torch.mean(obj_att_loss)
            attn_loss = attn_loss + obj_att_loss
        attn_loss = self.cfg.attn_loss_weight * attn_loss

        ####################################################################
        # Accuracies
        ####################################################################
        pred_accu = net.collect_accuracies(inf_outs, ref_inds)
        pred_accu = pred_accu * ref_msks
        comp_accu = torch.sum(torch.sum(pred_accu, 0), 0)
        comp_msks = torch.sum(torch.sum(ref_msks, 0), 0)
        pred_accu = comp_accu/(comp_msks + self.cfg.eps)

        ####################################################################
        # Dump predicted vectors
        ####################################################################
        if (db is not None) and (patch_inds is not None):
            tmp_vecs = pred_vecs.clone()
            tmp_vecs = tmp_vecs.detach().cpu().data.numpy()
            bsize, slen, fsize = tmp_vecs.shape
            for i in range(bsize):
                for j in range(slen):
                    patch_index = patch_inds[i, j]
                    if patch_index < 0:
                        continue
                    patch = db.patchdb[patch_index]
                    image_index = patch['image_index']
                    instance_ind = patch['instance_ind']
                    patch_feature_path = db.patch_path_from_indices(image_index, instance_ind, 'predicted_feature', 'pkl', None)
                    patch_feature_dir = osp.dirname(patch_feature_path)
                    maybe_create(patch_feature_dir)
                    features = tmp_vecs[i, j].flatten()
                    with open(patch_feature_path, 'wb') as fid:
                        pickle.dump(features, fid, pickle.HIGHEST_PROTOCOL)


        return pred_loss, embed_loss, attn_loss, pred_accu

    def train(self, train_db, val_db, test_db):
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
        for epoch in range(self.epoch, self.cfg.n_epochs):
            ##################################################################
            ## Training
            ##################################################################
            torch.cuda.empty_cache()
            train_loss, train_accu = self.train_epoch(train_db, epoch)

            ##################################################################
            ## Validation
            ##################################################################
            torch.cuda.empty_cache()
            val_loss, val_accu = self.validate_epoch(val_db, epoch)

            ##################################################################
            ## Logging
            ##################################################################

            # update optim scheduler
            current_val_loss = np.mean(val_loss[:,0])
            # self.optimizer.update(current_val_loss, epoch)
            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)
            logz.log_tabular("AverageLoss",         np.mean(train_loss[:, 0]))
            logz.log_tabular("AveragePredLoss",     np.mean(train_loss[:, 1]))
            logz.log_tabular("AverageEmbedLoss",    np.mean(train_loss[:, 2]))
            logz.log_tabular("AverageAttnLoss",     np.mean(train_loss[:, 3]))
            logz.log_tabular("AverageObjAccu",      np.mean(train_accu[:, 0]))
            logz.log_tabular("AverageCoordAccu",    np.mean(train_accu[:, 1]))
            logz.log_tabular("AverageScaleAccu",    np.mean(train_accu[:, 2]))
            logz.log_tabular("AverageRatioAccu",    np.mean(train_accu[:, 3]))

            logz.log_tabular("ValAverageLoss",      np.mean(val_loss[:, 0]))
            logz.log_tabular("ValAveragePredLoss",  np.mean(val_loss[:, 1]))
            logz.log_tabular("ValAverageEmbedLoss", np.mean(val_loss[:, 2]))
            logz.log_tabular("ValAverageAttnLoss",  np.mean(val_loss[:, 3]))
            logz.log_tabular("ValAverageObjAccu",   np.mean(val_accu[:, 0]))
            logz.log_tabular("ValAverageCoordAccu", np.mean(val_accu[:, 1]))
            logz.log_tabular("ValAverageScaleAccu", np.mean(val_accu[:, 2]))
            logz.log_tabular("ValAverageRatioAccu", np.mean(val_accu[:, 3]))
            logz.dump_tabular()

            ##################################################################
            ## Checkpoint
            ##################################################################
            self.save_checkpoint(epoch)
                           
    def train_epoch(self, train_db, epoch):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        train_db.cfg.sent_group = -1
        seq_db = sequence_loader(train_db)

        train_loader = DataLoader(seq_db,
            batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True)

        all_losses, all_accuracies = [], []

        for cnt, batched in enumerate(train_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            input_inds, input_lens, fg_onehots, bg_imgs, \
            fg_imgs, neg_imgs, fg_resnets, neg_resnets,\
            gt_inds, gt_msks, patch_inds = \
                self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.train()
            self.net.zero_grad()

            inputs = (input_inds, input_lens, bg_imgs, fg_onehots, fg_imgs, neg_imgs, fg_resnets, neg_resnets)
            inf_outs, _, pos_vecs, neg_vecs = self.net(inputs)
            pred_loss, embed_loss, attn_loss, pred_accu = self.evaluate(inf_outs, pos_vecs, neg_vecs, gt_inds, gt_msks)

            loss = pred_loss + embed_loss + attn_loss
            loss.backward()
            self.optimizer.step()

            ##################################################################
            ## Collect info
            ##################################################################
            all_losses.append(np.array(
                [
                    loss.cpu().data.item(), 
                    pred_loss.cpu().data.item(), 
                    embed_loss.cpu().data.item(), 
                    attn_loss.cpu().data.item()
                ]))
            all_accuracies.append(pred_accu.cpu().data.numpy())

            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Epoch %03d, iter %07d:'%(epoch, cnt))
                tmp_losses = np.stack(all_losses, 0)
                tmp_accuracies = np.stack(all_accuracies, 0)
                print('losses: ', np.mean(tmp_losses[:,0]), np.mean(tmp_losses[:,1]), np.mean(tmp_losses[:,2]), np.mean(tmp_losses[:,3]))
                print('accuracies: ', np.mean(tmp_accuracies[:,0]), np.mean(tmp_accuracies[:,1]), np.mean(tmp_accuracies[:,2]), np.mean(tmp_accuracies[:,3]))
                print('-------------------------')

        all_losses = np.stack(all_losses, 0)
        all_accuracies = np.stack(all_accuracies, 0)

        return all_losses, all_accuracies

    def validate_epoch(self, val_db, epoch):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        all_losses, all_accuracies = [], []
        for G in range(5):
            val_db.cfg.sent_group = G
            seq_db = sequence_loader(val_db)
            val_loader = DataLoader(seq_db,
                batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, pin_memory=True)

            for cnt, batched in enumerate(val_loader):
                ##################################################################
                ## Batched data
                ##################################################################
                input_inds, input_lens, fg_onehots, bg_imgs, \
                fg_imgs, neg_imgs, fg_resnets, neg_resnets,\
                gt_inds, gt_msks, patch_inds = \
                    self.batch_data(batched)

                ##################################################################
                ## Validate one step
                ##################################################################
                self.net.eval()
                with torch.no_grad():
                    inputs = (input_inds, input_lens, bg_imgs, fg_onehots, fg_imgs, neg_imgs, fg_resnets, neg_resnets)
                    inf_outs, _, pos_vecs, neg_vecs = self.net(inputs)
                    pred_loss, embed_loss, attn_loss, pred_accu = self.evaluate(inf_outs, pos_vecs, neg_vecs, gt_inds, gt_msks)

                loss = pred_loss + embed_loss + attn_loss
                all_losses.append(np.array(
                    [
                        loss.cpu().data.item(), 
                        pred_loss.cpu().data.item(), 
                        embed_loss.cpu().data.item(), 
                        attn_loss.cpu().data.item()
                    ]))
                all_accuracies.append(pred_accu.cpu().data.numpy())
                print(epoch, G, cnt)

        all_losses = np.stack(all_losses, 0)
        all_accuracies = np.stack(all_accuracies, 0)

        return all_losses, all_accuracies

    def save_checkpoint(self, epoch):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'composites_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()        
        }
        torch.save(states, osp.join(checkpoint_dir, "ckpt-%03d.pkl"%epoch))

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
        output_dir = osp.join(self.cfg.model_dir, 'composites_samples')
        maybe_create(output_dir)
        plt.switch_backend('agg')
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        num_sents = len(input_sentences)
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
            self.net.eval()
            with torch.no_grad():
                inf_outs, env = net.inference(input_inds, input_lens, -1, 1.0, 0, None, None, nn_table)
            frames, _, _, _, _ = env.batch_redraw(return_sequence=True)
            frames = frames[0]
            # _, _, _, _, _, what_wei, where_wei = inf_outs
            # if self.cfg.what_attn:
            #     what_attn_words = self.decode_attention(
            #         input_inds_np, input_lens_np, what_wei.squeeze(0))
            # if self.cfg.where_attn > 0:
            #     where_attn_words = self.decode_attention(
            #         input_inds_np, input_lens_np, where_wei.squeeze(0))

            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(32, 32))
            plt.suptitle(sentence, fontsize=40)
            for j in range(len(frames)):
                # subtitle = ''
                # if self.cfg.what_attn:
                #     subtitle = subtitle + ' '.join(what_attn_words[j])
                # if self.cfg.where_attn > 0:
                #     subtitle = subtitle + '\n' + ' '.join(where_attn_words[j])
                plt.subplot(4, 4, j+1)
                # plt.title(subtitle, fontsize=30)
                if self.cfg.use_color_volume:
                    vis_img, _ = heuristic_collage(frames[j], 83)
                else:
                    vis_img = frames[j][:,:,-3:]
                vis_img = clamp_array(vis_img[ :, :, ::-1], 0, 255).astype(np.uint8)
                plt.imshow(vis_img)
                plt.axis('off')
            out_path = osp.join(output_dir, '%09d.png'%i)
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        
    def sample_for_vis(self, epoch, test_db, N, random_or_not=False, nn_table=None):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'vis')
        maybe_create(output_dir)

        seq_db = sequence_loader(test_db)

        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if random_or_not:
            indices = np.random.permutation(range(len(test_db.scenedb)))
        else:
            indices = range(len(test_db.scenedb))
        indices = indices[:min(N, len(test_db.scenedb))]

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for i in indices:
            entry    = seq_db[i]
            gt_scene = test_db.scenedb[i]
            image_index = gt_scene['image_index']
            image_path = test_db.color_path_from_index(image_index)

            gt_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
                inf_outs, env = net.inference(input_inds, input_lens, -1, 1.0, 0, None, None, nn_table)
            frames, _, _, _, _ = env.batch_redraw(return_sequence=True)
            frames = frames[0]
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
                if self.cfg.use_color_volume:
                    vis_img, _ = heuristic_collage(frames[j], 83)
                else:
                    vis_img = frames[j][:,:,-3:]
                vis_img = clamp_array(vis_img[ :, :, ::-1], 0, 255).astype(np.uint8)
                plt.imshow(vis_img)
                plt.axis('off')
            plt.subplot(4, 4, 16)
            plt.imshow(gt_img[:, :, ::-1])
            plt.axis('off')


            name = osp.splitext(osp.basename(image_path))[0]
            out_path = osp.join(output_dir, name+'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print('sampling: %d, %d'%(epoch, i))

    def sample_for_eval(self, test_db, nn_table=None):
        ##############################################################
        # Output prefix
        ##############################################################
        # gt_dir    = osp.join(self.cfg.model_dir, 'gt')
        # frame_dir = osp.join(self.cfg.model_dir, 'proposal_images')
        # noice_dir = osp.join(self.cfg.model_dir, 'proposal_noices')
        # label_dir = osp.join(self.cfg.model_dir, 'proposal_labels')
        # mask_dir  = osp.join(self.cfg.model_dir, 'proposal_masks')
        # info_dir  = osp.join(self.cfg.model_dir, 'proposal_info')

        main_dir = 'puzzle_results'
        maybe_create(main_dir)

        gt_dir    = osp.join(main_dir, 'gt')
        frame_dir = osp.join(main_dir, 'proposal_images')
        noice_dir = osp.join(main_dir, 'proposal_noices')
        label_dir = osp.join(main_dir, 'proposal_labels')
        mask_dir  = osp.join(main_dir, 'proposal_masks')
        info_dir  = osp.join(main_dir, 'proposal_info')

        maybe_create(gt_dir)
        maybe_create(frame_dir)
        maybe_create(noice_dir)
        maybe_create(label_dir)
        maybe_create(mask_dir)
        maybe_create(info_dir)

        seq_db = sequence_loader(test_db)
        ##############################################################
        # Main loop
        ##############################################################
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        # start_ind = 0
        # end_ind = len(seq_db)
        start_ind = self.cfg.seed * 1250
        end_ind = (self.cfg.seed+1) * 1250
        # start_ind = 35490
        # end_ind = len(seq_db)
        for i in range(start_ind, end_ind):
            entry    = seq_db[i]
            gt_scene = test_db.scenedb[i]
            image_index = gt_scene['image_index']
            image_path = test_db.color_path_from_index(image_index)
            name = osp.splitext(osp.basename(image_path))[0]
            gt_path = osp.join(gt_dir, osp.basename(image_path))
            # save gt 
            shutil.copy2(image_path, gt_path)

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
                inf_outs, env = net.inference(input_inds, input_lens, -1, 1.0, 0, None, None, nn_table)
            frame, noice, mask, label, env_info = env.batch_redraw(return_sequence=False)
            frame = frame[0][0]; noice = noice[0][0]; mask = mask[0][0]; label = label[0][0]
            env_info = env_info[0]
            frame_path = osp.join(frame_dir, name+'.jpg')
            noice_path = osp.join(noice_dir, name+'.jpg')
            mask_path  = osp.join(mask_dir,  name+'.png')
            label_path = osp.join(label_dir, name+'.png')
            info_path  = osp.join(info_dir, name+'.json')

            if self.cfg.use_color_volume:
                frame, _ = heuristic_collage(frame, 83)
                noice, _ = heuristic_collage(noice, 83)
            else:
                frame  = frame[:,:,-3:]
                noice  = noice[:,:,-3:]
            cv2.imwrite(frame_path, clamp_array(frame, 0, 255).astype(np.uint8))
            cv2.imwrite(noice_path, clamp_array(noice, 0, 255).astype(np.uint8))
            cv2.imwrite(mask_path, clamp_array(255*mask, 0, 255))
            cv2.imwrite(label_path, label)

            # info
            pred_info = {}
            pred_info['width'] = env_info['width']
            pred_info['height'] = env_info['height']
            pred_info['clses'] = env_info['clses'].tolist()
            pred_info['boxes'] = [x.tolist() for x in env_info['boxes']]
            current_patches = env_info['patches']
            current_image_indices = []
            current_instance_inds = []
            for j in range(len(pred_info['clses'])):
                current_image_indices.append(current_patches[j]['image_index'])
                current_instance_inds.append(current_patches[j]['instance_ind'])
            pred_info['image_indices'] = current_image_indices
            pred_info['instance_inds'] = current_instance_inds
            with open(info_path, 'w') as fp:
                json.dump(pred_info, fp, indent=4, sort_keys=True)
            print('sampling: %d, %s'%(i, name))


            