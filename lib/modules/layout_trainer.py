#!/usr/bin/env python

import os, sys, cv2, math
import random, json, logz
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from layout_utils import *
from layout_config import get_config
from modules.layout_evaluator import *
from modules.layout_model import DrawModel
from optim import Optimizer


class SupervisedTrainer(object):
    def __init__(self, db):
        self.cfg = db.cfg 
        self.db = db
        net = DrawModel(db)
        if self.cfg.cuda:
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            net = net.cuda()
        self.net = net
        if self.cfg.pretrained is not None:
            self.load_pretrained_net(self.cfg.pretrained)
        
    def load_pretrained_net(self, pretrained_name):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, 'layout_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path) 
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage) 
        net.load_state_dict(states)

    def batch_data(self, entry):
        ################################################
        # Inputs
        ################################################
        input_inds = entry['word_inds'].long()
        input_lens = entry['word_lens'].long()
        fg_inds = entry['fg_inds'].long()
        bg_imgs = entry['background'].float()
        fg_onehots = indices2onehots(fg_inds, self.cfg.output_cls_size)

        ################################################
        # Outputs
        ################################################
        gt_inds = entry['out_inds'].long()
        gt_msks = entry['out_msks'].float()
        gt_scene_inds = entry['scene_idx'].long().numpy()

        if self.cfg.cuda:
            input_inds = input_inds.cuda()
            input_lens = input_lens.cuda()
            fg_onehots = fg_onehots.cuda()
            bg_imgs = bg_imgs.cuda()
            gt_inds = gt_inds.cuda()
            gt_msks = gt_msks.cuda()
        
        return input_inds, input_lens, bg_imgs, fg_onehots, gt_inds, gt_msks, gt_scene_inds
    
    def evaluate(self, inf_outs, ref_inds, ref_msks):
        ####################################################################
        # Prediction loss
        ####################################################################
        _, _, _, enc_msks, what_wei, where_wei = inf_outs

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        logits = net.collect_logits(inf_outs, ref_inds)
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
        
        eos_loss = 0
        if self.cfg.what_attn and self.cfg.eos_loss_weight > 0:
            # print('-------------------')
            # print('obj_msks: ', obj_msks.size())
            inds_1 = torch.sum(obj_msks, 1, keepdim=True) - 1
            # print('inds_1: ', inds_1.size())
            bsize, tlen, slen = what_att_logits.size()
            # print('what_att_logits: ', what_att_logits.size())
            inds_1 = inds_1.expand(bsize, 1, slen).long()
            local_eos_probs = torch.gather(what_att_logits, 1, inds_1).squeeze(1)
            # print('local_eos_probs: ', local_eos_probs.size())
            # print('encoder_msks: ', encoder_msks.size())
            inds_2 = torch.sum(encoder_msks, 1, keepdim=True) - 1
            # print('inds_2: ', inds_2.size())
            eos_probs  = torch.gather(local_eos_probs, 1, inds_2.long())
            norm_probs = torch.gather(raw_obj_att_loss, 1, inds_2.long())
            # print('norm_probs:', norm_probs.size())
            # print('eos_probs: ', eos_probs.size())
            eos_loss = -torch.log(eos_probs.clamp(min=self.cfg.eps))
            eos_loss = torch.mean(eos_loss)
            diff = torch.sum(norm_probs) - 1.0
            norm_loss = diff * diff
            # print('obj_att_loss: ', att_loss)
            # print('eos_loss: ', eos_loss)
            # print('norm_loss: ', norm_loss)
        eos_loss = self.cfg.eos_loss_weight * eos_loss
        

        ####################################################################
        # Accuracies
        ####################################################################
        pred_accu = net.collect_accuracies(inf_outs, ref_inds)
        pred_accu = pred_accu * ref_msks
        comp_accu = torch.sum(torch.sum(pred_accu, 0), 0)
        comp_msks = torch.sum(torch.sum(ref_msks, 0), 0)
        pred_accu = comp_accu/(comp_msks + self.cfg.eps)

        return pred_loss, attn_loss, eos_loss, pred_accu
        
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
                {'params': image_encoder_trainable_paras},
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
        for epoch in range(self.cfg.n_epochs):
            ##################################################################
            ## Training
            ##################################################################
            torch.cuda.empty_cache()
            train_pred_loss, train_attn_loss, train_eos_loss, train_accu = \
                self.train_epoch(train_db, optimizer, epoch)
        
            ##################################################################
            ## Validation
            ##################################################################
            torch.cuda.empty_cache()
            val_loss, val_accu, val_infos = self.validate_epoch(val_db)
            
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
            optimizer.update(np.mean(val_loss), epoch)
                
            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)

            logz.log_tabular("TrainAverageError", np.mean(train_pred_loss))
            logz.log_tabular("TrainAverageAccu", np.mean(train_accu))
            logz.log_tabular("ValAverageError", np.mean(val_loss))
            logz.log_tabular("ValAverageAccu", np.mean(val_accu))
            logz.log_tabular("ValAverageObjAccu", np.mean(val_accu[:, 0]))
            logz.log_tabular("ValAverageCoordAccu", np.mean(val_accu[:, 1]))
            logz.log_tabular("ValAverageScaleAccu", np.mean(val_accu[:, 2]))
            logz.log_tabular("ValAverageRatioAccu", np.mean(val_accu[:, 3]))
            logz.log_tabular("ValUnigramF3", np.mean(val_infos.unigram_F3()))
            logz.log_tabular("ValBigramF3",  np.mean(val_infos.bigram_F3()))
            logz.log_tabular("ValUnigramP",  np.mean(val_infos.unigram_P()))
            logz.log_tabular("ValUnigramR",  np.mean(val_infos.unigram_R()))
            logz.log_tabular("ValBigramP",   val_infos.mean_bigram_P())
            logz.log_tabular("ValBigramR",   val_infos.mean_bigram_R())
            logz.log_tabular("ValUnigramScale", np.mean(val_infos.scale()))
            logz.log_tabular("ValUnigramRatio", np.mean(val_infos.ratio()))
            logz.log_tabular("ValUnigramSim",   np.mean(val_infos.unigram_coord()))
            logz.log_tabular("ValBigramSim",    val_infos.mean_bigram_coord())

            logz.dump_tabular()

            ##################################################################
            ## Checkpoint
            ##################################################################
            log_info = [np.mean(val_loss), np.mean(val_accu)]
            self.save_checkpoint(epoch, log_info)
            torch.cuda.empty_cache()

    def train_epoch(self, train_db, optimizer, epoch):
        train_db.cfg.sent_group = -1
        train_loader = DataLoader(train_db, 
            batch_size=self.cfg.batch_size, shuffle=True, 
            num_workers=self.cfg.num_workers)

        train_pred_loss, train_attn_loss, train_eos_loss, train_accu = [], [], [], []

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for cnt, batched in enumerate(train_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            input_inds, input_lens, bg_imgs, fg_onehots, \
            gt_inds, gt_msks, gt_scene_inds = \
                self.batch_data(batched)
            gt_scenes = [deepcopy(train_db.scenedb[x]) for x in gt_scene_inds]
                
            ##################################################################
            ## Train one step
            ##################################################################
            self.net.train()
            self.net.zero_grad()

            if self.cfg.teacher_forcing:
                inf_outs, _ = self.net((input_inds, input_lens, bg_imgs, fg_onehots))
            else:
                inf_outs, _ = net.inference(input_inds, input_lens, -1, -0.1, 0, gt_inds)

            # print('image_encoder: ', self.net.image_encoder.training)
            # print('text_encoder: ',  self.net.text_encoder.training)
            # print('what_decoder: ',  self.net.what_decoder.training)
            # print('where_decoder: ', self.net.where_decoder.training)

            pred_loss, attn_loss, eos_loss, pred_accu = self.evaluate(inf_outs, gt_inds, gt_msks)
            loss = pred_loss + attn_loss + eos_loss
            loss.backward()
            optimizer.step()

            ##################################################################
            ## Collect info
            ##################################################################
            train_pred_loss.append(pred_loss.cpu().data.item())
            if attn_loss == 0:
                attn_loss_np = 0
            else:
                attn_loss_np = attn_loss.cpu().data.item()
            train_attn_loss.append(attn_loss_np)
            if eos_loss == 0:
                eos_loss_np = 0
            else:
                eos_loss_np = eos_loss.cpu().data.item()
            train_eos_loss.append(eos_loss_np)
            train_accu.append(pred_accu.cpu().data.numpy())


            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Epoch %03d, iter %07d:'%(epoch, cnt))
                print('loss: ', np.mean(train_pred_loss), np.mean(train_attn_loss), np.mean(train_eos_loss))
                print('accu: ', np.mean(np.array(train_accu), 0))
                print('-------------------------')

        return train_pred_loss, train_attn_loss, train_eos_loss, train_accu

    def validate_epoch(self, val_db):
        val_loss, val_accu, top1_scores = [], [], []
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for G in range(5):
            val_db.cfg.sent_group = G
            val_loader = DataLoader(val_db, 
                batch_size=16, shuffle=False, 
                num_workers=self.cfg.num_workers)
            
            for cnt, batched in enumerate(val_loader):
                ##################################################################
                ## Batched data
                ##################################################################
                input_inds, input_lens, bg_imgs, fg_onehots, \
                gt_inds, gt_msks, gt_scene_inds = \
                    self.batch_data(batched)
                gt_scenes = [deepcopy(val_db.scenedb[x]) for x in gt_scene_inds]

                ##################################################################
                ## Validate one step
                ##################################################################
                self.net.eval()
                with torch.no_grad():
                    _, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)
                    # infos = env.batch_evaluation(gt_inds.cpu().data.numpy())
                    scores = env.batch_evaluation(gt_scenes)
                    # scores = np.stack(scores, 0)
                    # infos = eval_info(self.cfg, scores)
                    inf_outs, _ = net.teacher_forcing(input_inds, input_lens, bg_imgs, fg_onehots)
                    # print('gt_inds', gt_inds)
                    pred_loss, attn_loss, eos_loss, pred_accu = self.evaluate(inf_outs, gt_inds, gt_msks)
                
                top1_scores.extend(scores)
                val_loss.append(pred_loss.cpu().data.item())
                val_accu.append(pred_accu.cpu().data.numpy())  

                print(G, cnt)
                # print('pred_loss', pred_loss.data.item())
                # print('pred_accu', pred_accu)
                # print('scores', scores)
                # if cnt > 0:
                #     break
        
        top1_scores = np.stack(top1_scores, 0)
        val_loss = np.array(val_loss)
        val_accu = np.stack(val_accu, 0)
        infos = eval_info(self.cfg, top1_scores)

        return val_loss, val_accu, infos
                        
    def sample(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'vis')
        pred_dir   = osp.join(self.cfg.model_dir, '%03d'%epoch, 'pred')
        gt_dir     = osp.join(self.cfg.model_dir, '%03d'%epoch, 'gt')
        img_dir  = osp.join(self.cfg.model_dir, '%03d'%epoch, 'color')

        maybe_create(output_dir); maybe_create(pred_dir)
        maybe_create(gt_dir); maybe_create(img_dir)

        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if random_or_not:
            indices = np.random.permutation(range(len(test_db)))
        else:
            indices = range(len(test_db))
        indices = indices[:N]

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for i in indices:
            entry    = test_db[i]
            gt_scene = test_db.scenedb[i]

            gt_img = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
            gt_img, _, _ = create_squared_image(gt_img)
            gt_img = cv2.resize(gt_img, (self.cfg.draw_size[0], self.cfg.draw_size[1]))

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
                inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)
            frames = env.batch_redraw(return_sequence=True)[0]
            _, _, _, _, what_wei, where_wei = inf_outs
            
            if self.cfg.what_attn:
                what_attn_words = self.decode_attention(
                    input_inds_np, input_lens_np, what_wei.squeeze(0))
            if self.cfg.where_attn > 0:
                where_attn_words = self.decode_attention(
                    input_inds_np, input_lens_np, where_wei.squeeze(0))
        
            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(60, 30))
            plt.suptitle(entry['sentence'], fontsize=50)
            for j in range(frames.shape[0]):
                # print(attn_words[j])
                subtitle = ''
                if self.cfg.what_attn:
                    subtitle = subtitle + ' '.join(what_attn_words[j])
                if self.cfg.where_attn > 0:
                    subtitle = subtitle + '\n' + ' '.join(where_attn_words[j])

                plt.subplot(4, 4, j+1)
                plt.title(subtitle, fontsize=30)
                plt.imshow(frames[j, :, :, ::-1])
                plt.axis('off')
            plt.subplot(4, 4, 16)
            plt.imshow(gt_img[:, :, ::-1])
            plt.axis('off')

            name = osp.splitext(osp.basename(entry['color_path']))[0]
            out_path = osp.join(output_dir, name+'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            
            cv2.imwrite(osp.join(pred_dir, name+'.png'), frames[-1])
            cv2.imwrite(osp.join(img_dir,  name+'.png'), gt_img)
            gt_layout = self.db.render_scene_as_output(gt_scene, False, gt_img)
            cv2.imwrite(osp.join(gt_dir, name+'.png'), gt_layout)
            print('sampling: %d, %d'%(epoch, i))

    def show_metric(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'metric')
        maybe_create(output_dir)
        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if random_or_not:
            indices = np.random.permutation(range(len(test_db)))
        else:
            indices = range(len(test_db))
        indices = indices[:N]
        test_db.cfg.sent_group=1

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        ev = evaluator(self.db)

        for i in indices:
            entry    = test_db[i]
            gt_scene = test_db.scenedb[i]
            scene_idx = int(gt_scene['img_idx'])
            name = osp.splitext(osp.basename(entry['color_path']))[0]

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
                inf_outs, env = self.net.inference(input_inds, input_lens, -1, 2.0, 0, None)
            frame = env.batch_redraw(return_sequence=False)[0][0]
            raw_pred_scene = env.scenes[0]

            pred_inds = deepcopy(raw_pred_scene['out_inds'])
            pred_inds = np.stack(pred_inds, 0)
            pred_scene = self.db.output_inds_to_scene(pred_inds)

            graph_1 = scene_graph(self.db, pred_scene, None, False)
            graph_2 = scene_graph(self.db, gt_scene,   None, False)

            color_1 = frame.copy()
            gt_img = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
            gt_img, _, _ = create_squared_image(gt_img)
            gt_img = cv2.resize(gt_img, (self.cfg.draw_size[0], self.cfg.draw_size[1]))
            color_2 = gt_img

            cv2.imwrite('%09d_b.png'%i, color_1)
            cv2.imwrite('%09d_i.png'%i, color_2)

            color_1 = visualize_unigram(self.cfg, color_1, graph_1.unigrams, (225, 0, 0))
            color_2 = visualize_unigram(self.cfg, color_2, graph_2.unigrams, (225, 0, 0))
            color_1 = visualize_bigram(self.cfg, color_1, graph_1.bigrams, (0, 0, 255))
            color_2 = visualize_bigram(self.cfg, color_2, graph_2.bigrams, (0, 0, 255))

            scores = ev.evaluate_graph(graph_1, graph_2)

            color_1 = visualize_unigram(self.cfg, color_1, ev.common_pred_unigrams, (0, 225, 0))
            color_2 = visualize_unigram(self.cfg, color_2, ev.common_gt_unigrams,   (0, 225, 0))
            color_1 = visualize_bigram(self.cfg, color_1, ev.common_pred_bigrams, (0, 255, 255))
            color_2 = visualize_bigram(self.cfg, color_2, ev.common_gt_bigrams, (0, 255, 255))

            info = eval_info(self.cfg, scores[None, ...])

            plt.switch_backend('agg')
            fig = plt.figure(figsize=(16, 10))
            title = entry['sentence']
            title += 'UR:%f,UP:%f,BR:%f,BP:%f\n'%(info.unigram_R()[0], info.unigram_P()[0], info.bigram_R()[0], info.bigram_P()[0])
            title += 'scale: %f, ratio: %f, coord: %f, b:%f \n'%(info.scale()[0], info.ratio()[0], info.unigram_coord()[0], info.bigram_coord()[0])

            plt.suptitle(title)
            plt.subplot(1, 2, 1); plt.imshow(color_1[:,:,::-1]); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(color_2[:,:,::-1]); plt.axis('off')

            out_path = osp.join(output_dir, name +'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

    def sample_all_top1(self, test_db):
        ##############################################################
        # Output prefix
        ##############################################################
        out_dir = osp.join(self.cfg.model_dir, 'top1_scenes')
        maybe_create(out_dir)
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        indices = range(len(test_db))
        scores = []
        for G in range(5):
            test_db.cfg.sent_group = G
            G_dir = osp.join(out_dir, '%02d'%G)
            maybe_create(G_dir)
            img_dir = osp.join(G_dir, 'images')
            maybe_create(img_dir)
            scene_dir = osp.join(G_dir, 'scenes')
            maybe_create(scene_dir)

            for i in indices:
                entry    = test_db[i]
                gt_scene = test_db.scenedb[i]
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
                    inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)

                # for j in range(len(env.scenes)):
                #     bar_scene = env.scenes[j]
                #     bar_inds = bar_scene['out_inds']
                #     if bar_inds[0][0] > self.cfg.EOS_idx:
                #         break
                # if bar_inds[0][0] <= self.cfg.EOS_idx:
                #     continue
                # pred_scene = deepcopy(bar_scene)
                pred_scene = env.scenes[0]


                ##############################################################
                # Evaluate
                ##############################################################
                pred_scores = env.evaluate_scene(pred_scene, gt_scene)
                scores.append(pred_scores)

                ##############################################################
                # Draw
                ##############################################################
                frame = env.batch_redraw(return_sequence=False)[0][0]

                ##############################################################
                # Save scene
                ##############################################################
                pred_inds = deepcopy(pred_scene['out_inds'])
                pred_inds = np.stack(pred_inds, 0)
                foo = test_db.output_inds_to_scene(pred_inds)
                out_scene = {}
                out_scene['boxes'] = [x.tolist() for x in foo['boxes'].astype(np.float64)]
                out_scene['clses'] = foo['clses'].astype(np.int64).tolist()
                out_scene['caption'] = entry['sentence']
                out_scene['cap_idx'] = int(entry['cap_idx'])
                out_scene['width']   = int(gt_scene['width'])
                out_scene['height']  = int(gt_scene['height'])
                img_idx = int(gt_scene['img_idx'])
                out_scene['img_idx'] = img_idx
                scene_path = osp.join(scene_dir, '%02d_'%G+str(img_idx).zfill(12)+'.json')
                img_path = osp.join(img_dir, '%02d_'%G+str(img_idx).zfill(12)+'.jpg')
                cv2.imwrite(img_path, frame)
                with open(scene_path, 'w') as fp:
                    json.dump(out_scene, fp, indent=4, sort_keys=True)
                print(G, i, img_idx)

                # if len(scores) > 5:
                #     break

        scores = np.stack(scores, 0).astype(np.float64)
        infos = eval_info(self.cfg, scores)
        info_path = osp.join(out_dir, 'eval_info_top1.json')
        log_coco_scores(infos, info_path)

    def sample_demo(self, input_sentences):
        output_dir = osp.join(self.cfg.model_dir, 'layout_samples')
        maybe_create(output_dir)
        num_sents = len(input_sentences)
        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
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
                inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)
            frames = env.batch_redraw(return_sequence=True)[0]
            # _, _, _, _, what_wei, where_wei = inf_outs
            # if self.cfg.what_attn:
            #     what_attn_words = self.decode_attention(
            #         input_inds_np, input_lens_np, what_wei.squeeze(0))
            # if self.cfg.where_attn > 0:
            #     where_attn_words = self.decode_attention(
            #         input_inds_np, input_lens_np, where_wei.squeeze(0))
            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(60, 40))
            plt.suptitle(sentence, fontsize=40)
            for j in range(frames.shape[0]):
                # subtitle = ''
                # if self.cfg.what_attn:
                #     subtitle = subtitle + ' '.join(what_attn_words[j])
                # if self.cfg.where_attn > 0:
                #     subtitle = subtitle + '\n' + ' '.join(where_attn_words[j])
                plt.subplot(4, 3, j+1)
                # plt.title(subtitle, fontsize=30)
                plt.imshow(frames[j, :, :, ::-1])
                plt.axis('off')
            out_path = osp.join(output_dir, '%09d.png'%i)
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

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
        checkpoint_dir = osp.join(self.cfg.model_dir, 'layout_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_name = "ckpt-%03d-%.4f-%.4f.pkl" % (epoch, log[0], log[1])
        torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))




