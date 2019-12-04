#!/usr/bin/env python

import os, sys, cv2, math
import random, json, logz
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
from glob import glob
from composites_utils import *
from composites_config import get_config
from optim import Optimizer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.composites_loader import sequence_loader, proposal_loader

from modules.synthesis_model import SynthesisModel


class SynthesisTrainer(object):
    def __init__(self, config):
        self.cfg = config
        self.net = SynthesisModel(config)
        if self.cfg.cuda:
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        raw_optimizer = optim.Adam([
                {'params': net.encoder.parameters()},
                {'params': net.decoder.parameters()}
            ], lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer)
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
        pretrained_path = osp.join(cache_dir, 'synthesis_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        # states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(states['state_dict'])
        self.optimizer.optimizer.load_state_dict(states['optimizer'])
        self.epoch = states['epoch']

    def save_checkpoint(self, epoch):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'synthesis_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()        
        }
        torch.save(states, osp.join(checkpoint_dir, "ckpt-%03d.pkl"%epoch))

    def batch_data(self, entry):
        ################################################
        # Inputs
        ################################################
        proposals = entry['input_vol'].float()
        gt_images = entry['gt_image'].float()
        gt_labels = entry['gt_label'].long()

        if self.cfg.cuda:
            proposals = proposals.cuda(non_blocking=True)
            gt_images = gt_images.cuda(non_blocking=True)
            gt_labels = gt_labels.cuda(non_blocking=True)

        return proposals, gt_images, gt_labels

    def weighted_l1_loss(self, inputs, targets, weights=None):
        # masked L1 loss on features
        d = torch.abs(inputs - targets)
        d = torch.mean(d, 1, keepdim=False)
        if weights is not None:
            d = d * weights
        d = torch.mean(d)
        return d

    def compute_loss(self, synthesized_images, gt_images,
            synthesized_features, gt_features,
            synthesized_labels, gt_labels, weights=None):

        loss_wei = [50.0, 20.0, 14.0, 20.0, 35.0]
        ####################################################################
        # Prediction loss
        ####################################################################
        cross_entropy_metric = nn.CrossEntropyLoss()
        bsize, nlabels, h, w = synthesized_labels.size()
        synthesized_labels = synthesized_labels.permute(0, 2, 3, 1).contiguous()
        synthesized_labels = synthesized_labels.view(-1, nlabels)
        gt_labels = gt_labels.view(-1)
        pred_loss = cross_entropy_metric(synthesized_labels, gt_labels)

        ####################################################################
        # Perceptual_loss
        ####################################################################

        if weights is not None:
            p1_weights = F.interpolate(weights, size=[gt_features[1].size(2), gt_features[1].size(3)], mode='bilinear', align_corners=True)
            p2_weights = F.interpolate(weights, size=[gt_features[2].size(2), gt_features[2].size(3)], mode='bilinear', align_corners=True)
            p3_weights = F.interpolate(weights, size=[gt_features[3].size(2), gt_features[3].size(3)], mode='bilinear', align_corners=True)
            p4_weights = F.interpolate(weights, size=[gt_features[4].size(2), gt_features[4].size(3)], mode='bilinear', align_corners=True)

            pi = self.weighted_l1_loss(synthesized_images, gt_images, weights)
            p0 = self.weighted_l1_loss(synthesized_features[0], gt_features[0], weights)
            p1 = self.weighted_l1_loss(synthesized_features[1], gt_features[1], p1_weights)
            p2 = self.weighted_l1_loss(synthesized_features[2], gt_features[2], p2_weights)
            p3 = self.weighted_l1_loss(synthesized_features[3], gt_features[3], p3_weights)
            p4 = self.weighted_l1_loss(synthesized_features[4], gt_features[4], p4_weights)
        else:
            pi = self.weighted_l1_loss(synthesized_images, gt_images)
            p0 = self.weighted_l1_loss(synthesized_features[0], gt_features[0])
            p1 = self.weighted_l1_loss(synthesized_features[1], gt_features[1])
            p2 = self.weighted_l1_loss(synthesized_features[2], gt_features[2])
            p3 = self.weighted_l1_loss(synthesized_features[3], gt_features[3])
            p4 = self.weighted_l1_loss(synthesized_features[4], gt_features[4])

        ####################################################################
        # Weighted loss
        ####################################################################
        p0 = p0 * loss_wei[0]; p1 = p1 * loss_wei[1];
        p2 = p2 * loss_wei[2]; p3 = p3 * loss_wei[3]; p4 = p4 * loss_wei[4]
        pred_loss = 10 * pred_loss
        loss = pi + p0 + p1 + p2 + p3 + p4 + pred_loss
        losses = torch.stack([loss, pred_loss, pi, p0, p1, p2, p3, p4]).clone().flatten()
        return loss, losses

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
            train_loss = self.train_epoch(train_db, epoch)

            ##################################################################
            ## Validation
            ##################################################################
            torch.cuda.empty_cache()
            val_loss = self.validate_epoch(val_db, epoch)
            # val_loss = train_loss

            ##################################################################
            ## Sample
            ##################################################################
            torch.cuda.empty_cache()
            self.sample_for_vis(epoch, test_db, self.cfg.n_samples)
            torch.cuda.empty_cache()
            ##################################################################
            ## Logging
            ##################################################################

            # update optim scheduler
            current_val_loss = np.mean(val_loss)

            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)
            logz.log_tabular("AverageTotalError", np.mean(train_loss[:, 0]))
            logz.log_tabular("AveragePredError",  np.mean(train_loss[:, 1]))
            logz.log_tabular("AverageImageError", np.mean(train_loss[:, 2]))
            logz.log_tabular("AverageFeat0Error", np.mean(train_loss[:, 3]))
            logz.log_tabular("AverageFeat1Error", np.mean(train_loss[:, 4]))
            logz.log_tabular("AverageFeat2Error", np.mean(train_loss[:, 5]))
            logz.log_tabular("AverageFeat3Error", np.mean(train_loss[:, 6]))
            logz.log_tabular("AverageFeat4Error", np.mean(train_loss[:, 7]))
            logz.log_tabular("ValAverageTotalError", np.mean(val_loss[:, 0]))
            logz.log_tabular("ValAveragePredError",  np.mean(val_loss[:, 1]))
            logz.log_tabular("ValAverageImageError", np.mean(val_loss[:, 2]))
            logz.log_tabular("ValAverageFeat0Error", np.mean(val_loss[:, 3]))
            logz.log_tabular("ValAverageFeat1Error", np.mean(val_loss[:, 4]))
            logz.log_tabular("ValAverageFeat2Error", np.mean(val_loss[:, 5]))
            logz.log_tabular("ValAverageFeat3Error", np.mean(val_loss[:, 6]))
            logz.log_tabular("ValAverageFeat4Error", np.mean(val_loss[:, 7]))
            logz.dump_tabular()

            ##################################################################
            ## Checkpoint
            ##################################################################
            if min_val_loss > current_val_loss:
                min_val_loss = current_val_loss
            self.save_checkpoint(epoch)
            torch.cuda.empty_cache()

    def train_epoch(self, train_db, epoch):
        syn_db = synthesis_loader(train_db)
        loader = DataLoader(syn_db, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True)
        errors_list = []

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        self.net.train()
        net.lossnet.eval()
        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            proposals, gt_images, gt_labels = self.batch_data(batched)
            gt_images = gt_images.permute(0,3,1,2)

            weights = None
            if self.cfg.weighted_synthesis:
                weights = proposals[:,:,:,-4].clone().detach()
                weights = 0.5 * (1.0 + weights)

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.zero_grad()
            synthesized_images, synthesized_labels, synthesized_features, gt_features = \
                self.net(proposals, True, gt_images)
            loss, losses = self.compute_loss(synthesized_images, gt_images, 
                synthesized_features, gt_features, synthesized_labels, gt_labels, weights)
            loss.backward()
            self.optimizer.step()

            ##################################################################
            ## Collect info
            ##################################################################
            errors_list.append(losses.cpu().data.numpy().flatten())

            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                tmp = np.stack(errors_list, 0)
                print('Epoch %03d, iter %07d:'%(epoch, cnt))
                print(np.mean(tmp[:,0]), np.mean(tmp[:,1]), np.mean(tmp[:,2]))
                print(np.mean(tmp[:,3]), np.mean(tmp[:,4]), np.mean(tmp[:,5]), np.mean(tmp[:,6]), np.mean(tmp[:,7]))
                print('-------------------------')

        return np.array(errors_list)

    def validate_epoch(self, val_db, epoch):
        syn_db = synthesis_loader(val_db)
        loader = DataLoader(syn_db, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True)
        errors_list = []
        # if self.cfg.cuda and self.cfg.parallel:
        #     net = self.net.module
        # else:
        #     net = self.net
        self.net.eval()
        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            proposals, gt_images, gt_labels = self.batch_data(batched)
            gt_images = gt_images.permute(0,3,1,2)

            weights = None
            if self.cfg.weighted_synthesis:
                weights = proposals[:,:,:,-4].clone().detach()
                weights = 0.5 * (1.0 + weights)

            ##################################################################
            ## Validate one step
            ##################################################################
            with torch.no_grad():
                synthesized_images, synthesized_labels, synthesized_features, gt_features = \
                    self.net(proposals, True, gt_images)
                
                loss, losses = self.compute_loss(synthesized_images, gt_images, 
                    synthesized_features, gt_features,
                    synthesized_labels, gt_labels, weights)

            ##################################################################
            ## Collect info
            ##################################################################
            errors_list.append(losses.cpu().data.numpy().flatten())

            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                tmp = np.stack(errors_list, 0)
                print('Val epoch %03d, iter %07d:'%(epoch, cnt))
                print(np.mean(tmp[:,0]), np.mean(tmp[:,1]), np.mean(tmp[:,2]))
                print(np.mean(tmp[:,3]), np.mean(tmp[:,4]), np.mean(tmp[:,5]), np.mean(tmp[:,6]), np.mean(tmp[:,7]))
                print('-------------------------')

        return np.array(errors_list)

    def sample_for_vis(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        plt.switch_backend('agg')
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'vis')
        maybe_create(output_dir)

        ##############################################################
        # Main loop
        ##############################################################
        syn_db = synthesis_loader(test_db)
        loader = DataLoader(syn_db, batch_size=self.cfg.batch_size, shuffle=random_or_not, pin_memory=True)

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        max_cnt = min(N, len(test_db.scenedb))

        self.net.eval()
        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            proposals, gt_images, gt_labels = self.batch_data(batched)
            image_indices = batched['image_index'].cpu().data.numpy()

            ##################################################################
            ## Train one step
            ##################################################################
            with torch.no_grad():
                synthesized_images, synthesized_labels, _, _ = \
                    self.net(proposals, False, None)

            for i in range(synthesized_images.size(0)):
                synthesized_image = synthesized_images[i].cpu().data.numpy()
                synthesized_image = synthesized_image.transpose((1,2,0))
                gt_image = gt_images[i].cpu().data.numpy()

                synthesized_label = torch.max(synthesized_labels[i], 0)[-1]
                synthesized_label = synthesized_label.cpu().data.numpy()
                synthesized_label = test_db.decode_semantic_map(synthesized_label)
                gt_label = gt_labels[i].cpu().data.numpy()
                gt_label = test_db.decode_semantic_map(gt_label)

                fig = plt.figure(figsize=(32, 32))
                plt.subplot(2, 2, 1)
                plt.imshow(clamp_array(synthesized_image, 0, 255).astype(np.uint8))
                plt.axis('off')
                plt.subplot(2, 2, 2)
                plt.imshow(clamp_array(gt_image, 0, 255).astype(np.uint8))
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.imshow(clamp_array(synthesized_label, 0, 255).astype(np.uint8))
                plt.axis('off')
                plt.subplot(2, 2, 4)
                plt.imshow(clamp_array(gt_label, 0, 255).astype(np.uint8))
                plt.axis('off')

                image_idx = image_indices[i]
                name = '%03d_'%cnt + str(image_idx).zfill(12)
                out_path = osp.join(output_dir, name+'.png')

                fig.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print('sampling: %d, %d, %d'%(epoch, cnt, i))

            if (cnt+1) * self.cfg.batch_size >= max_cnt:
                break
    
    def sample_for_eval(self, test_db):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, 'synthesis_images')
        maybe_create(output_dir)
        # test_db.scenedb = test_db.scenedb[:2980] + test_db.scenedb[3000:]
        ##############################################################
        # Main loop
        ##############################################################
        syn_db = proposal_loader(test_db)
        loader = DataLoader(syn_db, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True)

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net


        self.net.eval()
        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            proposals = batched['input_vol'].float()
            if self.cfg.cuda:
                proposals = proposals.cuda()
            boxes = batched['box'].long().cpu().data.numpy()
            image_indices = batched['image_index'].cpu().data.numpy()

            ##################################################################
            ## Train one step
            ##################################################################
            with torch.no_grad():
                synthesized_images, _, _, _ = \
                    self.net(proposals, False, None)

            for i in range(synthesized_images.size(0)):
                xyxy = boxes[i]
                synthesized_image = synthesized_images[i].cpu().data.numpy()
                synthesized_image = synthesized_image.transpose((1,2,0))
                synthesized_image = clamp_array(synthesized_image, 0, 255).astype(np.uint8)
                synthesized_image = synthesized_image[:,:,::-1].copy()

                # synthesized_image = synthesized_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]

                image_idx = image_indices[i]
                name = str(image_idx).zfill(12)
                out_path = osp.join(output_dir, 'COCO_val2014_'+name+'.jpg')
                cv2.imwrite(out_path, synthesized_image)
                print('sampling: %d, %d'%(cnt, i))
