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
from optim import Optimizer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.image_synthesis_model import ImageSynthesisModel

sp=256

class ImageSynthesisTrainer(object):
    def __init__(self, config):
        self.cfg = config
        net = ImageSynthesisModel(config)
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
        pretrained_path = osp.join(cache_dir, 'synthesis_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(states)

    def save_checkpoint(self):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'synthesis_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_name = "ckpt-best.pkl"
        torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))

    def batch_data(self, entry):
        ################################################
        # Inputs
        ################################################
        proposal_images = entry['proposal_image'].float()
        proposal_labels = entry['proposal_label'].float()
        proposal_masks = entry['proposal_mask'].float()
        gt_images = entry['gt_image'].float()
        gt_labels = entry['gt_label'].float()

        inputs = torch.cat([proposal_images, proposal_labels, proposal_masks], 1)

        inputs = F.interpolate(inputs, size=[sp, 2*sp], mode='bilinear', align_corners=True)
        proposal_labels = F.interpolate(proposal_labels, size=[sp, 2*sp], mode='bilinear', align_corners=True)
        gt_images = F.interpolate(gt_images, size=[sp, 2*sp], mode='bilinear', align_corners=True)
        gt_labels = F.interpolate(gt_labels, size=[sp, 2*sp], mode='bilinear', align_corners=True)

        if self.cfg.cuda:
            inputs = inputs.cuda()
            proposal_labels = proposal_labels.cuda()
            gt_images = gt_images.cuda()
            gt_labels = gt_labels.cuda()

        return inputs, proposal_labels, gt_images, gt_labels

    def weighted_l1_loss(self, inputs, targets, weights):
        d = torch.abs(inputs - targets)
        d = torch.mean(d, 1, keepdim=True)
        d = d * weights
        bsize, nlabels, h, w = d.size()
        d = torch.transpose(d, 0, 1).contiguous()
        d = d.view(nlabels, -1)
        d = torch.mean(d, -1)
        return torch.sum(d)

    def compute_loss(self, proposal_labels,
            synthesized_images, gt_images,
            synthesized_features, gt_features,
            synthesized_labels, gt_labels):
        '''
            proposal_labels: masks used to weight the loss
        '''

        loss_wei = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 1.0/0.15]
        ####################################################################
        # Prediction loss
        ####################################################################
        cross_entropy_metric = nn.CrossEntropyLoss()
        bsize, nlabels, h, w = gt_labels.size()
        gt_labels = torch.transpose(gt_labels, 1, 3).contiguous()
        synthesized_labels = torch.transpose(synthesized_labels, 1, 3).contiguous()
        gt_labels = gt_labels.view(-1, nlabels)
        synthesized_labels = synthesized_labels.view(-1, nlabels)
        pred_loss = cross_entropy_metric(synthesized_labels, torch.max(gt_labels, -1)[-1])

        ####################################################################
        # Perceptual_loss
        ####################################################################

        p1_weights = F.interpolate(proposal_labels, size=[gt_features[1].size(2), gt_features[1].size(3)], mode='bilinear', align_corners=True)
        p2_weights = F.interpolate(proposal_labels, size=[gt_features[2].size(2), gt_features[2].size(3)], mode='bilinear', align_corners=True)
        p3_weights = F.interpolate(proposal_labels, size=[gt_features[3].size(2), gt_features[3].size(3)], mode='bilinear', align_corners=True)
        p4_weights = F.interpolate(proposal_labels, size=[gt_features[4].size(2), gt_features[4].size(3)], mode='bilinear', align_corners=True)

        pi = self.weighted_l1_loss(synthesized_images, gt_images, proposal_labels)
        p0 = self.weighted_l1_loss(synthesized_features[0], gt_features[0], proposal_labels)
        p1 = self.weighted_l1_loss(synthesized_features[1], gt_features[1], p1_weights)
        p2 = self.weighted_l1_loss(synthesized_features[2], gt_features[2], p2_weights)
        p3 = self.weighted_l1_loss(synthesized_features[3], gt_features[3], p3_weights)
        p4 = self.weighted_l1_loss(synthesized_features[4], gt_features[4], p4_weights)

        ####################################################################
        # Weighted loss
        ####################################################################
        loss = pi + p0 * loss_wei[0] + p1 * loss_wei[1] + p2 * loss_wei[2] + p3 * loss_wei[3] + p4 * loss_wei[4] + 10 * pred_loss
        losses = torch.stack([loss.clone(), pred_loss, pi, p0, p1, p2, p3, p4]).flatten()
        return loss, losses

    def train(self, train_db, val_db):
        ##################################################################
        ## Optimizer
        ##################################################################
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        optimizer = optim.Adam([
                {'params': net.encoder.parameters()},
                {'params': net.decoder.parameters()}
            ], lr=self.cfg.lr)

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
            # val_loss = train_loss

            ##################################################################
            ## Sample
            ##################################################################
            torch.cuda.empty_cache()
            self.sample(epoch, val_db, self.cfg.n_samples)
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
                self.save_checkpoint()
                torch.cuda.empty_cache()

    def train_epoch(self, train_db, optimizer, epoch):

        loader = DataLoader(train_db, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers)
        errors_list = []
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            inputs, proposal_labels, gt_images, gt_labels = self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.train()
            self.net.zero_grad()
            synthesized_images, synthesized_labels, synthesized_features, gt_features = \
                self.net(inputs, True, gt_images)

            loss, losses = self.compute_loss(proposal_labels,
                    synthesized_images, gt_images,
                    synthesized_features, gt_features,
                    synthesized_labels, gt_labels)
            loss.backward()
            optimizer.step()

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
        loader = DataLoader(val_db, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers)

        errors_list = []
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            inputs, proposal_labels, gt_images, gt_labels = self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.eval()
            with torch.no_grad():
                synthesized_images, synthesized_labels, synthesized_features, gt_features = \
                    self.net(inputs, True, gt_images)
                loss, losses = self.compute_loss(proposal_labels,
                        synthesized_images, gt_images,
                        synthesized_features, gt_features,
                        synthesized_labels, gt_labels)

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

    def sample(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        plt.switch_backend('agg')
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'vis')
        maybe_create(output_dir)

        ##############################################################
        # Main loop
        ##############################################################
        loader = DataLoader(test_db, batch_size=1, shuffle=random_or_not)

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        max_cnt = min(N, len(test_db))

        for cnt, batched in enumerate(loader):
            ##################################################################
            ## Batched data
            ##################################################################
            inputs, proposal_labels, gt_images, gt_labels = self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            self.net.eval()
            with torch.no_grad():
                synthesized_images, synthesized_labels, _, _ = \
                    self.net(inputs, False, None)

            synthesized_image = unnormalize(synthesized_images[0].cpu().data.numpy())
            gt_image = unnormalize(gt_images[0].cpu().data.numpy())
            synthesized_label = synthesized_labels[0].cpu().data.numpy()
            synthesized_label = test_db.decode_semantic_map(synthesized_label)
            gt_label = gt_labels[0].cpu().data.numpy()
            gt_label = test_db.decode_semantic_map(gt_label)


            fig = plt.figure(figsize=(32, 32))
            plt.subplot(2, 2, 1)
            plt.imshow(clamp_array(synthesized_image[:, :, ::-1], 0, 255).astype(np.uint8))
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(clamp_array(gt_image[:, :, ::-1], 0, 255).astype(np.uint8))
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(clamp_array(synthesized_label, 0, 255).astype(np.uint8))
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(clamp_array(gt_label, 0, 255).astype(np.uint8))
            plt.axis('off')

            image_idx = batched['image_index'][0]
            name = '%03d_'%cnt + str(image_idx).zfill(8)
            out_path = osp.join(output_dir, name+'.png')

            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print('sampling: %d, %d'%(epoch, cnt))

            if cnt >= max_cnt:
                break
