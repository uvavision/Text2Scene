#!/usr/bin/env python

import argparse
import os.path as osp


this_dir = osp.dirname(__file__)


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()


##################################################################
# Data
##################################################################
parser.add_argument('--input_image_size',  nargs='+', type=int, default=[128, 128])
parser.add_argument('--output_image_size', nargs='+', type=int, default=[256, 256])
parser.add_argument('--input_patch_size',  nargs='+', type=int, default=[64, 64])
parser.add_argument('--input_vocab_size', type=int, default=8552)
parser.add_argument('--output_vocab_size', type=int, default=98)
parser.add_argument('--use_super_category', type=str2bool, default=True)
parser.add_argument('--grid_size', default=[32, 32])
parser.add_argument('--num_scales', type=int, default=17)
parser.add_argument('--num_ratios', type=int, default=17)
parser.add_argument('--pixel_means', nargs='+', type=int, default=[103.53, 116.28, 123.675])
parser.add_argument('--coco_min_area', type=float, default=0.01)
parser.add_argument('--coco_min_ratio', type=float, default=0.5)
parser.add_argument('--sent_group', type=int, default=-1)

##################################################################
# Language vocabulary
##################################################################
parser.add_argument('--PAD_idx', type=int, default=0)
parser.add_argument('--SOS_idx', type=int, default=1)
parser.add_argument('--EOS_idx', type=int, default=2)

##################################################################
# Model
##################################################################
parser.add_argument('--use_normalization', type=str2bool, default=False)
parser.add_argument('--use_global_resnet', type=str2bool, default=False)
parser.add_argument('--use_resnet',        type=str2bool, default=True)
parser.add_argument('--use_color_volume',  type=str2bool, default=True)
parser.add_argument('--use_hard_mining',   type=str2bool, default=False)
parser.add_argument('--max_input_length',  type=int, default=13)
parser.add_argument('--max_output_length', type=int, default=11)
parser.add_argument('--n_patch_features', type=int, default=128)
parser.add_argument('--n_nntable_trees', type=int, default=20)
parser.add_argument('--use_patch_background', type=str2bool, default=True)
parser.add_argument('--use_separable_convolution', type=str2bool, default=False)
# Text encoder
parser.add_argument('--n_conv_hidden', type=int, default=512)
parser.add_argument('--n_src_hidden',  type=int, default=256)
parser.add_argument('--n_tgt_hidden',  type=int, default=256)
parser.add_argument('--n_shape_hidden', type=int, default=256)
parser.add_argument('--bidirectional', type=str2bool, default=True)
parser.add_argument('--n_rnn_layers',  type=int, default=1)
parser.add_argument('--rnn_cell', type=str, default='GRU')
parser.add_argument('--n_embed',  type=int, default=300)
parser.add_argument('--emb_dropout_p', type=float, default=0.0)
parser.add_argument('--rnn_dropout_p', type=float, default=0.0)
# Attention
parser.add_argument('--attn_type', type=str, default='general')
parser.add_argument('--attn_emb',  type=str2bool, default=True)
parser.add_argument('--what_attn', type=str2bool, default=True)
parser.add_argument('--where_attn', type=int, default=2)
parser.add_argument('--what_attn_2d', type=str2bool, default=True)
parser.add_argument('--where_attn_2d', type=str2bool, default=True)
# Decoders
parser.add_argument('--use_bg_to_pred', type=str2bool, default=False)
parser.add_argument('--use_fg_to_pred', type=int, default=2)
parser.add_argument('--use_bg_to_locate', type=str2bool, default=False)
##################################################################
# Training parameters
##################################################################
parser.add_argument('--cuda', '-gpu', action='store_true')
parser.add_argument('--teacher_forcing', type=str2bool, default=True)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loader')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--finetune_lr', type=float, default=1e-5)
parser.add_argument('--grad_norm_clipping', type=float, default=10)
parser.add_argument('--log_per_steps', type=int, default=10)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--n_epochs',  type=int, default=100)
parser.add_argument('--margin', type=float, default=0.3)

# loss weights
parser.add_argument('--obj_loss_weight',   type=float, default=5.0)
parser.add_argument('--coord_loss_weight', type=float, default=2.0)
parser.add_argument('--scale_loss_weight', type=float, default=1.0)
parser.add_argument('--ratio_loss_weight', type=float, default=1.0)
parser.add_argument('--embed_loss_weight', type=float, default=10.0)
parser.add_argument('--attn_loss_weight',  type=float, default=0.5)
parser.add_argument('--regression_loss_weight', type=float, default=1.0)
parser.add_argument('--weighted_synthesis',  type=str2bool, default=False)
##################################################################


##################################################################
# evaluation
##################################################################
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--composer_pretrained', type=str, default=None)
parser.add_argument('--inpainter_pretrained', type=str, default=None)
parser.add_argument('--for_visualization', type=str2bool, default=True)
##################################################################


##################################################################
# Misc
##################################################################
parser.add_argument('--exp_name', type=str, default='composites', help='experiment name for logging')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--eps',  type=float, default=1e-10, help='epsilon')
parser.add_argument('--log_dir',  type=str, default=osp.join(this_dir, '..', 'logs'))
parser.add_argument('--data_dir', type=str, default=osp.join(this_dir, '..', 'data'))
parser.add_argument('--root_dir', type=str, default=osp.join(this_dir, '..'))
##################################################################


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
