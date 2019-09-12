#!/usr/bin/env python


import argparse
import os.path as osp


this_dir = osp.dirname(__file__)


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()


# arguments with different values for Abstract Scene and COCO Layout
parser.add_argument('--input_size',  nargs='+', type=int, default=[64, 64]) # 64 64 for COCO Layout, 224 224 for Abstract Scene
parser.add_argument('--input_vocab_size', type=int, default=9908) # 9908 for COCO Layout, 2538 for Abstract Scene
parser.add_argument('--output_cls_size',  type=int, default=83) # 80 + 3 = 83 for COCO Layout, 58 + 3 = 61 for Abstract Scene
parser.add_argument('--max_input_length', type=int, default=8) # 8 for COCO Layout, 5 for Abstract Scene
parser.add_argument('--n_conv_hidden',    type=int, default=256)  # 256 for COCO Layout, 1024 for Abstract Scene 
parser.add_argument('--use_bn', type=str2bool, default=False) # false for COCO Layout, true for Abstract Scene 
parser.add_argument('--obj_loss_weight', type=float, default=8.0) # 8.0 for COCO Layout, 5.0 for Abstract Scene 

# arguments set to the same values for Abstract Scene and COCO Layout
##################################################################
# Data
##################################################################
parser.add_argument('--PAD_idx', type=int, default=0)
parser.add_argument('--SOS_idx', type=int, default=1)
parser.add_argument('--EOS_idx', type=int, default=2)
parser.add_argument('--image_size',  default=[500, 400], help='resolution of the images in Abstract Scene')
parser.add_argument('--draw_size',   default=[1000, 1000], help='resolution of rendered layouts for COCO')
parser.add_argument('--grid_size',   default=[28, 28], help='location resolution')
parser.add_argument('--scales',      default=[1.0, 0.7, 0.49], help='scales for Abstract Scene')
parser.add_argument('--margin',      type=int, default=1, help='margin value for grid locations, lazy trick to handle the edge cases')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loader')
parser.add_argument('--sent_group',  type=int, default=-1, help='sentence group index, there are 2 for Abstract Scene, 5 for COCO, -1 indicates random group')
parser.add_argument('--coco_min_area', type=float, default=0.01, help='threshold for the box size')
parser.add_argument('--num_scales',    type=int,   default=17, help='size resolution for COCO')
parser.add_argument('--num_ratios',    type=int,   default=17, help='aspect ratio resolution for COCO')
parser.add_argument('--max_output_length', type=int, default=9) 
parser.add_argument('--object_first',  type=str2bool, default=False)

##################################################################
# Model
##################################################################

# Text encoder
parser.add_argument('--n_src_hidden',  type=int, default=256)
parser.add_argument('--n_tgt_hidden',  type=int, default=256)
parser.add_argument('--bidirectional', type=str2bool, default=True)
parser.add_argument('--n_rnn_layers',  type=int, default=1)
parser.add_argument('--rnn_cell', type=str, default='GRU')
parser.add_argument('--n_embed',  type=int, default=300, help='GloVec dimension')
parser.add_argument('--emb_dropout_p', type=float, default=0.0)
parser.add_argument('--rnn_dropout_p', type=float, default=0.0)
parser.add_argument('--hidden_pooling_mode', type=int, default=0, help='pooling mode for the rnn features')
parser.add_argument('--shuffle_sents', type=str2bool, default=False, help="deprecated")

# Attention
parser.add_argument('--attn_type',     type=str, default='general', help='attention model to use')
parser.add_argument('--attn_2d',       type=str2bool, default=True)
parser.add_argument('--where_attn_2d', type=str2bool, default=True)
parser.add_argument('--attn_emb',      type=str2bool, default=True)
parser.add_argument('--what_attn', type=str2bool, default=True, help='whether to attention for object prediction')
parser.add_argument('--where_attn', type=int, default=2, help='whether to attention for attribute prediction')

# Decoders
parser.add_argument('--use_bg_to_pred', type=str2bool, default=False, help='whether to use S_t for object prediction, not helpful')
parser.add_argument('--use_fg_to_pred', type=int, default=2, help='how to use the o_{t-1}')
parser.add_argument('--use_bg_to_locate', type=str2bool, default=False, help='whether to use S_t for attribute prediction, not helpful')

##################################################################
# Training parameters
##################################################################
parser.add_argument('--cuda', '-gpu', action='store_true')
parser.add_argument('--teacher_forcing', type=str2bool, default=True)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--finetune_lr', type=float, default=0.0)
parser.add_argument('--grad_norm_clipping', type=float, default=5)
parser.add_argument('--log_per_steps', type=int, default=10)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_epochs',  type=int, default=50)

# loss weights
parser.add_argument('--pose_loss_weight',  type=float, default=2.0)
parser.add_argument('--expr_loss_weight',  type=float, default=2.0)
parser.add_argument('--coord_loss_weight', type=float, default=2.0)
parser.add_argument('--scale_loss_weight', type=float, default=2.0)
parser.add_argument('--ratio_loss_weight', type=float, default=2.0)
parser.add_argument('--flip_loss_weight',  type=float, default=2.0)
parser.add_argument('--attn_loss_weight',  type=float, default=0.0)
parser.add_argument('--eos_loss_weight',   type=float, default=0.0)
##################################################################

##################################################################
# evaluation
##################################################################
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--rel_mode', type=int, default=1, help='relation distance mode, 0 for L2, 1 for polar')
parser.add_argument('--sigmas', default=[0.2, 0.2, 0.2, 0.2], help='gaussian kernel size used in evaluation')
parser.add_argument('--recall_weight', type=float, default=9.0, help="F3 score, deprecated")
##################################################################

##################################################################
# Sampling
##################################################################
parser.add_argument('--beam_size',   type=int, default=8)
parser.add_argument('--sample_mode', type=int, default=0, help="0: top 1, 1: multinomial")
##################################################################

##################################################################
# Misc
##################################################################
parser.add_argument('--exp_name', type=str, default='layout', help='experiment name for logging')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--eps',  type=float, default=1e-10, help='epsilon')
parser.add_argument('--log_dir',  type=str, default=osp.join(this_dir, '..', 'logs'))
parser.add_argument('--data_dir', type=str, default=osp.join(this_dir, '..', 'data'))
parser.add_argument('--root_dir', type=str, default=osp.join(this_dir, '..'))
##################################################################


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed