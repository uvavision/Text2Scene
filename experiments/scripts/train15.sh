#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train15.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_puzzle_model.py --cuda --parallel --batch_size=4 --num_workers=1 --use_super_category=True --use_patch_background=True --n_shape_hidden=256 --use_resnet=False --where_attn=2 --where_attn_2d=True --n_src_hidden=256 --n_tgt_hidden=256 --embed_loss_weight=20.0