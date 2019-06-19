#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train8.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_puzzle_model.py --cuda --parallel --batch_size=4 --use_super_category=True --num_workers=1 --use_patch_background=True --use_global_resnet=True --n_shape_hidden=128 --where_attn=2 --where_attn_2d=True