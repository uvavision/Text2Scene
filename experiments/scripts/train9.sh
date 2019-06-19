#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train9.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_puzzle_model.py --cuda --parallel --batch_size=4 --use_super_category=False --num_workers=1 --use_patch_background=True --where_attn=2 --n_shape_hidden=128 --where_attn_2d=True