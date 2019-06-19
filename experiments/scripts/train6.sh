#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train6.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_puzzle_model.py --cuda --parallel --batch_size=12 --use_super_category=True --num_workers=1 --use_patch_background=True --n_shape_hidden=128 --use_color_volume=False --where_attn=2 --where_attn_2d=True
