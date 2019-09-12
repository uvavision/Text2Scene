#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_composites.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_composites.py --cuda --parallel --batch_size=4 --use_super_category=True --num_workers=1 --use_patch_background=True --n_shape_hidden=256 --where_attn=2 --where_attn_2d=True
