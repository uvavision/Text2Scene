#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/sample17_eval.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/puzzle_model_inference.py --for_visualization=False --cuda --use_super_category=True --use_patch_background=True --n_shape_hidden=256 --use_resnet=False --what_attn=True --what_attn_2d=True --where_attn=0 --where_attn_2d=False --n_src_hidden=256 --n_tgt_hidden=256 --pretrained=train17-010
