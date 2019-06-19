#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/sample_syn1.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_synthesis.py --cuda --batch_size=4 --num_workers=1 --use_color_volume=False --use_normalization=False --pretrained=final_syn