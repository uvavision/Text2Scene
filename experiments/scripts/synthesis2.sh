#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/synthesis2.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_synthesis.py --cuda --parallel --batch_size=16 --num_workers=2 --use_color_volume=False --use_normalization=True --lr=1e-5 