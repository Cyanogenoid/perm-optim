#!/bin/bash

DATASET="imagenet"
TILES="2 3 4 5"
PARAMS="--epochs 1 --num-workers 16 --multi-gpu"

mkdir -p logs/mosaic

set -x


for i in $TILES
do
    NAME="mosaic/$DATASET-$i"
    PARAMS2="--tiles-per-side $i --dataset $DATASET --conv-channels 128"

    python train.py --name "$NAME-both" --steps 4 $PARAMS \
        mosaic $PARAMS2 --avoid-nans
    python train.py --name "$NAME-optim" --steps 4 $PARAMS \
        mosaic $PARAMS2 --uniform-init
    python train.py --name "$NAME-linear" --steps 0 $PARAMS \
        mosaic $PARAMS2 --avoid-nans
done
