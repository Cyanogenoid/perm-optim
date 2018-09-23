#!/bin/bash

DATASET="imagenet"
TILES="2 3 4 5"
PARAMS="--epochs 1 --no-hard-assign --multi-gpu --num-workers 10"

mkdir -p logs/classify

set -x



for i in $TILES
do
    NAME="classify/$DATASET-$i"
    PARAMS2="--tiles-per-side $i --freeze-resblocks --dataset $DATASET --conv-channels 128"

    python train.py --input-sorted --name "$NAME" --multi-gpu --num-workers 16 --steps 0 --no-hard-assign --epochs 4 \
        classify --dataset $DATASET --tiles-per-side $i --no-permute

    python train.py --input-sorted --eval-only --resume "logs/$NAME" --name "$NAME-max" --steps 0 $PARAMS \
        classify $PARAMS2 --no-permute  # sanity check, should match the last eval of the command above
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-baseline" --steps 0 $PARAMS \
        classify $PARAMS2 --no-permute
    python train.py --resume "logs/$NAME" --name "$NAME-both" --steps 4 $PARAMS \
        classify $PARAMS2 --avoid-nans
    python train.py --resume "logs/$NAME" --name "$NAME-optim" --steps 4 $PARAMS \
        classify $PARAMS2 --uniform-init
    python train.py --resume "logs/$NAME" --name "$NAME-linear" --steps 0 $PARAMS \
        classify $PARAMS2 --avoid-nans
done
