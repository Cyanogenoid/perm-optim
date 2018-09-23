#!/bin/bash

DATASET="$1"
TILES="2 3 4 5"
PARAMS="--epochs 20 --no-hard-assign"

mkdir -p logs/classify

set -x


for i in $TILES
do
    NAME="classify/$DATASET-$i"
    PARAMS2="--tiles-per-side $i --freeze-resblocks --dataset $DATASET --conv-channels 64"

    python train.py --input-sorted --name "$NAME" --steps 0 --no-hard-assign --epochs 20 \
        classify --dataset $DATASET --tiles-per-side $i --no-permute
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-baseline" --steps 0 $PARAMS \
        classify $PARAMS2 --no-permute
    python train.py --resume "logs/$NAME" --name "$NAME-both" --steps 4 $PARAMS \
        classify $PARAMS2 --avoid-nans
    python train.py --resume "logs/$NAME" --name "$NAME-optim" --steps 4 $PARAMS \
        classify $PARAMS2 --uniform-init
    python train.py --resume "logs/$NAME" --name "$NAME-linear" --steps 0 $PARAMS \
        classify $PARAMS2 --avoid-nans
done
