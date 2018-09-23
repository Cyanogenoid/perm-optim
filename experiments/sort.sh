#!/bin/bash

PARAMS="--steps 6 --lr 1e-1 --epochs 1"
TEST_PARAMS="--size-pow 9 --double"  #without double, there aren't enough unique f32 numbers between 1000 and 1001 to get 100% accuracy on because of birthday problem
mkdir -p logs/sort

NUMBERS="5 10 15 80 100 120"  # match gumbel sinkhorn paper

set -x


for i in $NUMBERS
do
    NAME="sort/$i"

    python train.py --train-only --batch-size 512 --name "$NAME" $PARAMS \
        sort --length $i --size-pow 18
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-0-to-1" $PARAMS \
        sort --length $i --low 0 --high 1 $TEST_PARAMS
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-0-to-10" $PARAMS \
        sort --length $i --low 0 --high 10 $TEST_PARAMS
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-0-to-1000" $PARAMS \
        sort --length $i --low 0 --high 1000 $TEST_PARAMS
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-1-to-2" $PARAMS \
        sort --length $i --low 1 --high 2 $TEST_PARAMS
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-10-to-11" $PARAMS \
        sort --length $i --low 10 --high 11 $TEST_PARAMS
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-100-to-101" $PARAMS \
        sort --length $i --low 100 --high 101 $TEST_PARAMS
    python train.py --eval-only --resume "logs/$NAME" --name "$NAME-eval-1000-to-1001" $PARAMS \
        sort --length $i --low 1000 --high 1001 $TEST_PARAMS
done
