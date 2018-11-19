#!/bin/bash

NUMS="10"

source activate ban
set -x

BASE=po-weight-max-objects

for i in $NUMS
do
    rm $BASE-$i.txt
    python evaluate.py --input saved_models/eval/$BASE-$i.pth >> $BASE-$i.txt
    python evaluate.py --input saved_models/eval/$BASE-$i.pth --category yes/no >> $BASE-$i.txt
    python evaluate.py --input saved_models/eval/$BASE-$i.pth --category number >> $BASE-$i.txt
    python evaluate.py --input saved_models/eval/$BASE-$i.pth --category other >> $BASE-$i.txt
done
