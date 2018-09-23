#!/bin/bash

experiments/sort.sh
experiments/mosaic.sh mnist
experiments/mosaic.sh cifar10
experiments/mosaic-imagenet.sh
experiments/classify.sh mnist
experiments/classify.sh cifar10
experiments/classify-imagenet.sh

python collect-logs.py sort
python collect-logs.py mosaic
python collect-logs.py classify

python render-table.py test_acc logs/mosaic.json
python render-table.py test_acc logs/classify.json
