#!/bin/bash

set -x

python reconstruct-mosaics.py --resume logs/mosaic/mnist-3-optim classify --conv-channels 64 --tiles-per-side 3 --dataset mnist --uniform-init
python reconstruct-mosaics.py --resume logs/classify/mnist-3-optim classify --conv-channels 64 --tiles-per-side 3 --dataset mnist --uniform-init

python reconstruct-mosaics.py --resume logs/mosaic/cifar10-3-optim classify --conv-channels 64 --tiles-per-side 3 --dataset cifar10 --uniform-init
python reconstruct-mosaics.py --resume logs/classify/cifar10-3-optim classify --conv-channels 64 --tiles-per-side 3 --dataset cifar10 --uniform-init

python reconstruct-mosaics.py --resume logs/mosaic/imagenet-3-optim classify --conv-channels 128 --tiles-per-side 3 --dataset imagenet --uniform-init
python reconstruct-mosaics.py --resume logs/classify/imagenet-3-optim classify --conv-channels 128 --tiles-per-side 3 --dataset imagenet --uniform-init

python reconstruct-mosaics.py --resume logs/classify/cifar10-2-optim classify --conv-channels 64 --tiles-per-side 2 --dataset cifar10 --uniform-init
python reconstruct-mosaics.py --resume logs/classify/imagenet-2-optim classify --conv-channels 128 --tiles-per-side 2 --dataset imagenet --uniform-init
