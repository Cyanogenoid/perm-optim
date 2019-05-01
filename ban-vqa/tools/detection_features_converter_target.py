"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa

Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import utils

target = 'test2015'

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'data/%s_36/%s_resnet101_faster_rcnn_genome_36.tsv' % (target, target)
data_file = 'data/%s36.hdf5' % target
indices_file = 'data/%s36_imgid2idx.pkl' % target
ids_file = 'data/%s_ids.pkl' % target

feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    h = h5py.File(data_file, "w")

    if os.path.exists(ids_file):
        imgids = cPickle.load(open(ids_file, 'rb'))
    else:
        imgids = utils.load_imageid('data/%s' % target)
        cPickle.dump(imgids, open(ids_file, 'wb'))

    indices = {}

    img_bb = h.create_dataset(
        'image_bb', (len(imgids), num_fixed_boxes, 4), 'f')
    img_features = h.create_dataset(
        'image_features', (len(imgids), num_fixed_boxes, feature_length), 'f')
    spatial_img_features = h.create_dataset(
        'spatial_features', (len(imgids), num_fixed_boxes, 6), 'f')

    counter = 0

    print("reading tsv...")
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])
            item['boxes'] = bytes(item['boxes'], 'utf')
            item['features'] = bytes(item['features'], 'utf')
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in imgids:
                imgids.remove(image_id)
                indices[image_id] = counter
                img_bb[counter, :, :] = bboxes
                img_features[counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                spatial_img_features[counter, :, :] = spatial_features
                counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(imgids) != 0:
        print('Warning: image_ids is not empty')

    cPickle.dump(indices, open(indices_file, 'wb'))
    h.close()
    print("done!")
