import argparse
import os
import json

import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('task', choices=['sort', 'mosaic', 'classify'])
parser.add_argument('path', nargs='?')
args = parser.parse_args()


def load_logs(path):
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        log = torch.load(full_path)
        yield log


def last_epoch_stats(tracker, keys):
    d = {}
    for key in keys:
        last_epoch = tracker[key][-1]
        d[key] = np.mean(last_epoch)
    return d


def keys_for_task(task):
    keys = {
        'sort': {'acc'},
        'mosaic': {'acc', 'l2', 'l1'},
        'classify': {'acc', 'l2', 'l1', 'loss'},
    }[task]
    return sorted('test_{}'.format(key) for key in keys)


data = {}
keys = keys_for_task(args.task)
path = os.path.join('logs', args.task) if not args.path else args.path
for log in load_logs(path):
    print('processing', log['name'])
    data[log['name']] = last_epoch_stats(log['tracker'], keys)

target_name = '{}.json'.format(args.task)
with open(os.path.join('logs', target_name), 'w') as fd:
    json.dump(data, fd)


# crazy formatting into a table
print()
print('name\t\t\t{}\n'.format('\t'.join(k.split('_')[1] for k in keys)))
for experiment in sorted(data.keys()):
    experiment_data = data[experiment]
    d = [experiment_data[key] for key in keys]
    formatted_data = map('{:.4f}'.format, d)
    s = '{}\t{}'.format(experiment.split('/')[1].ljust(20), '\t'.join(formatted_data))
    print(s)
