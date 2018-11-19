import sys
import os
import numpy as np


paths = sys.argv[1:]

scores = []
for path in paths:
    with open(path) as fd:
        log = fd.readlines()
    for line in log:
        line = line.strip()
        if line.startswith('eval'):
            v = float(line.split(' ')[2])
            scores.append(v)

categories = ['overall', 'y/n', 'number', 'other']
for i, cat in enumerate(categories):
    s = scores[i::len(categories)]
    print()
    print(cat)
    print(s)
    print('mean', np.mean(s))
    print('stdev', np.std(s, ddof=1))
