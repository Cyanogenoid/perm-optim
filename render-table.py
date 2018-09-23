import json
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('metric')
parser.add_argument('path')
args = parser.parse_args()


with open(args.path) as fd:
    j = json.load(fd)

data = {}
for key in sorted(j.keys()):
    value = j[key]
    task, name = key.split('/')
    dataset, tiles, *version = name.split('-')
    if not version:
        version = None
    else:
        version = version[0]
    data[dataset, tiles, version] = j[key]


accepted_tiles = [2, 3, 4, 5]
accepted_datasets = ['mnist', 'cifar10', 'imagenet']
rename_dataset = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR10',
    'imagenet': 'ImageNet',
}
rename_version = {
    None: '\\emph{max}',
    'baseline': '\\emph{min}',
    'linear': 'LinAssign',
    'optim': 'PO-U',
    'both': 'PO-LA',
}
if 'mosaic' in args.path:
    accepted_versions = ['linear', 'optim', 'both']
else:
    accepted_versions = [None, 'baseline', 'linear', 'optim', 'both']

# header
print('\\toprule')
print('&', ' & '.join(r'\multicolumn{{4}}{{c}}{{{}}}'.format(rename_dataset[ds]) for ds in accepted_datasets), '\\\\')
print('\\cmidrule(l{4pt}){2-5} \\cmidrule(l{4pt}){6-9} \\cmidrule(l{4pt}){10-13}')
print('Model &', ' & '.join(map('\small ${0} \\times {0}$'.format, accepted_tiles * len(accepted_datasets))), '\\\\')
print('\\midrule')

# find entry to bolden
best = set()
for ds, tiles in itertools.product(accepted_datasets, accepted_tiles):
    reduction = min if not 'acc' in args.metric else max
    best_version = reduction([(version, data[ds, str(tiles), version][args.metric]) for version in accepted_versions if version != None], key=lambda x: x[1])[0]
    best.add((ds, str(tiles), best_version))
format_best = lambda d, t, v, x: x if (d, str(t), v) not in best else '\\textbf{{{}}}'.format(x)

# body
lines = []
for version in accepted_versions:
    stats = [data[ds, str(tiles), version] for ds, tiles in itertools.product(accepted_datasets, accepted_tiles)]
    metric = [stat[args.metric] for stat in stats]
    format_str = '{:.2f}'
    if 'acc' in args.metric:
        metric = [m * 100 for m in metric]
        format_str = '{:.1f}'
    metric = map(format_str.format, metric)
    metric = [format_best(d, t, version, x) for (d, t), x in zip(itertools.product(accepted_datasets, accepted_tiles), metric)]
    if version is None or version is 'baseline':
        metric = [f'\\textit{{{x}}}' for x in metric]
    metric_str = ' & '.join(metric)
    v = rename_version[version]
    s = f'{v} & {metric_str} \\\\'
    lines.append(s)
print('\n'.join(lines))
print('\\bottomrule')
