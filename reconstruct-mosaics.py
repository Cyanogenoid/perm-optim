import os
import argparse
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.optimize
import numpy as np

import data
import track
from model import MosaicNet, PermNet, ClassifyNet


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
subparser.required = True
# generic params
parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), help='Name to store the log file as')
parser.add_argument('--resume', help='Path to log file to resume from')
parser.add_argument('--steps', type=int, default=4, help='Number of inner gradient descent steps')
parser.add_argument('--temp', type=float, default=1, help='Temperature of sinkhorn operator')
parser.add_argument('--inner-lr', type=float, default=1, help='Initial value for learning rate of inner gradient descent')
parser.add_argument('--no-hard-assign', action='store_true', help='Disable Hungarian algorithm at eval time')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train with')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of model')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size to train with')
parser.add_argument('--num-workers', type=int, default=0, help='Number of threads for data loader')
parser.add_argument('--no-cuda', action='store_true', help='Run on CPU instead of GPU (not recommended)')
parser.add_argument('--train-only', action='store_true', help='Only run training, no evaluation')
parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, no training')
parser.add_argument('--input-sorted', action='store_true', help='Input the correctly sorted sequence instead of random order')
parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs')
# mosaic and classify
for task in ['mosaic', 'classify']:
    task_parser = subparser.add_parser(task)
    task_parser.add_argument('--tiles-per-side', type=int, default=3, help='How many tiles per side to split an image in')
    task_parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'imagenet'], default='mnist', help='Dataset to use as base')
    task_parser.add_argument('--uniform-init', action='store_true', help='Disable initial assignment through linear assignment')
    task_parser.add_argument('--no-permute', action='store_true', help='Randomly assemble image instead of learning a permutation')
    task_parser.add_argument('--avoid-nans', action='store_true', help='Gumbel-sinkhorn model can run into numerical issues, use this if it\'s acting up')
    task_parser.add_argument('--imagenet-path', help='Path to ImageNet data', default='/ssd/ILSVRC2012')
    task_parser.add_argument('--conv-channels', help='How many conv channels to use', type=int, default=64)
classify_parser = task_parser
classify_parser.add_argument('--freeze-resblocks', action='store_true', help='Freeze the weights within the residual tower')
args = parser.parse_args()


if args.task == 'mosaic':
    args.task = 'classify'

if args.task == 'sort':
    if not args.size_pow_test:
        args.size_pow_test = args.size_pow
    dataset_train = data.SortNumbers(args.length, low=args.low, high=args.high, size=2**args.size_pow, double=args.double)
    dataset_test = data.SortNumbers(args.length, low=args.low, high=args.high, size=2**args.size_pow_test, double=args.double)
    net = PermNet(steps=args.steps, temp=args.temp)
    net.lr = torch.nn.Parameter((args.inner_lr * net.lr).detach())
elif args.task == 'mosaic' or args.task == 'classify':
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./mnist', download=True, train=True)
        dataset_test = datasets.MNIST('./mnist', download=True, train=False)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        input_channels = 1
        image_size = 28
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('./cifar10', download=True, train=True)
        dataset_test = datasets.CIFAR10('./cifar10', download=True, train=False)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        input_channels = 3
        image_size = 32
    elif args.dataset == 'imagenet':
        dataset_train = datasets.ImageFolder(os.path.join(args.imagenet_path, 'train'))
        dataset_test = datasets.ImageFolder(os.path.join(args.imagenet_path, 'val'))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_channels = 3
        image_size = 64
    else:
        raise ValueError
    dataset_train = data.Mosaic(dataset_train, num_tiles=args.tiles_per_side, transform=transform, image_size=image_size)
    dataset_test = data.Mosaic(dataset_test, num_tiles=args.tiles_per_side, transform=transform, image_size=image_size)
    if args.task == 'mosaic':
        net_class = MosaicNet
    else:
        net_class = ClassifyNet
    net = net_class(args.tiles_per_side, input_channels, conv_channels=args.conv_channels, tile_size=dataset_train.tile_size, uniform_init=args.uniform_init, steps=args.steps, temp=args.temp, avoid_nans=args.avoid_nans)
    if args.task == 'classify' and args.dataset == 'imagenet':
        net.init_imagenet()
    if args.task == 'classify' and args.freeze_resblocks:
        net.lock_residual_params()
    if args.inner_lr != 1.0:
        raise NotImplementedError

if args.task == 'classify':
    net.no_permute = args.no_permute

if args.input_sorted:
    print('Warning: Input will be already correctly sorted. Don\'t use this setting with models that are not permutation invariant.')

if not args.no_cuda:
    net = net.cuda()
if args.task == 'sort' and args.double:
    net = net.double()
if args.multi_gpu:
    @property
    def lr_getter(self):
        return self.module.lr
    torch.nn.DataParallel.lr = lr_getter
    net = torch.nn.DataParallel(net)

# not passing shuffle=False here for variety of classes (e.g. in ImageNet all the first images appear to be fish...)
test_loader = data.get_loader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

if args.resume:
    log = torch.load(args.resume)
    weights = log['weights']
    n = net
    strict = True
    if args.multi_gpu:
        n = n.module
    if args.task == 'classify':
        if 'mosaic' in args.resume:
            n = n.mosaic
    n.load_state_dict(weights, strict=strict)


def run(net, loader):
    net.eval()

    x, sorted_x, idx, label = map(lambda x: x.cuda(async=True), next(iter(loader)))
    with torch.no_grad():
        _ = net(x)
    return x


def extract_output(f, l):
    def wrapper( *args, **kwargs):
        y = f(*args, **kwargs)
        l.append(y)
        return y
    return wrapper

assignments = []
import permutation

permutation.sinkhorn = extract_output(permutation.sinkhorn, assignments)

torch.backends.cudnn.benchmark = True

x = run(net, test_loader)

if args.dataset == 'mnist':
    unnormalise = transforms.Compose([
        transforms.Normalize((0,), (1/0.3081,)),
        transforms.Normalize((-0.1307,), (1,)),
    ])
elif args.dataset == 'cifar10':
    unnormalise = transforms.Compose([
        transforms.Normalize((0, 0, 0), (1/0.2023, 1/0.1994, 1/0.2010)),
        transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
    ])
elif args.dataset == 'imagenet':
    unnormalise = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1]),
    ])

os.makedirs('imgs', exist_ok=True)
for i, assignment in enumerate(assignments):
    permuted = permutation.apply_assignment(x, assignment)
    imgs = net.assemble_mosaic(permuted)
    for j, img in enumerate(imgs[:8]):
        img = unnormalise(img)
        img = transforms.functional.to_pil_image(img.cpu())

        name = 'both' if 'both' in args.resume else 'optim' if 'optim' in args.resume else 'linear'
        task = 'mosaic' if 'mosaic' in args.resume else 'classify'
        img.save(f'imgs/{task}-{name}-{args.dataset}-{args.tiles_per_side}-{j}-step{i}.png')
