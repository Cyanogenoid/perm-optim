import os
import argparse
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.optimize
import numpy as np
from tqdm import tqdm

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
parser.add_argument('--num-workers', type=int, default=4, help='Number of threads for data loader')
parser.add_argument('--no-cuda', action='store_true', help='Run on CPU instead of GPU (not recommended)')
parser.add_argument('--train-only', action='store_true', help='Only run training, no evaluation')
parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, no training')
parser.add_argument('--input-sorted', action='store_true', help='Input the correctly sorted sequence instead of random order')
parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs')
# sort
sort_parser = subparser.add_parser('sort')
sort_parser.add_argument('--length', type=int, default=10, help='How many numbers to sort')
sort_parser.add_argument('--low', type=float, default=0, help='Low end of interval numbers are sampled from')
sort_parser.add_argument('--high', type=float, default=1, help='High end of interval numbers are sampled from')
sort_parser.add_argument('--size-pow', type=int, default=14, help='Training set has 2**n size')
sort_parser.add_argument('--size-pow-test', type=int, help='Test set has 2**n size, defaults to same as training set size')
sort_parser.add_argument('--double', action='store_true', help='Use f64 instead of f32 to avoid numerical issues with dataset')
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

optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)

train_loader = data.get_loader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)
test_loader = data.get_loader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

tracker = track.Tracker(
    train_loss=track.ExpMean(),
    train_acc=track.ExpMean(),
    train_l1=track.ExpMean(),
    train_l2=track.ExpMean(),
    train_lr=track.Identity(),

    test_loss=track.Mean(),
    test_acc=track.Mean(),
    test_l1=track.Mean(),
    test_l2=track.Mean(),
    test_lr=track.Identity(),
)

if args.resume:
    log = torch.load(args.resume)
    weights = log['weights']
    n = net
    strict = True
    if args.multi_gpu:
        n = n.module
    if args.task == 'classify':
        # we only want to load the classifier portion of the model, not the
        # mosaic portion because it changes with different tile size
        weights = {k: v for k, v in weights.items() if k.startswith('model')}
        strict = False
    n.load_state_dict(weights, strict=strict)


def permutation_acc(assignment, idx):
    """ Checks whether the assignment produced by Hungarian algorithm fully matches the given assignment
    """
    assignment = -assignment.transpose(1, 2).detach().cpu().numpy()
    correct = 0
    total = 0
    hard_assignments = []
    for matrix, target in zip(assignment, idx.cpu().numpy()):
        _, col_idx = scipy.optimize.linear_sum_assignment(matrix)
        hard_assignments.append(col_idx)
        if np.array_equal(col_idx, target):
            correct += 1
        total += 1
    acc = correct / total
    return acc, hard_assignments

def apply_hard_assignment(x, idx):
    permuted = [sample[:, torch.from_numpy(i)] for sample, i in zip(x, idx)]
    return torch.stack(permuted)

def run(net, loader, optimizer, train=False, epoch=0):
    if train:
        net.train()
        prefix = 'train'
    else:
        net.eval()
        prefix = 'test'

    loader = tqdm(loader, ncols=0, desc='{1} E{0:02d}'.format(epoch, 'train' if train else 'test '))
    for sample in loader:
        x, sorted_x, idx, label = map(lambda x: x.cuda(async=True), sample)

        if args.input_sorted:
            x = sorted_x

        reconstruction, assignment, pred = net(x)

        if not train and not args.no_hard_assign:
            acc, hard_assignments = permutation_acc(assignment, idx)
            reconstruction = apply_hard_assignment(x, hard_assignments)
        else:
            acc = 0

        l2 = (reconstruction - sorted_x).pow(2).mean()
        l1 = (reconstruction - sorted_x).abs().mean()

        if not args.task == 'classify':
            loss = l2
        else:
            loss = torch.nn.functional.cross_entropy(pred, label)
            pred_class = pred.max(1, keepdim=True)[1]
            correct = pred_class.eq(label.view_as(pred_class)).sum().item()
            acc = correct / pred.size(0)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tracked_loss = tracker.update('{}_loss'.format(prefix), loss.item())
        tracked_acc = tracker.update('{}_acc'.format(prefix), acc)
        tracked_l1 = tracker.update('{}_l1'.format(prefix), l1.item())
        tracked_l2 = tracker.update('{}_l2'.format(prefix), l2.item())
        tracked_lr = tracker.update('{}_lr'.format(prefix), net.lr.item())

        fmt = '{:.5f}'.format
        loader.set_postfix(
            loss=fmt(tracked_loss),
            acc=fmt(tracked_acc),
            l1=fmt(tracked_l1),
            l2=fmt(tracked_l2),
            lr=tracked_lr,
        )


import subprocess
git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])

torch.backends.cudnn.benchmark = True

for epoch in range(args.epochs):
    tracker.new_epoch()
    if not args.eval_only:
        run(net, train_loader, optimizer, train=True, epoch=epoch)
    if not args.train_only:
        run(net, test_loader, optimizer, train=False, epoch=epoch)

    results = {
        'name': args.name,
        'tracker': tracker.data,
        'weights': net.state_dict() if not args.multi_gpu else net.module.state_dict(),
        'args': vars(args),
        'hash': git_hash,
    }
    torch.save(results, os.path.join('logs', args.name))
    if args.eval_only:
        break
