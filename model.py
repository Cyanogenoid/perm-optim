import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

import permutation
from permutation import Conv1


class MosaicNet(nn.Module):
    def __init__(self, tiles_per_side, input_channels, tile_size, conv_channels=32, steps=4, temp=1, uniform_init=False, avoid_nans=False):
        super().__init__()
        self.input_channels = input_channels
        self.tiles_per_side = tiles_per_side
        self.tile_size = tile_size
        self.uniform_init = uniform_init
        self.lr = nn.Parameter(torch.ones(1))
        self.steps = steps
        self.temp = temp
        self.avoid_nans = avoid_nans
        n = conv_channels * (self.tile_size // 2)**2
        self.compare = permutation.Comparator(nn.Sequential(
            Conv1(2*n, 64),
            nn.ReLU(inplace=True),
            Conv1(64, 2, bias=False),
        ))
        self.coord = permutation.LinearAssign(nn.Sequential(
            Conv1(n, tiles_per_side ** 2, bias=False),
        ))
        self.conv_stack = nn.Sequential(
            nn.Conv3d(self.input_channels, conv_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_img = x
        x = self.conv_stack(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # put sequence dim last
        x = x.view(x.size(0), -1, x.size(-1))

        c = self.compare(x)
        pos_cost = self.coord(x) if not self.uniform_init else None
        if pos_cost is not None and self.avoid_nans:
            pos_cost = pos_cost.clamp(min=-10, max=10)  # [4e-5, 22026] should be plenty for anything sensible

        a = permutation.calculate_assignment(
            cost_matrix=c,
            assignment=pos_cost,
            lr=self.lr.abs(),
            temp=self.temp,
            steps=self.steps,
            size_2d=(self.tiles_per_side, self.tiles_per_side)
        )
        x_img = permutation.apply_assignment(x_img, a)

        return x_img, a.squeeze(1), None


class PermNet(nn.Module):
    def __init__(self, steps=8, temp=1):
        super().__init__()
        self.steps = steps
        self.temp = temp
        self.lr = nn.Parameter(torch.ones(1))
        comp_head = nn.Sequential(
        )
        self.compare = permutation.Comparator(nn.Sequential(
            Conv1(2, 16),
            nn.ReLU(inplace=True),
            Conv1(16, 1),
        ))

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        c = self.compare(x)
        a = permutation.calculate_assignment(c, lr=self.lr.abs(), steps=self.steps, temp=self.temp)
        x = permutation.apply_assignment(x, a)
        return x, a.squeeze(1), None


class ClassifyNet(nn.Module):
    def __init__(self, tiles_per_side, input_channels, **kwargs):
        super().__init__()
        self.tiles_per_side = tiles_per_side
        self.mosaic = MosaicNet(tiles_per_side, input_channels, **kwargs)
        self.model = torchvision.models.resnet18(num_classes=10)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)  # smaller kernel size and no striding
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')  # same init as original model

    def init_imagenet(self):
        self.model = torchvision.models.resnet18(num_classes=1000)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def lock_residual_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_in = x

        reconstruction, assignment, _ = self.mosaic(x)
        if not self.no_permute:
            x = reconstruction
        else:
            reconstruction = x
        x = self.assemble_mosaic(x)

        x = self.model(x)

        return reconstruction, assignment, x

    def assemble_mosaic(self, x):
        x = x.view(x.size(0), x.size(1), self.tiles_per_side, self.tiles_per_side, x.size(-2), x.size(-1))
        x = x.permute(2, 3, 0, 1, 4, 5).contiguous()  # move sequence dims to the front
        x = torch.cat(tuple(x), dim=-1)
        x = torch.cat(tuple(x), dim=-2)
        return x

    @property
    def lr(self):
        return self.mosaic.lr
