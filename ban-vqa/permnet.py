import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import permutation
from permutation import Conv1


class PermNet(nn.Module):
    def __init__(self, objects, input_features, mid_features, output_features):
        super().__init__()
        self.objects = objects
        self.lr = nn.Parameter(torch.ones(1))
        mid_features += 5  # x1, y1, x2, y2, att
        self.lstm = nn.LSTM(mid_features, output_features)
        self.skew = permutation.PairwiseSkew(lambda: nn.Sequential(
            Conv1(2 * mid_features, 128),
            nn.ReLU(inplace=True),
            Conv1(128, 1),
        ))
        self.conv = Conv1(2048, mid_features - 5)
        self.drop = nn.Dropout(0.5)
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform(m.weight)
                m.bias.data.zero_()

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def forward(self, boxes, attention, features):
        boxes = boxes.contiguous()
        boxes = boxes / boxes.view(boxes.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(dim=-1)
        features = features.transpose(1, 2).contiguous()
        features = self.conv(features)
        f = features
        features = torch.cat([boxes, attention.unsqueeze(dim=1), features], dim=1)

        c = self.skew(features)
        a = permutation.calculate_assignment(c, lr=self.lr.abs(), temp=1, steps=3)
        features = torch.cat([boxes, attention.unsqueeze(dim=1), f], dim=1) * F.sigmoid(attention.unsqueeze(1))
        x = permutation.apply_assignment(features, a)

        x = x.permute(2, 0, 1).contiguous()
        _, (_, cell_state) = self.lstm(x)

        return cell_state.squeeze(0)
