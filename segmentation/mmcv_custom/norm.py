# -*- coding: utf-8 -*-

import torch
from torch import nn
from mmcv.cnn.bricks.registry import NORM_LAYERS


class CustomBatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.bn = nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.gamma = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.float()
        x = self.bn(x)
        x = x.half() * self.gamma[:, None, None] + self.beta[:, None, None]
        return x


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm, eps=1e-5):
        super().__init__()
        self.ln = norm_layer(normalized_shape, eps=eps)

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


NORM_LAYERS.register_module(name='LN2D', module=LayerNorm2D)
NORM_LAYERS.register_module('CustomBN2D', module=CustomBatchNorm2D)
