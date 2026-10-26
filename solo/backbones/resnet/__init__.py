# Copyright 2023 solo-learn development team.
import copy
from typing import Union, Optional, Callable

import torch
from torch import nn, Tensor
from torch.nn import Flatten
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, resnet101
from torchvision.utils import _log_api_usage_once

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from .resnet import resnet18 as default_resnet18
from .resnet import resnet50 as default_resnet50


def add_sinusoidal_positional_encoding(x):
    B, C, H, W = x.shape
    device = x.device

    pe = torch.zeros((1, C, H, W), device=device)
    pos_h = torch.arange(H, dtype=torch.float, device=device).unsqueeze(1).repeat(1, W) / H
    pos_w = torch.arange(W, dtype=torch.float, device=device).repeat(H, 1) / W

    pos_h = pos_h.unsqueeze(0).unsqueeze(0)
    pos_w = pos_w.unsqueeze(0).unsqueeze(0)

    # scale to match number of channels
    pe[:, 0::2, :, :] = torch.sin(pos_h * 10000 ** (torch.arange(0, C, 2, device=device) / C).view(1, -1, 1, 1))
    pe[:, 1::2, :, :] = torch.cos(pos_w * 10000 ** (torch.arange(1, C, 2, device=device) / C).view(1, -1, 1, 1))

    return x + pe

class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()
        h, w = 224, 224

        y = torch.linspace(-1, 1, steps=h, device="cpu")
        x = torch.linspace(-1, 1, steps=w, device="cpu")

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)



    def forward(self, x):
        grid = self.grid.to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # (B, 2, H, W)
        return torch.cat((x, grid), dim=1)

def resnet18(method, *args, **kwargs):
    return default_resnet18(*args, **kwargs)


def resnet50(method, *args, **kwargs):
    return default_resnet50(*args, **kwargs)

def resnet100(method, *args, **kwargs):
    return resnet101(*args, **kwargs)

def resnet50_dorsal(method, *args, **kwargs):
    class DorsalResNet(ResNet):
        def __init__(self, *args, sum=False, **kwargs):
            super().__init__(*args, **kwargs)
            # self.add_module("layer5", copy.deepcopy(self.layer4))
            self.layer5 = copy.deepcopy(self.layer4)
            # self.layer5 = nn.Sequential(*[copy.deepcopy(block) for block in self.layer4])
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.inplanes = 4096 if not sum else 2048
            self.sum = sum
            self.flatten = Flatten()

        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x1 = self.layer4(x)
            x1 = self.avgpool(x1)
            x2 = self.layer5(x)
            x2 = self.avgpool2(x2)

            return self.flatten(torch.cat((x1,x2), dim=1)) if not self.sum else self.flatten(x1 + x2)

        def forward(self, x):
            return self._forward_impl(x)

    from torchvision.models.resnet import Bottleneck
    return DorsalResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)


def resnet50_dorsal2(method, *args, **kwargs):
    class DorsalResNet(ResNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layer5 = copy.deepcopy(self.layer3)
            self.layer6 = copy.deepcopy(self.layer4)
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.inplanes = 4096
            self.flatten = Flatten()

        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x1 = self.layer3(x)
            x1 = self.layer4(x1)
            x1 = self.avgpool(x1)
            x2 = self.layer5(x)
            x2 = self.layer6(x2)
            x2 = self.avgpool2(x2)

            return self.flatten(torch.cat((x1,x2), dim=1))

        def forward(self, x):
            return self._forward_impl(x)

    from torchvision.models.resnet import Bottleneck
    return DorsalResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)

def resnet50pe(method, *args, **kwargs):
    model = default_resnet50(*args, **kwargs)

    def fn(module, __, output):
        return add_sinusoidal_positional_encoding(output)
    model.maxpool.register_forward_hook(fn)
    return model

def resnet50p(method, *args, **kwargs):
    model = default_resnet50(*args, **kwargs)
    model.add_module("add_coords", AddCoords())
    model.conv1 = nn.Conv2d(3 + 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def reg(model):
        def fn(module, inputs):
            output = model.add_coords(inputs[0])
            return output
        return fn
    model.conv1.register_forward_pre_hook(reg(model))
    return model

def resnet50_proj(method, *args, **kwargs):
    class ProjResNet(ResNet):
        def __init__(self, *args, num_features=4096, **kwargs):
            super().__init__(*args, **kwargs)
            self.proj = nn.Sequential(nn.Linear(2048, num_features), nn.BatchNorm1d(num_features) ,nn.ReLU())
            self.num_features = num_features
            self.inplanes = num_features

        def _forward_impl(self, x):
            return self.proj(super()._forward_impl(x))

        def forward(self, x):
            return self._forward_impl(x)

    from torchvision.models.resnet import Bottleneck
    return ProjResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)


__all__ = ["resnet18", "resnet50", "resnet50pe"]
