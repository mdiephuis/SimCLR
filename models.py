import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Resnet, Copy-paste from torchvision.models.resnet,
# and https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# slightly modified for cifar10
#
# Resnet, Copy-paste from torchvision.models.resnet,
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer_block = nn.Sequential(
            self._make_layer(block, 64, layers[0], stride=1),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[2], stride=2)
        )

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.layer_block(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet18_attention():
    return ResNet(AttentionBlock, [1, 1, 1, 1])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
