# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

__all__ = ["alexnet"]    # restrict that only vgg16 can be imported


class ConvRelu(nn.Module):    # 卷积后Relu操作
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvRelu, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convrelu(x)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convbnrelu(x)
        return x


class Alexnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Alexnet, self).__init__()

        self.conv1 = ConvBnRelu(in_channels=1, out_channels=96, kernel_size=11,
                              padding=1, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = ConvBnRelu(in_channels=96, out_channels=256, kernel_size=5,
                              padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = ConvBnRelu(in_channels=256, out_channels=384)
        self.conv4 = ConvBnRelu(in_channels=384, out_channels=384)
        self.conv5 = ConvBnRelu(in_channels=384, out_channels=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=6400, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def alexnet(num_classes=10):
    return Alexnet(num_classes=num_classes)


if __name__ == "__main__":
    model = alexnet(num_classes=8)
    print(model)
    x = torch.randn([1, 3, 128, 128])

    y = model(x)

    print(y.shape)
