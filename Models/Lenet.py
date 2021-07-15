# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

__all__ = ["lenet5"]  # restrict that only lenet can be imported


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


class Lenet(nn.Module):
    def __init__(self, num_classes=10):  # 针对CIFAR10，只有10个类别
        super(Lenet, self).__init__()

        self.conv1 = ConvBnRelu(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # nn.Conv2d(in_channels=3, out_channels=18, kernel_size=5, stride=1, padding=0)    # 28*28
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 14*14
        self.conv2 = ConvBnRelu(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # nn.Conv2d(in_channels=18, out_channels=48, kernel_size=5, stride=1, padding=0)  # 10*10
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 5*5
        self.fc1 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        # self.fc2 = nn.Linear(in_features=360, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=84)
        self.fc4 = nn.Linear(in_features=84, out_features=num_classes)
        # init params as we need
        self.init_params()

    def init_params(self):
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
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        return x


def lenet5(num_classes=10):
    return Lenet(num_classes=num_classes)


if __name__ == "__main__":
    model = Lenet()
    print(model)
    x = torch.randn([8, 3, 32, 32])
    y = model(x)

    print(y.shape)
