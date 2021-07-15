# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F


__all__ = ["resnet18"]    # restrict that only resnet18 can be imported


class Residual(nn.Module):
    """
    基本的残差网络块：
    X --> conv1 --> BN --> ReLU --> conv2 --> BN -->   +  --> ReLU --> Y
    |                                                  |
    -------------------- 1x1 conv3 --------------------
    """
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               stride=strides, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:    # 引入残差
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X    # 这里注意，原地改变了Y，与C style不同
        return F.relu(Y)


def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    """
    这里生成一个resnet网络块
    :param input_channels: 输入通道数
    :param output_channels: 输出通道数
    :param num_residuals: 重复几个基本的残差基本模块
    :param first_block: 第一个基本块与后续的稍有不同，这是控制参数
    :return: blk: 残差模块
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:    # 对于第一个块中的第一基本块
            blk.append(Residual(in_channels=input_channels, out_channels=output_channels,
                                use_1x1conv=True, strides=2))
        else:    # 对于不是第一个模块以及，其他基本模块
            blk.append(Residual(output_channels, output_channels))

    return blk    # 注意这是一个list


def resnet18(num_classes=10):
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),
                        nn.Linear(512, num_classes))

    return net


if __name__ == "__main__":
    model = resnet18()
    print(model)
    X = torch.randn([8, 3, 224, 224])

    y = model(X)
    print(y.shape)