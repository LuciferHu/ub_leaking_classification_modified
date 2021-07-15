# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["vgg11"]  # restrict that only vgg16 can be imported


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


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.convpool = nn.Sequential(
            ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding),
            ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        x = self.convpool(x)
        return x


class BlockFC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.fc(x)

        return x


class VGG11(nn.Module):
    def __init__(self, num_classes=1000):    # 针对ImageNet，有1000个类别
        super().__init__()

        self.block1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.block2_1 = Block(in_channels=128, out_channels=256)
        self.block2_2 = Block(in_channels=256, out_channels=512)
        self.block2_3 = Block(in_channels=512, out_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.block_fc1 = BlockFC(in_features=512*4*4, out_features=4096)
        self.block_fc2 = BlockFC(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        # init params as we need
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)    # flatten
        x = self.block_fc1(x)
        x = self.block_fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def vgg11(num_classes=1000):
    return VGG11(num_classes=num_classes)


if __name__ == '__main__':
    model = vgg11(8)
    x = torch.randn([8, 1, 128, 128])

    y = model(x)
    print(y.shape)