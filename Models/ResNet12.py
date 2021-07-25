import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F


__all__ = ["resnet12"]    # restrict that only resnet18 can be imported


class ConvBn(nn.Module):
    """
    卷积归一化，注意没有ReLU，需自行添加
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(1, 1), padding=(1, 1)):
        super().__init__()

        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        x = self.convbn(x)
        return x


class ResidualBlock1(nn.Module):
    """
    第一个卷积块，输入1*64*128，输出64*64*128，残差块对输入做Normalize处理
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1, self).__init__()
        self.conv = nn.Sequential(
            ConvBn(in_channels, out_channels, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            ConvBn(out_channels, out_channels, kernel_size=3, stride=(1, 1), padding=(1, 1))
        )
        self.normalize = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.conv(x)
        x = self.normalize(x)
        y += x    # 注意，这里加入残差

        return F.relu(y)


class NormalResidual(nn.Module):
    """
    后续的残差模块
    """
    def __init__(self, in_channels, out_channels):
        super(NormalResidual, self).__init__()
        self.conv = nn.Sequential(
            ConvBn(in_channels, out_channels, kernel_size=3, stride=(1, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            ConvBn(out_channels, out_channels, kernel_size=3, stride=(1, 1), padding=(1, 1))
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 2), padding=0)

    def forward(self, x):
        y = self.conv(x)
        x = self.res_conv(x)
        y += x  # 注意，这里加入残差

        return F.relu(y)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = ResidualBlock1(in_channels=1, out_channels=64)
        self.block2 = nn.Sequential(
            NormalResidual(in_channels=64, out_channels=64),
            NormalResidual(in_channels=64, out_channels=128),
            NormalResidual(in_channels=128, out_channels=256)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Para_ResNet(nn.Module):
    def __init__(self, num_classes=9):
        super(Para_ResNet, self).__init__()
        self.res = ResNet()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))
        )
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.res(x)    # resnet, output shape=(n, 256, 128, 16)
        x = self.conv(x)   # downsampling, output shape=(n, 9, 128, 8)
        x = self.max_pool(x)    # GMP replace with MLP
        x = x.view(-1, 9)    # flatten
        x = F.log_softmax(x, dim=1)
        return x


def resnet12(num_classes=9):
    return Para_ResNet(num_classes)


if __name__ == '__main__':
    model = resnet12()
    x = torch.randn([8, 1, 128, 128])

    y = model(x)
    print(y.shape)