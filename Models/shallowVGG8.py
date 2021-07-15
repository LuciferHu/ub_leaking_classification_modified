import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["shallow_vgg8"]     # restrict that only shallow_vgg8 can be imported


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
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


class BlockS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.convpool = nn.Sequential(
            ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        x = self.convpool(x)
        return x


class BlockD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
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


class VGG8(nn.Module):
    def __init__(self, num_classes=10):  # 针对UrbanSound8K，有10个类别
        super().__init__()

        self.block1 = BlockS(in_channels=1, out_channels=32)
        self.block2 = BlockS(in_channels=32, out_channels=64)
        self.block3 = BlockD(in_channels=64, out_channels=128)
        self.block4 = BlockS(in_channels=128, out_channels=128)
        self.fc1 = BlockFC(in_features=128*5*5, out_features=1024)
        self.fc2 = BlockFC(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(x.shape[0], -1)     # flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def shallow_vgg8(num_classes=10):
    return VGG8(num_classes)


if __name__ == '__main__':
    model = shallow_vgg8()
    print(model)
    x = torch.randn([8, 1, 128, 128])
    # print(model.state_dict())
    y = model(x)
    print(y.shape)