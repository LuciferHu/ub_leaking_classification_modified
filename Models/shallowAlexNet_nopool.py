import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["shallow_alexnet_nopool"]     # restrict that only shallow_vgg8 can be imported


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


class Shallow_AlexNet_NoPool(nn.Module):
    def __init__(self, num_classes):
        super(Shallow_AlexNet_NoPool, self).__init__()
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=24, kernel_size=5)

        self.conv2 = ConvBnRelu(in_channels=24, out_channels=36, kernel_size=4)

        self.conv3 = ConvBnRelu(in_channels=36, out_channels=48, kernel_size=3)

        self.fc1 = BlockFC(in_features=48, out_features=60)

        self.fc2 = nn.Linear(in_features=60, out_features=num_classes)
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
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.shape[0], -1)  # flatten
        x = self.fc1(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x


def shallow_alexnet_nopool(num_classes=9):
    return Shallow_AlexNet_NoPool(num_classes)


if __name__ == '__main__':
    model = shallow_alexnet_nopool()
    print(model)
    x = torch.randn([8, 1, 128, 128])
    # print(model.state_dict())
    y = model(x)
    print(y.shape)