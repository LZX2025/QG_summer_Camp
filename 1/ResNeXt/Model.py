import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, groups=1, width_per_group=64):
        super(ResNeXt, self).__init__()
        self.in_channel = 64
        self.num_classes = num_classes
        self.groups = groups
        self.width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, channel, num_blocks, stride=1, ):
        downsample = None
        if channel * block.expansion != self.in_channel or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample, groups=self.groups, width_per_group=self.width))
        self.in_channel = channel * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width))

        return nn.Sequential(*layers)


class Block(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Block, self).__init__()

        width = int(out_channels * (width_per_group / 64.)) * groups
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        res = x
        if self.downsample is not None:
            res = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += res
        out = self.relu(out)
        return out

def res_next50_32x4d(num_classes = 1000, **kwargs):
    model = ResNeXt(Block, [3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)
    return model

