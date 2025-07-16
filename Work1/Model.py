
import torch
import torch.nn as nn

class ResNeXt(nn.Module):
    def __init__(self, block, num_block,input_channels=3, num_classes=1000, groups=1, width_per_group=64):
        super(ResNeXt, self).__init__()
        self.in_channel = 64
        self.num_classes = num_classes
        self.groups = groups
        self.width_per_group = width_per_group


        self.conv1 = nn.Conv2d(input_channels, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64 , num_block[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_block[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_block[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, channel, num_blocks, stride=1, ):
        downsample = None
        layers = []
        expansion = block.expansion
        if self.in_channel != channel * expansion or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * expansion),
            )
        layers.append(block(self.in_channel, channel, stride=stride, groups=self.groups, width_per_group=self.width_per_group, downsample=downsample))
        self.in_channel = channel * expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)


class Block(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, groups=1, width_per_group=64, downsample=None):
        super(Block, self).__init__()
        self.expansion = 4
        self.groups = groups
        self.width_per_group = width_per_group
        self.width = int(out_channel * (self.width_per_group / 64.)) * groups


        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.conv3 = nn.Conv2d(in_channels=self.width, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        res = x
        if self.downsample is not None:
            res = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += res
        x = self.relu(x)
        return x


def make_model(num_classes=1000, input_channel=3, groups=32, width_per_group=4, **kwargs):
    model = ResNeXt(block=Block, input_channels=input_channel, num_block=[3, 4, 6, 3], num_classes=num_classes, groups=groups,
                    width_per_group=width_per_group)
    return model


