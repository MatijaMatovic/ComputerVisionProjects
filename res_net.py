import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
       super(Block, self).__init__()
       self.expansion = 4  # n of chanels is alway 4x what entered
       self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
       self.bn1 = nn.BatchNorm2d(out_channels)
       self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
       self.bn2 = nn.BatchNorm2d(out_channels)
       self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
       self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
       self.relu = nn.ReLU()
       self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # as the input has higher dimensions and less channels than the ouptut
        # it is necessary to just adapt its size by a single conv layer
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # implementing the shortcut
        x += identity
        x = self.relu(x)
        return x
    

class ResNet(nn.Module):

    '''
    Args:
    
        block_counts - how many ResNet blocks there will be per layer
        image_channels - how many channels the image has. 3 for RGB, 1 for grayscale
    '''
    def __init__(self, n_res_blocks: list[int], image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layer1 = self._make_layer(n_res_blocks[0], out_channels=64, stride=1)
        self.res_layer2 = self._make_layer(n_res_blocks[1], out_channels=128, stride=2)
        self.res_layer3 = self._make_layer(n_res_blocks[2], out_channels=256, stride=2)
        self.res_layer4 = self._make_layer(n_res_blocks[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512*4, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)

        return x

    
    def _make_layer(self, blocks_count: int, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != None or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4)
            )

        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
    
        for _ in range(blocks_count-1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=10):
    return ResNet([3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=10):
    return ResNet([3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=10):
    return ResNet([3, 8, 36, 3], img_channels, num_classes)


def test():
    batch_size = 2
    img_channels = 3
    num_classes=10

    net = ResNet50(num_classes=num_classes)

    x = torch.randn(batch_size, img_channels, 224, 224)
    y = net(x)

    assert y.shape == (batch_size, num_classes)

if __name__ == '__main__':
    test()