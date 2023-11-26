import torch
import torch.nn as nn
from math import ceil

# tuples of (expand_ratio, channels, repeats, stride, kernel_size) for each block
base_model = [
    (1, 16,  1, 1, 3),
    (6, 24,  2, 2, 3),
    (6, 40,  2, 2, 5),
    (6, 80,  3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


# tuples of (phi_value, resolution, drop_rate)
# drop rate - each layer in the block has some prob
# to get dropped - similar to dropout, but for layers,
# instead of neurons
phi_values = {
    'b0': (0,   224, 0.2),
    'b1': (0.5, 240, 0.2),
    'b2': (1,   260, 0.3),
    'b3': (2,   300, 0.3),
    'b4': (3,   300, 0.3),
    'b5': (4,   456, 0.4),
    'b6': (5,   528, 0.5),
    'b7': (6,   600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=groups  # for depth-wise convolution
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


'''
First, we use an Adaptive Average Pool Layer
It turns a layer Li of (WxHxC) into (1x1xC),
by calculating the average of each channel
Then it passes it through a 1x1xC conv layer
And then it uses a sigmoid to get an output
Ai which is 1x1xC of [0,1] values
The input to the layer Li+1 is the output of Li
multiplied by Ai
This can be looked as learning the importance of
each channel in the layer
'''
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduce_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # HxWxC -> 1x1xC
            nn.Conv2d(in_channels, reduce_dim, 1), # bottleneck
            nn.SiLU(),
            nn.Conv2d(reduce_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

'''
Depth convolution as in MobileNet
The kernel has n 2d layers, which must be the same as the
depth of its input layer. Then each layer convolves over each
respective layer of the input.
This differs from the standard convolution where each kernel
convolves over all input layers, and then adds them up
'''
class InvertedResidualBlock(nn.Module):
    def __init__(
            self, 
            in_channels, out_channels, 
            kernel_size, stride, padding, 
            expand_ratio,
            reduction=4,  # for squeeze excitation
            survival_prob=0.8,  # for layer dropout - stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        reduce_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduce_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        dropout_mask = torch.rand(x.shape[0], 1, 1, 1) < self.survival_prob

        # divide by dropout prob, to accomodate for the mean and variance shift caused by dropout
        return torch.div(x, self.survival_prob) * dropout_mask
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        
        return self.conv(x)
        


class EfficientMobileNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientMobileNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self._calculate_factors(version)
        last_channels = ceil(1280*width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self._create_features(width_factor, depth_factor, last_channels)
        self.flat = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    '''
    Args:
        alpha - depth scaling factor
        beta - width scaling factor

        According to the paper, there are such alpha, beta and gamma, each responsible for depth, width and resolution scaling, such that a*b*g = 2, and if the depth, widht and res are scaled by alpha^phi, beta^phi and gamma^phi, the computational need will increase at most by 2^phi
    '''
    def _calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def _create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [
            CNNBlock(
                in_channels=3, out_channels=channels,
                kernel_size=3, stride=2, padding=1)
        ]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels*width_factor) / 4)  # so it is divisible by 4 for SE layer
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2  # so we have 'same' convolutions
                    )
                )

                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=1)
        )

        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(self.flat(x))
    

def test():
    version = 'b0'
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res))
    model = EfficientMobileNet(
        version=version,
        num_classes=num_classes
    )

    assert model(x).shape == (num_examples, num_classes)
    print(model(x).shape)

test()
