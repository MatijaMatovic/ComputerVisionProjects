import torch
from torch import nn

'''
Conv -> BN -> Leaky/PReLU
'''
class ConvBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, 
            discriminator=False, 
            use_act=True, use_bn=True, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.use_act = use_act
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if discriminator else
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        x = self.bn(self.cnn(x))
        if self.use_act:
            return self.act(x)
        else:
            return x



class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels*scale_factor ** 2, 
            kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale_factor)  # (in_ch * 4, H, W) -> (in_ch, H*2, W*2)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        params = {
            'in_channels': in_channels,
            'out_channels': in_channels,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
        self.block1 = ConvBlock(
            **params
        )
        self.block2 = ConvBlock(
            use_act=False,
            **params
        )

    def forward(self, x):
        inputs = x
        x = self.block1(x)
        x = self.block2(x)
        x = x + inputs
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(
            in_channels, num_channels, 
            kernel_size=9, 
            stride=1, 
            padding=4,  # same padding
            use_bn=False
        )
        self.residuals = nn.Sequential(*[
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        self.conv = ConvBlock(
            num_channels, num_channels, 
            kernel_size=3, stride=1, padding=1,
            use_act=False
        )
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2)
        )
        self.final = nn.Conv2d(
            in_channels=num_channels, 
            out_channels=in_channels,
            kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.conv(x) + initial
        x = self.upsample(x)
        return torch.tanh(self.final(x))


DISCRIMINATOR_LAYER_FILTERS = [64, 64, 128, 128, 256, 256, 512, 512]

class Discriminator(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        blocks = []
        for idx, filters in enumerate(DISCRIMINATOR_LAYER_FILTERS):
            block = ConvBlock(
                in_channels,
                out_channels=filters,
                discriminator=True,
                kernel_size=3,
                stride=1 + idx % 2,  # 1, 2, 1, 2, 1, ...
                padding=1,
                use_act=True,
                use_bn=False if idx == 0 else True
            )
            blocks.append(block)
            in_channels = filters

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # if the image has greater resolution than the original 96x96
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
    
def test():
    low_resolution = 24
    x = torch.randn((5, 3, low_resolution, low_resolution))

    gen = Generator()
    gen_out = gen(x)
    disc = Discriminator()
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)

    assert gen_out.shape[-2:] == (low_resolution*4, low_resolution*4)
    assert(disc_out.shape[-1] == 1)

if __name__ == '__main__':
    test()