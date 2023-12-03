import torch
from torch import nn
from srgan import Discriminator

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))
    

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels=in_channels, 
            kernel_size=3, stride=1, padding=1, 
            bias=True
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))
    

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2) -> None:
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList([
            ConvBlock(
                in_channels + channels * i,  # recieves skip con from all the previous blocks
                channels if i <= 3 else in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_act=True if i <= 3 else False,
            ) for i in range(5)
        ])

    def forward(self, x):
        inputs = x

        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], dim=1)

        return self.residual_beta * out + x
    

class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2) -> None:
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[
            DenseResidualBlock(in_channels) for _ in range(3)
        ])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x
    

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23) -> None:
        super().__init__()
        self.initial = nn.Conv2d(
            in_channels,
            out_channels=num_channels,
            kernel_size=3, stride=1, padding=1,
        )
        self.residuals = nn.Sequential(*[
            RRDB(num_channels) for _ in range(num_blocks)
        ])
        self.conv = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels), 
            UpsampleBlock(num_channels),
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)
    

# Discriminator is the same as in SRGAN, so no need to implement


def initialize_weights(model: nn.Module, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


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
