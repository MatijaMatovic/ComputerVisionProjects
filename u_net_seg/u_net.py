import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, channels=[64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of the net
        for n_channels in channels:
            self.downs.append(DoubleConv(in_channels, n_channels))
            in_channels = n_channels

        # Up part of the net
        for n_channels in reversed(channels):
            self.ups.append(
                nn.ConvTranspose2d(
                    n_channels*2, n_channels, 
                    kernel_size=2, stride=2
                )  # input is 2 * n_chan bekause of the skip conns, where we concat with the U-bridge
            )
            self.ups.append(
                DoubleConv(n_channels*2, n_channels)
            )

        self.bottom = DoubleConv(channels[-1], channels[-1]*2)

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        
        x = self.bottom(x)

        skip_connections = skip_connections[::-1]

        # because every second one is a double conv and it doesnt take the shortcut con
        for idx in range(0, len(self.ups), 2):  
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]  # because of the 2-step

            # if the input image is of shape like 181
            # the max pool layer will reduce it to 90
            # i.e. it will divide by 2 and floor the dim.
            # When we upsample with transpose conv
            # it will be increased to 90*2 = 180 != 181
            # then, concatenation will not work
            # Thus, an extra row and column should be added
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)



def test():
    batch_size, color_channels, img_size, num_classes = 3, 1, 160, 1

    x = torch.randn((
        batch_size,
        color_channels,
        img_size,
        img_size
    ))

    model = UNET(in_channels=color_channels, out_channels=num_classes)

    y = model(x)
    assert y.shape == (batch_size, color_channels, img_size, img_size)
    print(y.shape)


if __name__ == '__main__':
    test()