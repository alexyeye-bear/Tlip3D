import torch
import torch.nn as nn

from .helper_3d import DownSampleBlock3D, GroupNorm3D, ResidualBlock3D, Swish


class Encoder3D(nn.Module):
    def __init__(self, args):
        super(Encoder3D, self).__init__()
        channels = [32, 64, 128, 256, 256, 512]
        num_res_blocks = 2
        layers = [nn.Conv3d(args.image_channels, channels[0], 3, 1, 1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock3D(in_channels, out_channels))
                in_channels = out_channels
            if (i != 2) and (i != 3):
                layers.append(DownSampleBlock3D(channels[i + 1]))

        layers.append(ResidualBlock3D(channels[-1], channels[-1]))
        layers.append(ResidualBlock3D(channels[-1], channels[-1]))
        layers.append(GroupNorm3D(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], args.latent_dim, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Args:
    image_channels = 1
    latent_dim = 256


if __name__ == "__main__":
    args = Args()
    model = Encoder3D(args)
    x = torch.randn(2, args.image_channels, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    print("input shape:", x.shape)
    print("output shape:", out.shape)
