import torch
import torch.nn as nn

from .helper_3d import GroupNorm3D, NonLocalBlock3D, ResidualBlock3D, UpSampleBlock3D


class Decoder3D(nn.Module):
    def __init__(self, args):
        super(Decoder3D, self).__init__()
        ch_mult = [64, 128, 256, 256, 512]
        num_resolutions = len(ch_mult)
        block_in = ch_mult[num_resolutions - 1]

        layers = [
            nn.Conv3d(args.latent_dim, block_in, kernel_size=3, stride=1, padding=1),
            ResidualBlock3D(block_in, block_in),
            NonLocalBlock3D(block_in),
            ResidualBlock3D(block_in, block_in),
        ]

        for i in reversed(range(num_resolutions)):
            block_out = ch_mult[i]
            for _ in range(3):
                layers.append(ResidualBlock3D(block_in, block_out))
                block_in = block_out
            if i > 1:
                layers.append(UpSampleBlock3D(block_in))

        layers.append(GroupNorm3D(block_in))
        layers.append(nn.Conv3d(block_in, args.image_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Args:
    latent_dim = 256
    image_channels = 1


if __name__ == "__main__":
    args = Args()
    args.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    model = Decoder3D(args).to(args.device)
    dummy_input = torch.randn(2, args.latent_dim, 7, 8, 7).to(args.device)

    with torch.no_grad():
        output = model(dummy_input)

    print("output shape:", output.shape)
