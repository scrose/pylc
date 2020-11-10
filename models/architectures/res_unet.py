"""
Adapted from https://github.com/xiaochengcike/pytorch-unet-1

PyTorch implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation
(Ronneberger et al., 2015). This implementation has many tweakable options such as:

    Depth of the network
    Number of filters per layer
    Transposed convolutions vs. bilinear upsampling
    valid convolutions vs padding
    batch normalization
"""

import torch
from torch import nn


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=True,
        up_mode='upconv',
        dropout=0.3,
        activ_func=None
    ):
        """
        Residual adaptation of implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(ResUNet, self).__init__()

        assert up_mode in ('upconv', 'upsample')

        self.padding = padding
        self.depth = depth
        self.activation = activ_func if activ_func else nn.functional.relu(inplace=True)

        prev_channels = in_channels

        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(
                UNetResBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, dropout, self.activation)
            )
            prev_channels = 2 ** (wf + i)

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, dropout, self.activation)
            )
            prev_channels = 2 ** (wf + i)

        # output segmentation map (convert negative values to zero)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='ReLU')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        blocks = []
        # downsample path
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder) - 1:
                blocks.append(x)
                x = nn.functional.max_pool2d(x, 2)
        # upsample path
        for i, up in enumerate(self.decoder):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm, dropout, activ_func):
        super(UNetResBlock, self).__init__()
        self.in_channels, self.out_channels, self.activate = in_channels, out_channels, activ_func
        self.block = nn.Identity()
        self.shortcut = nn.Identity()

        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(activ_func)
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        block.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*block)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        residual = self.center_crop(residual, x.shape[2:])
        print(x.shape, residual.shape)
        x += residual
        x = self.activate(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, padding, batch_norm, dropout, activ_func):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

        self.conv_block = UNetResBlock(in_channels, out_channels, padding, batch_norm, dropout, activ_func)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        x = torch.cat([up, crop1], 1)
        x = self.conv_block(x)
        return x
