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
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        up_mode='upconv',
        dropout=None,
        activ_func=None,
        normalizer=None
    ):
        """
        Implementation of
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
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.n_classes = n_classes
        self.activ_func = activ_func
        self.normalizer = normalizer

        prev_channels = in_channels
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, dropout, activ_func, normalizer)
            )
            prev_channels = 2 ** (wf + i)

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, dropout, activ_func, normalizer)
            )
            prev_channels = 2 ** (wf + i)

        # output segmentation map (convert negative values to zero)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        blocks = []
        # downsample path
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        # upsample path
        for i, up in enumerate(self.decoder):
            x = up(x, blocks[-i - 1])


        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, dropout, activ_func, normalizer):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(normalizer.evaluate(out_size))
        block.append(activ_func)

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(normalizer.evaluate(out_size))
        block.append(activ_func)
        if dropout:
            block.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, dropout, activ_func, normalizer):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, dropout, activ_func, normalizer)

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
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out
