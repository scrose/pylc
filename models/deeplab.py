# Adapted from https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/models/deeplab.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.utils.aspp import build_aspp
from models.backbone import resnet
from models.decoder import build_decoder

class DeepLab(nn.Module):
    def __init__(self, activ_func, normalizer, backbone='resnet', output_stride=16, n_classes=11, freeze_bn=False):
        super(DeepLab, self).__init__()

        self.backbone = resnet.ResNet101(output_stride, normalizer)
        self.aspp = build_aspp(backbone, output_stride, normalizer)
        self.decoder = build_decoder(n_classes, backbone, normalizer)
        self.normalizer = normalizer
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, self.normalizer):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], self.normalizer):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], self.normalizer):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
