'''
From https://github.com/abman23/PMNet/tree/PMNet , we have been using the version from 4.7.2023 (has been updated later after running our experiments, )
Lee et al - PMNet: Robust Pathloss Map Prediction via Supervised Learning, December 2023, Proceedings of IEEE Global Communicaions Conference (GLOBECOM)
We have added a few options (e.g. varying number of in_ch) and fixed a msitake in _stem, otherwise the model is the same as proposed by the autors.

License:
MIT License

Copyright (c) 2023 Juhyung Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
from ..dcn import DeformableConv2d

_BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

# Conv, Batchnorm, Relu layers, basic building block.
class _ConvBnReLU(nn.Sequential):

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True, bn_eps : float = 1e-05, dcn=False
    ):
        super(_ConvBnReLU, self).__init__()
        if dcn and dilation > 1:
            self.add_module(
                "conv",
                DeformableConv2d(
                    in_ch, out_ch, kernel_size, stride, kernel_size//2, bias=False
                ),
            )
        else:
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
                ),
            )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=bn_eps, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

# Bottleneck layer cinstructed from ConvBnRelu layer block, buiding block for Res layers
class _Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, stride, dilation, downsample, bn_eps : float = 1e-05, dcn=False):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True, bn_eps=bn_eps)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True, bn_eps=bn_eps, dcn=dcn)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False, bn_eps=bn_eps)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False, bn_eps=bn_eps)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

# Res Layer used to costruct the encoder
class _ResLayer(nn.Sequential):

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None, bn_eps=1e-5, dcn=False):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                    bn_eps=bn_eps,
                    dcn=dcn
                ),
            )

# Stem layer is the initial interfacing layer
class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch, in_ch = 2, ceil_mode=True, bn_eps=1e-5):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(in_ch, out_ch, 7, 2, 3, 1, bn_eps=bn_eps))
        '''First argument for MaxPool2d in the original implementation is in_ch, which should be a mistake (we have up to 15 in channels in some configurations...). 
        We set it to 2 instead, since with the dataset used by the authors of PMNet, this should be the usual value of in_ch.'''
        self.add_module("pool", nn.MaxPool2d(2, 2, 0, ceil_mode=ceil_mode))

class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch, bn_eps=1e-5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1, bn_eps=bn_eps)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h

# Atrous spatial pyramid pooling
class _ASPP(nn.Module):

    def __init__(self, in_ch, out_ch, rates, bn_eps=1e-5, dcn=False):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1, bn_eps=bn_eps))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bn_eps=bn_eps, dcn=dcn),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

# Decoder layer constricted using these 2 blocks
def ConRu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True)
    )

def ConRuT(in_channels, out_channels, kernel, padding, output_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding, output_padding=output_padding),
        nn.ReLU(inplace=True)
    )

class PMNet(nn.Module):

    def __init__(
            self, 
            in_ch : int, 
            n_blocks : list,
            atrous_rates : list,
            multi_grids : list,
            output_stride : int,
            ceil_mode : bool = True, 
            output_padding=(0, 0),
            bn_eps : float = 1e-05,
            dcn = False
            ):

        super(PMNet, self).__init__()

        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        else: 
            raise ValueError(f'output_stride={output_stride}, but only 8, 16 allowed')

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0], in_ch=in_ch, ceil_mode=ceil_mode, bn_eps=bn_eps)
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0], bn_eps=bn_eps)
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1], bn_eps=bn_eps)
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[3], s[2], d[2], bn_eps=bn_eps)
        self.layer5 = _ResLayer(n_blocks[3], ch[3], ch[4], s[3], d[3], multi_grids, bn_eps=bn_eps, dcn=dcn)
        self.aspp = _ASPP(ch[4], 256, atrous_rates, bn_eps=bn_eps, dcn=dcn)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 512, 1, 1, 0, 1, bn_eps=bn_eps))
        self.reduce = _ConvBnReLU(256, 256, 1, 1, 0, 1, bn_eps=bn_eps)

        # Decoder
        self.conv_up5 = ConRu(512, 512, 3, 1)
        if output_stride==16:
            self.conv_up4 = ConRuT(512+512, 512, 3, 1, output_padding=output_padding[0])
        elif output_stride==8:
            self.conv_up4 = ConRu(512+512, 512, 3, 1)
        self.conv_up3 = ConRuT(512+512, 256, 3, 1, output_padding=output_padding[1])
        self.conv_up2 = ConRu(256+256, 256, 3, 1)
        self.conv_up1 = ConRu(256+256, 256, 3, 1)

        self.conv_up0 = ConRu(256+64, 128, 3, 1)
        self.conv_up00 = nn.Sequential(
                         nn.Conv2d(128+in_ch, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64, eps=bn_eps),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, kernel_size=3, padding=1),
                         nn.BatchNorm2d(64, eps=bn_eps),
                         nn.ReLU(),
                         nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.reduce(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.aspp(x6)
        x8 = self.fc1(x7)

        # Decoder
        xup5 = self.conv_up5(x8)
        xup5 = torch.cat([xup5, x5], dim=1)
        xup4 = self.conv_up4(xup5)
        xup4 = torch.cat([xup4, x4], dim=1)
        xup3 = self.conv_up3(xup4)
        xup3 = torch.cat([xup3, x3], dim=1)
        xup2 = self.conv_up2(xup3)
        xup2 = torch.cat([xup2, x2], dim=1)
        xup1 = self.conv_up1(xup2)
        xup1 = torch.cat([xup1, x1], dim=1)
        xup0 = self.conv_up0(xup1)

        xup0 = F.interpolate(xup0, size=x.shape[2:], mode="bilinear", align_corners=False)
        xup0 = torch.cat([xup0, x], dim=1)
        xup00 = self.conv_up00(xup0)
        
        return xup00

if __name__=="__main__":
    m = PMNet(n_classes=1,
    n_blocks=[3, 3, 27, 3],
    atrous_rates=[6, 12, 18],
    multi_grids=[1, 2, 4],
    output_stride=16,)

    B = 4
    H = 256

    input = torch.randn(B, 2, H, H)
    output = m(input)
    print(output.shape)