"""Create model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 16日 星期三 18:41:24 CST
# ***
# ************************************************************************************/
#
# The following code mainly comes from https://github.com/foolwood/SiamMask
#
# Thanks you, guys, I love you !
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import os
import math
import numpy as np
import pdb
from typing import List, Tuple, Optional

from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op

import siamese_cpp

# Only for typing annotations
Tensor = torch.Tensor

# affine_grid prototype:
#     torch.nn.functional.affine_grid(theta, size, align_corners=None)
@parse_args("v", "v", "i")
def affine_grid_generator(g, theta, size, align_corners):
    return g.op(
        "onnxservice::affine_grid_generator", theta, size, align_corners_i=align_corners
    )


register_op("affine_grid_generator", affine_grid_generator, "", 11)


@parse_args("v", "v", "i", "i", "i")
def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners=False):
    """
    torch.nn.functional.grid_sample(input, grid, mode='bilinear',
        padding_mode='zeros', align_corners=None)
    Need convert interpolation_mode, padding_mode? NO for simpler !!!
    """
    return g.op(
        "onnxservice::grid_sampler",
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners,
    )


register_op("grid_sampler", grid_sampler, "", 11)


class Anchors:
    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.anchor_density = 1

        self.__dict__.update(cfg)

        self.anchor_num = (
            len(self.scales) * len(self.ratios) * (self.anchor_density ** 2)
        )  # 5
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)  # (5, 4)
        self.generate_anchors()
        # (Pdb) self.anchors
        # array([[-64., -16.,  64.,  16.],
        #        [-44., -20.,  44.,  20.],
        #        [-32., -32.,  32.,  32.],
        #        [-20., -40.,  20.,  40.],
        #        [-16., -64.,  16.,  64.]], dtype=float32)

    def generate_anchors(self):
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size * 1.0 / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1


def conv2d_dw_group(x, kernel):
    # x.size(), kernel.size() --[1, 256, 29, 29], [1, 256, 5, 5]
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch * channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(
        batch * channel, 1, kernel.size(2), kernel.size(3)
    )  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    # out.size() -- [1, 256, 25, 25]
    return out


class SubWindowFunction(Function):
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)        
        output = siamese_cpp.sub_window(input, target)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        # Set grad as 1.0
        grad_input = torch.ones_like(input)
        grad_target = torch.ones_like(target)

        return (grad_input, grad_target)

    @staticmethod
    def symbolic(g, input, target):
        return g.op("siamese::sub_window", input, target) 


class SubWindow(nn.Module):
    def __init__(self, size):
        super(SubWindow, self).__init__()
        self.size = size # final size

    def forward(self, input, target):
        patch =  SubWindowFunction.apply(input, target)
        # zoom in/out patch to (size, size)
        return F.interpolate(patch, size=(self.size, self.size), mode="nearest")


class AnchorBgBoxFunction(Function):
    @staticmethod
    def forward(ctx, anchor, target):
        ctx.save_for_backward(anchor, target)        
        output = siamese_cpp.anchor_bgbox(anchor, target)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        anchor, target = ctx.saved_tensors

        # Set grad as 1.0
        grad_anchor = torch.ones_like(anchor)
        grad_target = torch.ones_like(target)

        return (grad_anchor, grad_target)

    @staticmethod
    def symbolic(g, anchor, target):
        return g.op("siamese::anchor_bgbox", anchor, target) 


class AnchorBgBox(nn.Module):
    def __init__(self, size):
        super(AnchorBgBox, self).__init__()

    def forward(self, anchor, target):
        # bgbox =  AnchorBgBoxFunction.apply(anchor, target)
        return AnchorBgBoxFunction.apply(anchor, target)


class AffineThetaFunction(Function):
    @staticmethod
    def forward(ctx, mask, bbox):
        ctx.save_for_backward(mask, bbox)        
        output = siamese_cpp.affine_theta(mask, bbox)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, bbox = ctx.saved_tensors

        # Set grad as 1.0
        grad_input = torch.ones_like(mask)
        grad_bbox = torch.ones_like(bbox)

        return (grad_input, grad_bbox)

    @staticmethod
    def symbolic(g, mask, bbox):
        return g.op("siamese::affine_theta", mask, bbox) 


class AffineTheta(nn.Module):
    def __init__(self, size):
        super(AffineTheta, self).__init__()

    def forward(self, mask, bbox):
        # theta =  AffineThetaFunction.apply(mask, bbox)
        return AffineThetaFunction.apply(mask, bbox)


# BestAnchor
class BestAnchorFunction(Function):
    @staticmethod
    def forward(ctx, score, bbox, target):
        ctx.save_for_backward(score, bbox, target)        
        return siamese_cpp.best_anchor(score, bbox, target)

    @staticmethod
    def backward(ctx, grad_output):
        score, bbox, target = ctx.saved_tensors

        # Set grad as 1.0
        grad_score = torch.ones_like(score)
        grad_bbox = torch.ones_like(bbox)
        grad_target = torch.ones_like(target)

        return (grad_score, grad_bbox, grad_target)

    @staticmethod
    def symbolic(g, score, bbox, target):
        return g.op("siamese::best_anchor", score, bbox, target) 


class BestAnchor(nn.Module):
    def __init__(self, size):
        super(BestAnchor, self).__init__()

    def forward(self, score, bbox, target):
        # anchor, new_target =  BestAnchorFunction.apply(score, bbox, target)
        return BestAnchorFunction.apply(score, bbox, target)


# AnchorPatchs
class AnchorPatchsFunction(Function):
    @staticmethod
    def forward(ctx, full_feature, corr_feature, anchor):
        ctx.save_for_backward(full_feature, corr_feature, anchor)        
        return siamese_cpp.anchor_patchs(full_feature, corr_feature, anchor)

    @staticmethod
    def backward(ctx, grad_output):
        full_feature, corr_feature, anchor = ctx.saved_tensors

        # Set grad as 1.0
        grad_full_feature = torch.ones_like(full_feature)
        grad_corr_feature = torch.ones_like(corr_feature)
        grad_anchor = torch.ones_like(anchor)

        return (grad_full_feature, grad_corr_feature, grad_anchor)

    @staticmethod
    def symbolic(g, mask, bbox):
        return g.op("siamese::anchor_patchs", mask, bbox) 


class AnchorPatchs(nn.Module):
    def __init__(self, size):
        super(AnchorPatchs, self).__init__()

    def forward(self, full_feature, corr_feature, anchor):
        # p0, p1, p2, p3 = AnchorPatchsFunction.apply(full_feature, corr_feature, anchor)
        return AnchorPatchsFunction.apply(full_feature, corr_feature, anchor)


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        # self = DepthCorr(
        #   (conv_kernel): Sequential(
        #     (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        #   (conv_search): Sequential(
        #     (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        #   (head): Sequential(
        #     (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #     (3): Conv2d(256, 10, kernel_size=(1, 1), stride=(1, 1))
        #   )
        # )
        # in_channels = 256
        # hidden = 256
        # out_channels = 10
        # kernel_size = 3

    # cross-correlated == siamese network !!!
    def forward_corr(self, kernel, input):
        # kernel.size(), input.size() -- [1, 256, 7, 7], [1, 256, 31, 31]
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        # feature.size() -- [1, 256, 25, 25]
        return feature

    def forward(self, kernel, search):
        # kernel.size() -- [1, 256, 7, 7]
        # search.size() -- [1, 256, 31, 31]

        # corr_feature
        feature = self.forward_corr(kernel, search)
        # feature.size() -- [1, 256, 25, 25] -- for mask
        out = self.head(feature)
        # out.size() -- [1, 10, 25, 25]

        return feature, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        # self = Bottleneck(
        #   (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #   (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (relu): ReLU(inplace=True)
        #   (downsample): Sequential(
        #     (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        # )
        # inplanes = 64
        # planes = 64
        # stride = 1
        # downsample = Sequential(
        #   (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        # dilation = 1

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, layer4=False, layer3=False):
        super(ResNet, self).__init__()
        # block = <class 'resnet.Bottleneck'>
        # layers = [3, 4, 6, 3]
        # layer4 = False
        # layer3 = True

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 31x31, 15x15

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1, dilation=2
            )  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=4
            )  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=3,
                        stride=stride,
                        bias=False,
                        padding=padding,
                        dilation=dd,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50."""
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
            nn.BatchNorm2d(outplane),
        )

    def forward(self, x):
        # x.size() -- [1, 1024, 15, 15]
        x = self.downsample(x)
        # x.size() -- [1, 256, 15, 15]
        # xxxx8888
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]

        # x.size() -- [1, 256, 7, 7]
        return x


class ResnetDown(nn.Module):
    def __init__(self, pretrain=False):
        super(ResnetDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        self.downsample = ResDownS(1024, 256)

    def forward(self, x) -> Tuple[List[Tensor], Tensor]:
        # x.size() -- [1, 3, 127, 127]
        output = self.features(x)
        # output[0].size(), output[1].size(), output[2].size(), output[3].size()
        # [1, 64, 61, 61], [1, 256, 31, 31],
        # [1, 512, 15, 15], [1, 1024, 15, 15]
        p3 = self.downsample(output[-1])
        # p3.size() -- [1, 256, 7, 7]

        return output, p3


class RPN(nn.Module):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(RPN, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num  # 2 means bg, fg score
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z, x) -> Tuple[Tensor, Tensor]:
        # z, x -- template_feature, search_feature
        # z.size() -- [1, 256, 7, 7] , x.size() -- [1, 256, 31, 31]
        _, cls = self.cls(z, x)
        _, loc = self.loc(z, x)
        return cls, loc


class MaskCorr(nn.Module):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz ** 2)

    def forward(self, z, x) -> Tuple[Tensor, Tensor]:
        # z, x -- template_feature, search_feature
        # z.size() -- [1, 256, 7, 7] , x.size() -- [1, 256, 31, 31]
        return self.mask(z, x)


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.v1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.v2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.h0 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

        for modules in [
            self.v0,
            self.v1,
            self.v2,
            self.h2,
            self.h1,
            self.h0,
            self.deconv,
            self.post0,
            self.post1,
            self.post2,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f: List[Tensor], corr_feature, anchor_r: int, anchor_c: int):
        # f -- full_feature, type(f), len(f), f[0].size(), f[1].size(), f[2].size(), f[3].size()
        # tuple, 4, [1, 64, 125, 125], [1, 256, 63, 63],[1, 512, 31, 31], [1, 1024, 31, 31]
        # corr_feature.size() -- [1, 256, 25, 25]

        p0 = F.pad(f[0], [16, 16, 16, 16])[
            :, :, 4 * anchor_r : 4 * anchor_r + 61, 4 * anchor_c : 4 * anchor_c + 61
        ]
        p1 = F.pad(f[1], [8, 8, 8, 8])[
            :, :, 2 * anchor_r : 2 * anchor_r + 31, 2 * anchor_c : 2 * anchor_c + 31
        ]
        p2 = F.pad(f[2], [4, 4, 4, 4])[
            :, :, anchor_r : anchor_r + 15, anchor_c : anchor_c + 15
        ]

        # pos = (12, 12)
        p3 = corr_feature[:, :, anchor_r, anchor_c].view(-1, 256, 1, 1)

        # xxxx6666 siameses::anchor_patchs(full_feature, anchor_tensor) ==> [p0, p1, p2, p3]

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out


def get_range_pad(y: int, d: int, maxy: int) -> Tuple[int, int, int, int]:
    y1 = int(y - d / 2)
    y2 = int(y1 + d - 1)

    pad1 = max(0, -y1)
    pad2 = max(0, y2 - maxy + 1)

    y1 = y1 + pad1
    y2 = y2 + pad1

    return int(y1), int(y2), int(pad1), int(pad2)


def get_subwindow(
    image, target_rc: int, target_cc: int, target_size: int, search_size: int):
    # batch = int(image.size(0))
    # chan = int(image.size(1))
    height = int(image.size(2))
    width = int(image.size(3))

    x1, x2, left_pad, right_pad = get_range_pad(target_cc, search_size, width)
    y1, y2, top_pad, bottom_pad = get_range_pad(target_rc, search_size, height)

    # padding_left,padding_right, padding_top, padding_bottom
    big = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))

    patch = big[:, :, y1 : y2 + 1, x1 : x2 + 1]

    return F.interpolate(patch, size=(target_size, target_size), mode="nearest")


def get_scale_size(h: int, w: int) -> int:
    # hc = h + (h + w)/2
    # wc = w + (h + w)/2
    # s = sqrt(hc * wc)
    return int(math.sqrt((3 * h + w) * (3 * w + h)) / 2)


def get_max_change(r):
    return torch.max(r, 1.0 / r)


def get_scale_tensor(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return torch.sqrt(sz2)


def generate_anchor(cfg, score_size):
    # cfg = {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8]}
    # score_size = 25

    anchors = Anchors(cfg)
    anchor = anchors.anchors

    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = anchors.stride
    # total_stride == 8

    anchor_num = anchor.shape[0]
    # anchor_num -- 5
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = -(score_size // 2) * total_stride
    # ori == -96

    xx, yy = np.meshgrid(
        [ori + total_stride * dx for dx in range(score_size)],
        [ori + total_stride * dy for dy in range(score_size)],
    )
    xx, yy = (
        np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
        np.tile(yy.flatten(), (anchor_num, 1)).flatten(),
    )

    # xx -- array([-96, -88, -80, ...,  80,  88,  96])
    # xx.shape -- (3125,)
    # yy -- array([-96, -96, -96, ...,  96,  96,  96])
    # yy.shape -- (3125,)

    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    # anchor
    # array([[-96., -96., 104.,  32.],
    #        [-88., -96., 104.,  32.],
    #        [-80., -96., 104.,  32.],
    #        ...,
    #        [ 80.,  96.,  32.,  96.],
    #        [ 88.,  96.,  32.,  96.],
    #        [ 96.,  96.,  32.,  96.]], dtype=float32)
    # anchor.shape
    # (3125, 4)
    # anchor[0] -- array([-96., -96., 104.,  32.], (cc, rc, w, h) ?

    return anchor


class SiameseTemplate(nn.Module):
    '''Limition: SubWindow only support CPU.'''

    def __init__(self):
        super(SiameseTemplate, self).__init__()

        self.subwindow = SubWindow(127)
        self.features = ResnetDown()
        self.load_weights("models/image_siammask.pth")

    def forward(self, image, target):
        z_crop = self.subwindow(image.cpu(), target.cpu()).to(image.device)
        # z_crop.shape -- [1, 3, 127, 127], format range: [0, 255]
        full_feature, template_feature = self.features(z_crop)
        # template_feature.size() -- [1, 256, 15, 15]
        # Continue sample down
        #     l = 4
        #     r = -4
        #     x = x[:, :, l:r, l:r]
        return template_feature[:, :, 4:12, 4:12]  # [1, 256, 7, 7]

    def load_weights(self, path):
        """Load model weights."""
        if not os.path.exists(path):
            print("File '{}' does not exist.".format(path))
            return
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        target_state_dict = self.state_dict()
        for n, p in state_dict.items():
            if n in target_state_dict.keys():
                target_state_dict[n].copy_(p)


class SiameseTracker(nn.Module):
    def __init__(self, device="cpu"):
        super(SiameseTracker, self).__init__()
        self.device = device
        self.config = {
            "stride": 8,
            "ratios": [0.25, 0.5, 1, 2, 4],
            "scales": [8],
            "base_size": 8,
        }
        self.anchor_num = len(self.config["ratios"]) * len(self.config["scales"])
        self.score_size = 25
        self.anchor = torch.from_numpy(
            generate_anchor(self.config, self.score_size)
        ).to(device)
        # 'anchor':([[-96., -96., 104.,  32.],
        #        [-88., -96., 104.,  32.],
        #        [-80., -96., 104.,  32.],
        #        ...,
        #        [ 80.,  96.,  32.,  96.],
        #        [ 88.,  96.,  32.,  96.],
        #        [ 96.,  96.,  32.,  96.]], dtype=float32)}
        # self.anchor.size() -- [3125, 4], 3125 = 25x25x5

        self.instance_size = 255  # for search size
        self.template_size = 127
        self.segment_threshold = 0.35

        self.rpn_model = RPN(
            anchor_num=self.anchor_num, feature_in=256, feature_out=256
        )
        self.features = ResnetDown()
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        self.reset_mode(is_training=False)

        # Set standard template features
        self.template_feature = torch.zeros(1, 256, 7, 7).to(self.device)

        # Reasonable setting ?
        self.image_height = 256
        self.image_width = 256

        # rc -- row center, cc -- column center
        self.target_rc = 64
        self.target_cc = 64
        self.target_h = 127
        self.target_w = 127

    # Since nothing in the model calls `set_reference`, the compiler
    # must be explicitly told to compiler this method
    @torch.jit.export
    def set_reference(self, reference, target):
        """Reference image: Tensor (1x3xHxW format, range: 0, 255, uint8"""
        # xxxx6666, siamese::sub_window -- via reference and target
        r = int(target[0])
        c = int(target[1])
        h = int(target[2])
        w = int(target[3])

        height = int(reference.size(2))
        width = int(reference.size(3))
        self.image_height = height
        self.image_width = width

        self.set_target(r, c, h, w)

        target_e = get_scale_size(h, w)

        z_crop = get_subwindow(reference, r, c, self.template_size, target_e)
        # xxxx6666 ==> class WindowSize(self.size=template_size/instance_size)


        # z_crop.shape -- [1, 3, 127, 127], format range: [0, 255]
        full_feature, temp_feature = self.features(z_crop)
        self.template_feature = temp_feature  # [1, 256, 7, 7]

    def reset_mode(self, is_training=False):
        if is_training:
            self.features.train()
            self.mask_model.train()
            self.refine_model.train()
        else:
            self.features.eval()
            self.mask_model.eval()
            self.refine_model.eval()

    def set_target(self, rc: int, cc: int, h: int, w: int):
        self.target_rc = rc
        self.target_cc = cc
        self.target_h = h
        self.target_w = w

    # ==> include in best_match ...
    def target_clamp(self):
        self.target_rc = max(0, min(self.image_height, self.target_rc))
        self.target_cc = max(0, min(self.image_width, self.target_cc))
        self.target_h = max(10, min(self.image_height, self.target_h))
        self.target_w = max(10, min(self.image_width, self.target_w))

    def track_mask(self, search) -> Tuple[Tensor, Tensor, Tensor, List[Tensor], Tensor]:
        # search.size() -- [1, 3, 255, 255]
        full_feature, search_feature = self.features(search)

        # self.template_feature.size() --[1, 256, 7, 7]
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(
            self.template_feature, search_feature
        )

        corr_feature, rpn_pred_mask = self.mask_model(
            self.template_feature, search_feature
        )

        rpn_pred_score = self.convert_score(rpn_pred_cls)
        rpn_pred_bbox = self.convert_bbox(rpn_pred_loc)

        return rpn_pred_score, rpn_pred_bbox, rpn_pred_mask, full_feature, corr_feature

    # xxxx6666, siamese::anchor_bgbox(Anchor_tensor, target_tensor) ==> Box_Tensor
    def anchor_bgbox(self, anchor_r, anchor_c, target_e):
        s = target_e / self.instance_size
        # e-target center: x, y format
        e_center = [self.target_cc - target_e / 2, self.target_rc - target_e / 2]
        # Anchor e_box center
        base_size = 8  # self.config["base_size"]
        config_stride = 8  # self.config["stride"]
        anchor_dr = (anchor_r - base_size / 2) * config_stride
        anchor_dc = (anchor_c - base_size / 2) * config_stride
        # Foreground box
        fg_box = [
            e_center[0] + anchor_dc * s,    # col
            e_center[1] + anchor_dr * s,    # row
            s * self.template_size,         # 
            s * self.template_size,
        ]

        s = self.instance_size / target_e
        bg_box = [
            int(-fg_box[0] * s),
            int(-fg_box[1] * s),
            int(self.image_width * s),
            int(self.image_height * s),
        ]
        return bg_box

    def track_refine(
        self,
        full_feature: List[Tensor],
        corr_feature,
        anchor_r: int,
        anchor_c: int,
        target_e: int,
    ):
        rpn_pred_mask = self.refine_model(
            full_feature, corr_feature, anchor_r, anchor_c
        )
        mask = rpn_pred_mask.sigmoid().view(1, 1, self.template_size, self.template_size)

        bg_box = self.anchor_bgbox(anchor_r, anchor_c, target_e)
        mask_in_img = self.crop_back(mask, bg_box)

        # mask.shape -- (127, 127)
        # mask_in_img.shape -- (480, 854)
        target_mask = mask_in_img > self.segment_threshold
        return target_mask

    def convert_bbox(self, delta):
        delta = delta.permute(1, 2, 3, 0).view(4, -1)
        # delta.size() -- [1, 20, 25, 25] --> [20, 25, 25] --> [4, 3125]

        # delta format: delta_x, delta_y, delta_w, delta_h ?
        # anchor format: (cc, rc, w, h)
        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0]  # x
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1]  # y
        delta[2, :] = torch.exp(delta[2, :]) * self.anchor[:, 2]  # w
        delta[3, :] = torch.exp(delta[3, :]) * self.anchor[:, 3]  # h
        return delta

    def convert_score(self, score):
        score = score.permute(1, 2, 3, 0).view(2, -1).permute(1, 0)
        # score.size() -- [1, 10, 25, 25] --> [10, 25, 25] --> [2, 5x25x25] --> [3125, 2]
        score = F.softmax(score, dim=1)
        score = score[:, 1]  # Only use fg score, drop out bg score
        return score

    # xxxx6666 siamese::affine_theta
    def affine_theta(self, mask, bbox):
        """
        https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/utils/net_utils.py
        affine input: (x1,y1,x2,y2)
        [  x2-x1             x1 + x2 - W + 1  ]
        [a=-----      0    c=---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0    b=-----  d=---------------  ]
        [           H - 1         H - 1      ]
        """
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        H = mask.shape[2]  # mask height
        W = mask.shape[3]  # mask width
        a = float((x2 - x1) / (W - 1))
        c = float((x1 + x2 - W + 1) / (W - 1))
        b = float((y2 - y1) / (H - 1))
        d = float((y1 + y2 - H + 1) / (H - 1))
        # theta = torch.Tensor([[a, 0.0, c], [0.0, b, d]]), !!! script not support !!!
        theta = torch.zeros(2, 3)
        theta[0][0] = a
        theta[0][2] = c
        theta[1][1] = b
        theta[1][2] = d

        return theta

    def crop_back(self, mask, bbox):
        # mask.size() -- [1, 1, 127, 127]

        theta = self.affine_theta(mask, bbox)
        theta = theta.unsqueeze(0)

        size = (1, 1, self.image_height, self.image_width)
        grid = F.affine_grid(theta, size, align_corners=False).to(mask.device)

        output = F.grid_sample(
            mask, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )
        output = output.squeeze(0).squeeze(0)

        return output

    # xxxx6666, siamese::best_match(score, bbox, target) ==> [anchor_tensor, new_target]
    def best_match(self, score, bbox, scale_x):
        # Size penalty
        # For scale_x=template_size/target_e, so template_h/w is virtual template size
        template_h = int(self.target_h * scale_x)
        template_w = int(self.target_w * scale_x)
        bbox_w = bbox[2, :]
        bbox_h = bbox[3, :]
        s_c = get_max_change(
            get_scale_tensor(bbox_w, bbox_h) / (get_scale_size(template_h, template_w))
        )  # scale penalty
        r_c = get_max_change(
            (self.target_w / self.target_h) / (bbox_w / bbox_h)
        )  # ratio penalty
        penalty = torch.exp(-(r_c * s_c - 1) * 0.04)  # penalty_k == 0.04
        penalty_score = penalty * score

        best_id = torch.argmax(penalty_score)
        lr = penalty[best_id] * score[best_id]

        # Mask Branch
        left = best_id % (self.score_size * self.score_size)
        anchor_r = int(left // self.score_size)
        anchor_c = int(left % self.score_size)
        best_bbox = bbox[:, best_id] / scale_x

        # anchor_r, anchor_c -- (12, 13), mask.size -- (127, 127)

        return anchor_r, anchor_c, lr, best_bbox

    # xxxx6666 SiameseTemplate(reference, target) ==> template_feature
    def forward(self, image, target: Optional[Tensor]):
        """image: Tensor (1x3xHxW format, range: 0, 255, uint8"""

        if target is not None:
            self.set_reference(image, target)

        # target_e -- target extend
        target_e = get_scale_size(self.target_h, self.target_w)

        x_crop = get_subwindow(
            image,
            self.target_rc,
            self.target_cc,
            self.instance_size,
            2.0*target_e,
        )
        # x_crop.shape -- [1, 3, 255, 255]

        score, bbox, mask, full_feature, corr_feature = self.track_mask(x_crop)

        scale_x = self.template_size / target_e
        # target_e -- 457.27， scale_x -- 0.2777325006938416
        anchor_r, anchor_c, lr, best_bbox = self.best_match(score, bbox, scale_x)

        # xxxx6666, track_refine( ... anchor_tensor, target ...)
        target_mask = self.track_refine(
            full_feature, corr_feature, anchor_r, anchor_c, 2.0*target_e
        )

        # Update target
        # xxxx6666 absorb by best_match !!!
        self.set_target(
            int(self.target_rc + best_bbox[1]),
            int(self.target_cc + best_bbox[0]),
            int(self.target_h * (1 - lr) + best_bbox[3] * lr),
            int(self.target_w * (1 - lr) + best_bbox[2] * lr),
        )
        self.target_clamp()

        return target_mask


if __name__ == "__main__":
    model = SiameseTemplate()
    model = model.eval()

    print(model)

    with torch.no_grad():
        output = model(torch.randn(1, 3, 1024, 1024), torch.Tensor([240, 330, 280, 180]))

    print(output)

    pdb.set_trace()
