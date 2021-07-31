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
from typing import List, Tuple

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

        # Set gradient as 1.0
        grad_input = torch.ones_like(input)
        grad_target = torch.ones_like(target)

        return (grad_input, grad_target)

    @staticmethod
    def symbolic(g, input, target):
        return g.op("siamese::sub_window", input, target)


class SubWindow(nn.Module):
    def __init__(self, size):
        super(SubWindow, self).__init__()
        self.size = size  # final size

    def forward(self, input, target):
        patch = SubWindowFunction.apply(input, target)
        # zoom in/out patch to (size, size)
        return F.interpolate(patch, size=(self.size, self.size), mode="nearest")


class AnchorBboxFunction(Function):
    @staticmethod
    def forward(ctx, image, target, anchor):
        ctx.save_for_backward(image, target, anchor)
        output = siamese_cpp.anchor_bbox(image, target, anchor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        image, target, anchor = ctx.saved_tensors

        # Set gradient as 1.0
        grad_image = torch.ones_like(image)
        grad_target = torch.ones_like(target)
        grad_anchor = torch.ones_like(anchor)

        return (grad_image, grad_target, grad_anchor)

    @staticmethod
    def symbolic(g, image, target, anchor):
        return g.op("siamese::anchor_bbox", image, target, anchor)


class AnchorBbox(nn.Module):
    def __init__(self):
        super(AnchorBbox, self).__init__()

    def forward(self, image, target, anchor):
        return AnchorBboxFunction.apply(image, target, anchor)


class AffineThetaFunction(Function):
    @staticmethod
    def forward(ctx, mask, bbox):
        ctx.save_for_backward(mask, bbox)
        output = siamese_cpp.affine_theta(mask, bbox)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, bbox = ctx.saved_tensors

        # Set gradient as 1.0
        grad_mask = torch.ones_like(mask)
        grad_bbox = torch.ones_like(bbox)

        return (grad_mask, grad_bbox)

    @staticmethod
    def symbolic(g, mask, bbox):
        return g.op("siamese::affine_theta", mask, bbox)


class AffineTheta(nn.Module):
    def __init__(self):
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

        # Set gradient as 1.0
        grad_score = torch.ones_like(score)
        grad_bbox = torch.ones_like(bbox)
        grad_target = torch.ones_like(target)

        return (grad_score, grad_bbox, grad_target)

    @staticmethod
    def symbolic(g, score, bbox, target):
        return g.op("siamese::best_anchor", score, bbox, target)


class BestAnchor(nn.Module):
    def __init__(self):
        super(BestAnchor, self).__init__()

    def forward(self, score, bbox, target):
        # anchor =  BestAnchorFunction.apply(score, bbox, target)
        return BestAnchorFunction.apply(score, bbox, target)


# AnchorPatches
class AnchorPatchesFunction(Function):
    @staticmethod
    def forward(ctx, full_feature, corr_feature, anchor):
        ctx.save_for_backward(full_feature, corr_feature, anchor)
        output = siamese_cpp.anchor_patches(full_feature, corr_feature, anchor)
        return output[0], output[1], output[2], output[3]

    @staticmethod
    def backward(ctx, grad_output):
        full_feature, corr_feature, anchor = ctx.saved_tensors

        # Set gradient as 1.0
        grad_full_feature = torch.ones_like(full_feature)
        grad_corr_feature = torch.ones_like(corr_feature)
        grad_anchor = torch.ones_like(anchor)

        return (grad_full_feature, grad_corr_feature, grad_anchor)

    @staticmethod
    def symbolic(g, mask, bbox):
        return g.op("siamese::anchor_patches", mask, bbox)


class AnchorPatches(nn.Module):
    def __init__(self):
        super(AnchorPatches, self).__init__()

    def forward(self, full_feature, corr_feature, anchor):
        p0, p1, p2, p3 = AnchorPatchesFunction.apply(full_feature, corr_feature, anchor)
        return p0, p1, p2, p3


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
        return self.downsample(x)


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

        self.anchor_patches = AnchorPatches()

    def forward(self, f: List[Tensor], corr_feature, anchor):
        # f -- full_feature, type(f), len(f), f[0].size(), f[1].size(), f[2].size(), f[3].size()
        # tuple, 4, [1, 64, 125, 125], [1, 256, 63, 63],[1, 512, 31, 31], [1, 1024, 31, 31]
        # corr_feature.size() -- [1, 256, 25, 25]

        p0, p1, p2, p3 = self.anchor_patches(f, corr_feature, anchor)
        p3 = p3.view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out


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
    """Limition: SubWindow only support CPU."""

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
        l = 4
        r = -4
        # x = x[:, :, l:r, l:r]
        return template_feature[:, :, l:r, l:r]  # [1, 256, 7, 7]

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
    def __init__(self):
        super(SiameseTracker, self).__init__()
        self.config = {
            "stride": 8,
            "ratios": [0.25, 0.5, 1, 2, 4],
            "scales": [8],
            "base_size": 8,
        }
        self.anchor_num = len(self.config["ratios"]) * len(self.config["scales"])
        self.score_size = 25
        self.anchor = torch.from_numpy(generate_anchor(self.config, self.score_size))
        # 'anchor':([[-96., -96., 104.,  32.],
        #        [-88., -96., 104.,  32.],
        #        [-80., -96., 104.,  32.],
        #        ...,
        #        [ 80.,  96.,  32.,  96.],
        #        [ 88.,  96.,  32.,  96.],
        #        [ 96.,  96.,  32.,  96.]], dtype=float32)}
        # self.anchor.size() -- [3125, 4], 3125 = 25x25x5

        self.template_size = 127
        self.instance_size = 255
        self.segment_threshold = 0.35

        self.rpn_model = RPN(
            anchor_num=self.anchor_num, feature_in=256, feature_out=256
        )
        self.features = ResnetDown()
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        self.reset_mode(is_training=False)

        # Call c++ ...
        self.affine_theta = AffineTheta()
        self.best_anchor = BestAnchor()
        self.subwindow = SubWindow(self.instance_size)
        self.anchor_bbox = AnchorBbox()

    def reset_mode(self, is_training=False):
        if is_training:
            self.features.train()
            self.mask_model.train()
            self.refine_model.train()
        else:
            self.features.eval()
            self.mask_model.eval()
            self.refine_model.eval()

    def refine_mask(self, image, mask, bbox):
        # mask.size() -- [1, 1, 127, 127]

        theta = self.affine_theta(mask, bbox)
        theta = theta.unsqueeze(0).to(mask.device)

        outmask = image[:, 0:1, :, :]  # ==> Get size
        grid = F.affine_grid(theta, outmask.size(), align_corners=False).to(mask.device)

        output = F.grid_sample(
            mask, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )
        return output

    def convert_score(self, score):
        score = score.permute(1, 2, 3, 0).view(2, -1).permute(1, 0)
        # score.size() -- [1, 10, 25, 25] --> [10, 25, 25] --> [2, 5x25x25] --> [3125, 2]
        score = F.softmax(score, dim=1)
        score = score[:, 1]  # Only use fg score, drop out bg score
        return score

    def convert_bbox(self, delta):
        delta = delta.permute(1, 2, 3, 0).view(4, -1)
        # delta.size() -- [1, 20, 25, 25] --> [20, 25, 25] --> [4, 3125]

        # delta format: delta_x, delta_y, delta_w, delta_h ?
        # anchor format: (cc, rc, w, h)
        # self.anchor = self.anchor.to(delta.device)
        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0]  # x
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1]  # y
        delta[2, :] = torch.exp(delta[2, :]) * self.anchor[:, 2]  # w
        delta[3, :] = torch.exp(delta[3, :]) * self.anchor[:, 3]  # h
        return delta

    def forward(self, image, template, target):
        """image: Tensor (1x3xHxW format, range: 0, 255, uint8
        template --[1, 256, 7, 7]
        """
        big_target = torch.cat((target[0:2], 2.0 * target[2:4]), dim=0)

        x_crop = self.subwindow(image, big_target)  # x_crop.size -- [1, 3, 255, 255]
        full_feature, search_feature = self.features(x_crop)
        # pdb.set_trace()

        rpn_score, rpn_bbox = self.rpn_model(template, search_feature)
        score = self.convert_score(rpn_score)
        bbox = self.convert_bbox(rpn_bbox)
        corr_feature, mask = self.mask_model(template, search_feature)

        # Do something around anchor ...
        anchor = self.best_anchor(score, bbox, target)

        # Track refine ...
        anchor_mask = self.refine_model(full_feature, corr_feature, anchor)
        anchor_mask = anchor_mask.sigmoid().view(
            1, 1, self.template_size, self.template_size
        )
        anchor_bbox = self.anchor_bbox(image, big_target, anchor)

        final_mask = self.refine_mask(image, anchor_mask, anchor_bbox)
        # anchor_mask.shape -- (1, 1, 127, 127)
        # final_mask.shape -- (480, 854)
        target_mask = (final_mask > self.segment_threshold).type_as(image)

        # Update target
        new_target = anchor[2:6]
        new_target = new_target.clamp(0, max(image.size(2), image.size(3)))

        return target_mask, new_target
