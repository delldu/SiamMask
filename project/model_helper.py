"""Create model."""# coding=utf-8
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
import math
import numpy as np

import pdb

from typing import List, Tuple
# Only for typing annotations
Tensor = torch.Tensor


class Anchors:
    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.anchor_density = 1

        self.__dict__.update(cfg)

        self.anchor_num = len(self.scales) * len(self.ratios) * (self.anchor_density**2)
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)  # in single position (anchor_num*4)
        self.generate_anchors()
        # pdb.set_trace()

    def generate_anchors(self):
        size = self.stride * self.stride
        count = 0
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density)*anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:
                ws = int(math.sqrt(size*1. / r))
                hs = int(ws * r)

                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [-w*0.5+x_offset, -h*0.5+y_offset, w*0.5+x_offset, h*0.5+y_offset][:]
                    count += 1

def conv2d_dw_group(x, kernel):
    # x.size(), kernel.size() --(torch.Size([1, 256, 29, 29]), torch.Size([1, 256, 5, 5]))
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    # pp out.size() -- torch.Size([1, 256, 25, 25])
    return out

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
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        # (Pdb) a
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

    def forward_corr(self, kernel, input):
        # (Pdb) kernel.size(), input.size() -- (torch.Size([1, 256, 7, 7]), torch.Size([1, 256, 31, 31]))
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        # (Pdb) feature.size() -- torch.Size([1, 256, 25, 25])
        return feature

    def forward(self, kernel, search) -> Tuple[Tensor, Tensor]:
        # (Pdb) kernel.size() -- torch.Size([1, 256, 7, 7])
        # (Pdb) search.size() -- torch.Size([1, 256, 31, 31])

        # corr_feature
        feature = self.forward_corr(kernel, search)
        # feature.size() -- torch.Size([1, 256, 25, 25])
        out = self.head(feature)
        # (Pdb) out.size() -- torch.Size([1, 10, 25, 25])

        return feature, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        # (Pdb) a
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
        assert stride==1 or dilation==1, "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # if out.size() != residual.size():
        #     print(out.size(), residual.size())
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # block = <class 'resnet.Bottleneck'>
        # layers = [3, 4, 6, 3]
        # layer4 = False
        # layer3 = True

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 31x31, 15x15

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2) # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x:x # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x:x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
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
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
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
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        # x.size() -- torch.Size([1, 1024, 15, 15])
        x = self.downsample(x)
        # (Pdb) pp x.size() -- torch.Size([1, 256, 15, 15])
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        # x.size() -- torch.Size([1, 256, 7, 7])
        return x


class ResnetDown(nn.Module):
    def __init__(self, pretrain=False):
        super(ResnetDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        self.downsample = ResDownS(1024, 256)

    def forward(self, x) -> Tuple[List[Tensor], Tensor]:
        # x.size() -- torch.Size([1, 3, 127, 127])
        output = self.features(x)
        # (Pdb) output[0].size(), output[1].size(), output[2].size(), output[3].size()
        # (torch.Size([1, 64, 61, 61]), torch.Size([1, 256, 31, 31]), 
        # torch.Size([1, 512, 15, 15]), torch.Size([1, 1024, 15, 15]))
        p3 = self.downsample(output[-1])
        # (Pdb) p3.size() -- torch.Size([1, 256, 7, 7])

        return output, p3

class RPN(nn.Module):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(RPN, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f) -> Tuple[Tensor, Tensor]:
        _, cls = self.cls(z_f, x_f)
        _, loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(nn.Module):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x) -> Tuple[Tensor, Tensor]:
        return self.mask(z, x)

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 4, 3, padding=1),nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)
        
        for modules in [self.v0, self.v1, self.v2, self.h2, self.h1, self.h0, self.deconv, self.post0, self.post1, self.post2,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f: List[Tensor], corr_feature, anchor_r: int, anchor_c: int):
        # (Pdb) f -- full_feature, type(f), len(f), f[0].size(), f[1].size(), f[2].size(), f[3].size()
        # (<class 'tuple'>, 4, torch.Size([1, 64, 125, 125]), torch.Size([1, 256, 63, 63]), torch.Size([1, 512, 31, 31]), torch.Size([1, 1024, 31, 31]))
        # corr_feature.size() -- torch.Size([1, 256, 25, 25])

        # test = True
        # if test:
        #     p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
        #     p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        #     p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
        # else:
        #     p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
        #     if not (pos is None): p0 = torch.index_select(p0, 0, pos)
        #     p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
        #     if not (pos is None): p1 = torch.index_select(p1, 0, pos)
        #     p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
        #     if not (pos is None): p2 = torch.index_select(p2, 0, pos)
        p0 = F.pad(f[0], [16, 16, 16, 16])[:, :, 4*anchor_r:4*anchor_r + 61, 4*anchor_c:4*anchor_c+61]
        p1 = F.pad(f[1], [8, 8, 8, 8])[:, :, 2 * anchor_r:2 * anchor_r + 31, 2 * anchor_c:2 * anchor_c + 31]
        p2 = F.pad(f[2], [4, 4, 4, 4])[:, :, anchor_r:anchor_r + 15, anchor_c:anchor_c + 15]

        # pos = (12, 12)
        # if not(pos is None):
        #     p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
        # else:
        #     p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)
        p3 = corr_feature[:, :, anchor_r, anchor_c].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out

def get_range_pad(y:int, d:int, maxy:int) -> Tuple[int, int, int, int]:
    y1 = int(y - d/2)
    y2 = int(y1 + d - 1)

    pad1 = max(0, -y1)
    pad2 = max(0, y2 - maxy + 1)

    y1 = y1 + pad1
    y2 = y2 + pad1

    return int(y1), int(y2), int(pad1), int(pad2)

def get_subwindow(image, target_rc:int, target_cc:int, target_size:int, search_size:int, bg_color):
    batch = int(image.size(0))
    chan = int(image.size(1))
    height = int(image.size(2))
    width = int(image.size(3))

    x1, x2, left_pad, right_pad = get_range_pad(target_cc, search_size, width)
    y1, y2, top_pad, bottom_pad = get_range_pad(target_rc, search_size, height)

    big = torch.zeros(batch, chan, height + top_pad + bottom_pad, width + left_pad + right_pad).to(image.device)

    big[:, :, top_pad:top_pad + height, left_pad:left_pad + width] = image

    # xxxx8888
    # big[:, :, 0:top_pad, left_pad:left_pad + width] = bg_color
    # big[:, :, height + top_pad:, left_pad:left_pad + width] = bg_color
    # big[:, :, :, 0:left_pad] = bg_color
    # big[:, :, :, width + left_pad:] = bg_color

    patch = big[:, :, y1:y2 + 1, x1:x2 + 1]

    return F.interpolate(patch, size=(target_size, target_size), mode='nearest')

def get_scale_size(h: int, w: int) -> int:
    # hc = h + (h + w)/2
    # wc = w + (h + w)/2
    # s = sqrt(hc * wc)
    return int(math.sqrt((3 * h + w) * (3 * w + h))/2)

def change(r):
    return torch.max(r, 1. / r)

def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return torch.sqrt(sz2)

def generate_anchor(cfg, score_size):
    # cfg = {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8]}
    # score_size = 25

    anchors = Anchors(cfg)
    anchor = anchors.anchors

    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

    total_stride = anchors.stride
    # total_stride == 8

    anchor_num = anchor.shape[0]
    # anchor_num -- 5
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    # (Pdb) ori == -96

    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()

    # (Pdb) xx -- array([-96, -88, -80, ...,  80,  88,  96])
    # (Pdb) xx.shape -- (3125,)
    # (Pdb) yy -- array([-96, -96, -96, ...,  96,  96,  96])
    # (Pdb) yy.shape -- (3125,)

    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

    # (Pdb) anchor
    # array([[-96., -96., 104.,  32.],
    #        [-88., -96., 104.,  32.],
    #        [-80., -96., 104.,  32.],
    #        ...,
    #        [ 80.,  96.,  32.,  96.],
    #        [ 88.,  96.,  32.,  96.],
    #        [ 96.,  96.,  32.,  96.]], dtype=float32)
    # (Pdb) anchor.shape
    # (3125, 4)
    # anchor[0] -- array([-96., -96., 104.,  32.], (cc, rc, w, h) ?

    return anchor


class SiameseTracker(nn.Module):
    def __init__(self, device='cpu'):
        super(SiameseTracker, self).__init__()
        self.device = device
        self.config = {'stride': 8, 'ratios': [0.25, 0.5, 1, 2, 4], 'scales': [8], 'base_size': 8}
        self.anchor_num = len(self.config["ratios"]) * len(self.config["scales"])
        self.score_size = 25
        self.anchor = torch.from_numpy(generate_anchor(self.config, self.score_size)).to(device)
        # 'anchor':([[-96., -96., 104.,  32.],
        #        [-88., -96., 104.,  32.],
        #        [-80., -96., 104.,  32.],
        #        ...,
        #        [ 80.,  96.,  32.,  96.],
        #        [ 88.,  96.,  32.,  96.],
        #        [ 96.,  96.,  32.,  96.]], dtype=float32)}

        self.instance_size = 255    # for search size
        self.template_size = 127
        self.segment_threshold = 0.35

        self.rpn_model = RPN(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.features = ResnetDown()
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        self.reset_mode(is_training=False)

        # torch.jit.script(self.features)
        # torch.jit.script(self.mask_model)
        # torch.jit.script(self.refine_model)

        window = torch.hamming_window(self.score_size)
        window = window.view(self.score_size, 1) * window.view(1, self.score_size)
        self.window = window.flatten().repeat(self.anchor_num).to(self.device)
        # self.window.size: (225 * 25 * 5) = 3125

        # Set standard template features
        self.zf = torch.zeros(1, 256, 7, 7).to(self.device)

        # Reasonable setting ?
        self.image_height = 256
        self.image_width  = 256
        self.background = torch.zeros([0, 0, 0]).to(self.device)

        # rc -- row center, cc -- column center
        self.target_rc = 64
        self.target_cc = 64
        self.target_h = 127
        self.target_w = 127

        # Last init, set reference
        # self.set_reference(refrence, r, c, h, w)

    # Since nothing in the model calls `set_reference`, the compiler
    # must be explicitly told to compile this method
    @torch.jit.export
    def set_reference(self, reference, r: int, c: int, h: int, w: int):
        """Reference image: Tensor (1x3xHxW format, range: 0, 255, uint8"""

        height = int(reference.size(2))
        width = int(reference.size(3))
        bg_color = reference.mean(dim=3, keepdim=False).mean(dim=2, keepdim=False).squeeze(0)
        self.set_image_size(height, width)
        self.set_background(bg_color)
        self.set_target(r, c, h, w)

        target_e = get_scale_size(h, w)
        z_crop = get_subwindow(reference, r, c, self.template_size, target_e, bg_color)

        # (Pdb) z_crop.shape -- torch.Size([1, 3, 127, 127]), format range: [0, 255]
        self.set_template(z_crop)

    def reset_mode(self, is_training=False):
        if is_training:
            self.features.train()
            self.mask_model.train()
            self.refine_model.train()
        else:
            self.features.eval()
            self.mask_model.eval()
            self.refine_model.eval()

    def set_image_size(self, h: int, w: int):
        self.image_height = h
        self.image_width = w

    def set_target(self, rc:int, cc:int, h:int, w:int):
        self.target_rc = rc
        self.target_cc = cc
        self.target_h = h
        self.target_w = w

    def target_clamp(self):
        self.target_rc = max(0, min(self.image_height, self.target_rc))
        self.target_cc = max(0, min(self.image_width, self.target_cc))
        self.target_h = max(10, min(self.image_height, self.target_h))
        self.target_w = max(10, min(self.image_width, self.target_w))

    def set_template(self, template):
        # (Pdb) template.size() -- torch.Size([1, 3, 127, 127])
        full_feature, template_feature = self.features(template)
        self.zf = template_feature

    def set_background(self, bgcolor):
        self.background = bgcolor

    def track_mask(self, search) -> Tuple[Tensor, Tensor, Tensor, List[Tensor], Tensor]:
        # (Pdb) search.size() -- torch.Size([1, 3, 255, 255])
        full_feature, search_feature = self.features(search)

        # (Pdb) self.zf.size() -- torch.Size([1, 256, 7, 7])
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, search_feature)

        corr_feature, rpn_pred_mask = self.mask_model(self.zf, search_feature)
        
        rpn_pred_score = self.convert_score(rpn_pred_cls)
        rpn_pred_bbox = self.convert_bbox(rpn_pred_loc)

        return rpn_pred_score, rpn_pred_bbox, rpn_pred_mask, full_feature, corr_feature

    def track_refine(self, full_feature: List[Tensor], corr_feature, anchor_r: int, anchor_c: int, target_e: int):
        rpn_pred_mask = self.refine_model(full_feature, corr_feature, anchor_r, anchor_c)
        mask = rpn_pred_mask.sigmoid().view(self.template_size, self.template_size)

        s = target_e / self.instance_size
        # e-target center: x, y format
        e_center = [self.target_cc - target_e/2, self.target_rc - target_e/2]
        # Anchor e_box center
        base_size = 8 #self.config["base_size"]
        config_stride = 8 #self.config["stride"]
        anchor_dr = (anchor_r - base_size / 2) * config_stride
        anchor_dc = (anchor_c - base_size / 2) * config_stride
        # Foreground box
        fg_box = [e_center[0] + anchor_dc * s, e_center[1] + anchor_dr * s, 
                s * self.template_size, s * self.template_size]

        s = self.instance_size / target_e
        bg_box = [int(-fg_box[0] * s), int(-fg_box[1] * s), int(self.image_width * s), int(self.image_height * s)]

        mask_in_img = self.crop_back(mask, bg_box)
        # mask.shape -- (127, 127)
        # (Pdb) mask_in_img.shape -- (480, 854)
        target_mask = (mask_in_img > self.segment_threshold)
        return target_mask


    def convert_bbox(self, delta):
        delta = delta.permute(1, 2, 3, 0).view(4, -1)

    	# delta format: delta_x, delta_y, delta_w, delta_h ?
    	# anchor format: (cc, rc, w, h)
        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0] # x
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1] # y
        delta[2, :] = torch.exp(delta[2, :]) * self.anchor[:, 2]    # w
        delta[3, :] = torch.exp(delta[3, :]) * self.anchor[:, 3]    # h
        return delta

    def convert_score(self, score):
    	# score.size() -- torch.Size([1, 10, 25, 25])
        score = score.permute(1, 2, 3, 0).view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1)
        score = score[:, 1]

        return score

    def crop_back(self, mask, bbox: List[int]):
        """
        https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/utils/net_utils.py        
        affine input: (x1,y1,x2,y2)
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1      ]
        """
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        W = mask.shape[0] # mask width
        H = mask.shape[1] # mask height
        a = float((x2 - x1)/(W - 1))
        c = float((x1 + x2 - W + 1)/(W - 1))
        b = float((y2 - y1)/(H - 1))
        d = float((y1 + y2 - H + 1)/(H - 1))
        # theta = torch.Tensor([[a, 0.0, c], [0.0, b, d]])
        theta = torch.zeros(2, 3)
        theta[0][0] = a
        theta[0][2] = c
        theta[1][1] = b
        theta[1][2] = d
        theta = theta.unsqueeze(0)

        H = int(self.image_height)
        W = int(self.image_width)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False).to(mask.device)

        input = mask.unsqueeze(0).unsqueeze(0)
        output = F.grid_sample(input, grid, mode='bilinear', align_corners=True, padding_mode="zeros")
        # pdb.set_trace()

        output = output.squeeze(0).squeeze(0)
        return output

    def forward(self, image):
        """image: Tensor (1x3xHxW format, range: 0, 255, uint8"""

        bg_color = self.background

        # target_e -- target extend
        target_e = get_scale_size(self.target_h, self.target_w)

        scale_x = self.template_size / target_e
        # target_e -- 457.27， scale_x -- 0.2777325006938416

        # p.instance_size -- 255, p.template_size -- 127
        d_search = (self.instance_size - self.template_size) / 2
        pad = int(d_search / scale_x)
        target_e = target_e + 2 * pad

        x_crop = get_subwindow(image, self.target_rc, self.target_cc, self.instance_size, target_e, bg_color)
        # (Pdb) pp x_crop.shape -- torch.Size([1, 3, 255, 255])

        score, bbox, mask, full_feature, corr_feature = self.track_mask(x_crop)
        # score.size()-- (torch.Size([1, 10, 25, 25]),
        # mask.size() --  torch.Size([1, 3969, 25, 25]))
        # bbox.size() -- torch.Size([1, 20, 25, 25])
        # score.shape -- (3125,)

        # Size penalty
        # For scale_x=template_size/target_e, so template_h/w is virtual template size
        template_h = int(self.target_h * scale_x)
        template_w = int(self.target_w * scale_x)
        bbox_w = bbox[2, :]
        bbox_h = bbox[3, :]
        s_c = change(sz(bbox_w, bbox_h) / (get_scale_size(template_h, template_w)))  # scale penalty
        r_c = change((self.target_w / self.target_h) / (bbox_w / bbox_h))  # ratio penalty
        penalty = torch.exp(-(r_c * s_c - 1) * 0.04)  #penalty_k == 0.04
        penalty_score = penalty * score

        # Smooth penalty score ...
        window_influence = 0.4
        penalty_score = penalty_score * (1 - window_influence) + self.window * window_influence

        best_id = torch.argmax(penalty_score)
        lr = penalty[best_id] * score[best_id]  # lr for OTB

        # for Mask Branch
        # best_anchor = np.unravel_index(best_id, (self.anchor_num, self.score_size, self.score_size))
        # anchor_r, anchor_c = best_anchor[1], best_anchor[2]
        left = best_id % (self.score_size * self.score_size)
        anchor_r = left // self.score_size
        anchor_c = int(left % self.score_size)
        #  pp anchor_r, anchor_c -- (12, 13), mask.size -- (127, 127)

        target_mask = self.track_refine(full_feature, corr_feature, anchor_r, anchor_c, target_e)

        # Update target
        best_bbox = bbox[:, best_id] / scale_x
        self.set_target(int(self.target_rc + best_bbox[1]), 
                        int(self.target_cc + best_bbox[0]),
                        int(self.target_h * (1 - lr) + best_bbox[3] * lr),
                        int(self.target_w * (1 - lr) + best_bbox[2] * lr))
        self.target_clamp()

        return target_mask
