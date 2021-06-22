# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchors import Anchors
from models.rpn import DepthCorr
import math
import numpy as np

from typing import Dict, List, Tuple

import pdb

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
        # (Pdb) a
        # self = ResNet()
        # block = <class 'resnet.Bottleneck'>
        # layers = [3, 4, 6, 3]
        # layer4 = False
        # layer3 = True

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, # 3
                               bias=False)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        # x.size() -- torch.Size([1, 1024, 15, 15])
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        # x.size() -- torch.Size([1, 256, 7, 7])
        return x


class ResDown(nn.Module):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        # if pretrain:
        #     load_pretrain(self.features, 'resnet.model')

        self.downsample = ResDownS(1024, 256)

    def forward(self, x):
        # x.size() -- torch.Size([1, 3, 127, 127])
        output = self.features(x)
        # (Pdb) output[0].size(), output[1].size(), output[2].size(), output[3].size()
        # (torch.Size([1, 64, 61, 61]), torch.Size([1, 256, 31, 31]), 
        # torch.Size([1, 512, 15, 15]), torch.Size([1, 1024, 15, 15]))
        p3 = self.downsample(output[-1])
        # (Pdb) p3.size() -- torch.Size([1, 256, 7, 7])

        return output, p3

    # def forward_all(self, x):
    #     # (Pdb) x.size() -- torch.Size([1, 3, 255, 255])
    #     output = self.features(x)
    #     p3 = self.downsample(output[-1])
    #     return output, p3


class UP(nn.Module):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(nn.Module):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)

        # self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        # pred_mask = self.mask_model.mask.head(self.corr_feature)


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

    def forward(self, f, corr_feature, pos):
        # (Pdb) type(f), len(f), f[0].size(), f[1].size(), f[2].size(), f[3].size()
        # (<class 'tuple'>, 4, torch.Size([1, 64, 125, 125]), torch.Size([1, 256, 63, 63]), torch.Size([1, 512, 31, 31]), torch.Size([1, 1024, 31, 31]))
        # corr_feature.size() -- torch.Size([1, 256, 25, 25])
        # (Pdb) type(pos), len(pos), type(pos[0]), type(pos[1])
        # (<class 'tuple'>, 2, <class 'numpy.int64'>, <class 'numpy.int64'>)

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
        p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
        p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]

        # pos = (12, 12)
        if not(pos is None):
            p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
        else:
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out


def generate_anchor(cfg, score_size):
    # cfg = {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'round_dight': 0}
    # score_size = 25

    anchors = Anchors(cfg)
    anchor = anchors.anchors

    # (Pdb) anchor == anchors.anchors
    # array([[-52., -16.,  52.,  16.],
    #        [-44., -20.,  44.,  20.],
    #        [-32., -32.,  32.,  32.],
    #        [-20., -40.,  20.,  40.],
    #        [-16., -48.,  16.,  48.]], dtype=float32)
    # (Pdb) anchors.anchors.shape -- (5, 4)

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

    return anchor


class SiameseTracker(nn.Module):
    def __init__(self):
        super(SiameseTracker, self).__init__()
        self.anchors = {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'base_size': 8}
        
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.score_size = 25
        self.anchor = generate_anchor(self.anchors, self.score_size)
        # 'anchor': array([[-96., -96., 104.,  32.],
        #        [-88., -96., 104.,  32.],
        #        [-80., -96., 104.,  32.],
        #        ...,
        #        [ 80.,  96.,  32.,  96.],
        #        [ 88.,  96.,  32.,  96.],
        #        [ 96.,  96.,  32.,  96.]], dtype=float32)}

        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)

        self.instance_size = 255    # for search size
        self.template_size = 127
        self.penalty_k = 0.04
        self.segment_threshold = 0.35

        self.features = ResDown()
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

    def template(self, template):
        # (Pdb) template.size() -- torch.Size([1, 3, 127, 127])
        _, self.zf = self.features(template)

    def track_mask(self, search):
        # (Pdb) search.size() -- torch.Size([1, 3, 255, 255])
        self.feature, self.search = self.features(search)
        # (Pdb) self.zf.size() -- torch.Size([1, 256, 7, 7])
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, self.search)


        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)

        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos)
        return pred_mask
