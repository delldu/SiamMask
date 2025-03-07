# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from typing import Dict, List, Tuple

import pdb


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

    def forward(self, kernel, search):
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

    def forward(self, z_f, x_f)->Tuple[torch.Tensor, torch.Tensor]:
        _, cls = self.cls(z_f, x_f)
        _, loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(nn.Module):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
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
        p0 = F.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
        p1 = F.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p2 = F.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]

        # pos = (12, 12)
        # if not(pos is None):
        #     p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
        # else:
        #     p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)
        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.interpolate(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.interpolate(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.interpolate(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out


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
    # anchor[0] -- array([-96., -96., 104.,  32.], (x, y, w, h) ?

    return anchor


class SiameseTracker(nn.Module):
    def __init__(self):
        super(SiameseTracker, self).__init__()
        self.config = {'stride': 8, 'ratios': [0.25, 0.5, 1, 2, 4], 'scales': [8], 'base_size': 8}

        self.anchor_num = len(self.config["ratios"]) * len(self.config["scales"])
        self.score_size = 25
        self.anchor = generate_anchor(self.config, self.score_size)
        # 'anchor': array([[-96., -96., 104.,  32.],
        #        [-88., -96., 104.,  32.],
        #        [-80., -96., 104.,  32.],
        #        ...,
        #        [ 80.,  96.,  32.,  96.],
        #        [ 88.,  96.,  32.,  96.],
        #        [ 96.,  96.,  32.,  96.]], dtype=float32)}

        self.rpn_model = RPN(anchor_num=self.anchor_num, feature_in=256, feature_out=256)

        self.instance_size = 255    # for search size
        self.template_size = 127
        self.segment_threshold = 0.35

        self.features = ResnetDown()
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

        # torch.jit.script(self.features)
        # torch.jit.script(self.mask_model)
        # torch.jit.script(self.refine_model)

        window = torch.hamming_window(self.score_size)
        window = window.view(self.score_size, 1) * window.view(1, self.score_size)
        self.window = window.flatten().repeat(self.anchor_num)
        # self.window.size: (225 * 25 * 5) = 3125

        # Set standard template features
        self.zf = torch.zeros(1, 256, 7, 7)

        # Reasonable setting ?
        self.image_height = 256
        self.image_width = 256
        self.background = torch.zeros([0, 0, 0])
        self.full_feature = None
        self.corr_feature = None

        # rc -- row center, cc -- column center
        self.target_rc = 64
        self.target_cc = 64
        self.target_h = 127
        self.target_w = 127

        self.reset_mode(is_training=False)

    def reset_mode(self, is_training=False):
        if is_training:
            self.features.train()
            self.mask_model.train()
            self.refine_model.train()
        else:
            self.features.eval()
            self.mask_model.eval()
            self.refine_model.eval()

    def set_image_size(self, h, w):
        self.image_height = h
        self.image_width = w

    def set_target(self, rc, cc, h, w):
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
        with torch.no_grad():
            _, self.zf = self.features(template)

    def set_background(self, bgcolor):
        self.background = bgcolor

    def track_mask(self, search):
        # (Pdb) search.size() -- torch.Size([1, 3, 255, 255])
        self.full_feature, self.search = self.features(search)
        # (Pdb) self.zf.size() -- torch.Size([1, 256, 7, 7])
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, self.search)
        self.corr_feature, rpn_pred_mask = self.mask_model(self.zf, self.search)
        
        rpn_pred_score = self.convert_score(rpn_pred_cls)
        rpn_pred_bbox = self.convert_bbox(rpn_pred_loc)

        return rpn_pred_score, rpn_pred_bbox, rpn_pred_mask

    def track_refine(self, pos):
        with torch.no_grad():
            rpn_pred_mask = self.refine_model(self.full_feature, self.corr_feature, pos=pos)
        return rpn_pred_mask.sigmoid()

    def convert_bbox(self, delta):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0] # x
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1] # y
        delta[2, :] = np.exp(delta[2, :]) * self.anchor[:, 2]    # w
        delta[3, :] = np.exp(delta[3, :]) * self.anchor[:, 3]    # h
        return delta

    def convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score
