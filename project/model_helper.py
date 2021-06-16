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

# from models.siammask_sharp import SiamMask
# from models.features import MultiStageFeature
# from models.rpn import RPN, DepthCorr
# from models.mask import Mask

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
# from resnet import resnet50
import pdb

def center2corner(center):
    """
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2

def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
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

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
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

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
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

    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
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
        self.downsample = ResDownS(1024, 256)

    def forward(self, x):
        # x.size() -- torch.Size([1, 3, 127, 127])
        output = self.features(x)
        # (Pdb) output[0].size(), output[1].size(), output[2].size(), output[3].size()
        # (torch.Size([1, 64, 61, 61]), torch.Size([1, 256, 31, 31]), 
        # torch.Size([1, 512, 15, 15]), torch.Size([1, 1024, 15, 15]))
        p3 = self.downsample(output[-1])
        # (Pdb) p3.size() -- torch.Size([1, 256, 7, 7])

        return p3

    def forward_all(self, x):
        # (Pdb) x.size() -- torch.Size([1, 3, 255, 255])
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return output, p3


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

    def forward(self, f, corr_feature, pos=None, test=False):
        # test = True
        if test:
            p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
            p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
            p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
        else:
            p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
            if not (pos is None): p0 = torch.index_select(p0, 0, pos)
            p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
            if not (pos is None): p1 = torch.index_select(p1, 0, pos)
            p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
            if not (pos is None): p2 = torch.index_select(p2, 0, pos)

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

class Anchors:
    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.round_dight = 0
        self.image_center = 0
        self.size = 0
        self.anchor_density = 1

        self.__dict__.update(cfg)

        self.anchor_num = len(self.scales) * len(self.ratios) * (self.anchor_density**2)
        self.anchors = None  # in single position (anchor_num*4)
        self.all_anchors = None  # in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        size = self.stride * self.stride
        count = 0
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density)*anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:
                if self.round_dight > 0:
                    ws = round(math.sqrt(size*1. / r), self.round_dight)
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size*1. / r))
                    hs = int(ws * r)

                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [-w*0.5+x_offset, -h*0.5+y_offset, w*0.5+x_offset, h*0.5+y_offset][:]
                    count += 1

    def generate_all_anchors(self, im_c, size):
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


class SiamMask(nn.Module):
    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        anchors = {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'round_dight': 0}
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(self.anchors["scales"])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

        self.all_anchors = None

    # def set_all_anchors(self, image_center, size):
    #     # cx,cy,w,h
    #     if not self.anchor.generate_all_anchors(image_center, size):
    #         return
    #     all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
    #     self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
    #     self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                      rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask, label_mask_weight)

        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature)  # (b, 256, w, h)
        rpn_pred_mask = self.refine_model(feature, corr_feature)

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']

        rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = \
            self.run(template, search, softmax=self.training)

        outputs = dict()

        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask, template_feature, search_feature]

        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
                self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
                                   rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc

class SiameseTracker(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(SiameseTracker, self).__init__(**kwargs)
        # pretrain = False
        # kwargs = {'anchors': {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'round_dight': 0}}
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()
        # pretrain = False

    def template(self, template):
        # (Pdb) template.size() -- torch.Size([1, 3, 127, 127])
        self.zf = self.features(template)

    def track_mask(self, search):
        # (Pdb) search.size() -- torch.Size([1, 3, 255, 255])
        self.feature, self.search = self.features.forward_all(search)
        # (Pdb) self.zf.size() -- torch.Size([1, 256, 7, 7])
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)

        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos, test=True)
        return pred_mask

if __name__ == '__main__':
    '''Test model'''

    model = SiameseTracker()
    print(model)