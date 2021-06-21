# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.anchors import Anchors
from typing import Dict, List, Tuple
import pdb

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
        # self = SiameseTracker(
        #   (upSample): UpsamplingBilinear2d(size=[127, 127], mode=bilinear)
        # )
        # anchors = {'stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'round_dight': 0}
        # o_sz = 127
        # g_sz = 127

    def feature_extractor(self, x):
        pdb.set_trace()
        return self.features(x)

    def rpn(self, template, search):
        # (Pdb) type(template), type(search)
        # (<class 'torch.Tensor'>, <class 'torch.Tensor'>)

        # (Pdb) template.size() -- torch.Size([1, 256, 7, 7])
        # (Pdb) search.size() -- torch.Size([1, 256, 31, 31])

        pred_cls, pred_loc = self.rpn_model(template, search)
        # (Pdb) pp pred_cls.size() -- torch.Size([1, 10, 25, 25])
        # (Pdb) pp pred_loc.size() -- torch.Size([1, 20, 25, 25])
        return pred_cls, pred_loc

    def mask(self, template, search):
        pdb.set_trace()

        pred_mask = self.mask_model(template, search)
        return pred_mask

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(template_feature, search_feature)  # (b, 256, w, h)
        pdb.set_trace()

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

    def forward(self, input: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
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
        pdb.set_trace()

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

        # if self.training:
        #     rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean, iou_acc_5, iou_acc_7 = \
        #         self._add_rpn_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
        #                            rpn_pred_cls, rpn_pred_loc, rpn_pred_mask)
        #     outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
        #     outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        pdb.set_trace()

        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)

        pdb.set_trace()

        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        pdb.set_trace()

        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)

        pdb.set_trace()

        return rpn_pred_cls, rpn_pred_loc
