# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division

import argparse
import math
import pdb
# from os import makedirs
from os.path import isdir, isfile, join

import cv2
# import logging
import numpy as np
import torch
# from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from utils.bbox_helper import cxy_wh_2_rect
from utils.benchmark_helper import dataset_zoo, load_dataset
from utils.config_helper import load_config
# from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain

# from utils.anchors import Anchors
# from utils.tracker_config import TrackerConfig


parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom', ],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True,
                    help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true',
                    help='whether use mask output')
parser.add_argument('--refine', action='store_true',
                    help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt",
                    type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true',
                    help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true',
                    help='whether use gt rect for davis (Oracle)')
parser.add_argument('--video', default='', type=str, help='test special video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--debug', action='store_true', help='debug mode')


def im_to_torch(img):
    # H, W, C --> C, H, W, Range: 0-255
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.from_numpy(img).float()
    return img


def get_range_pad(y, d, maxy):
    y1 = round(y - d/2)
    y2 = round(y1 + d - 1)

    pad1 = max(0, -y1)
    pad2 = max(0, y2 - maxy + 1)

    y1 = y1 + pad1
    y2 = y2 + pad1

    return y1, y2, pad1, pad2


def get_subwindow(im, target_rc, target_cc, target_size, search_size, bg_color):
    # (Pdb) type(im) -- <class 'numpy.ndarray'>, (Pdb) im.shape -- (480, 854, 3), range: [0, 255], uint8

    height, width, chan = im.shape

    x1, x2, left_pad, right_pad = get_range_pad(target_cc, search_size, width)
    y1, y2, top_pad, bottom_pad = get_range_pad(target_rc, search_size, height)

    big = np.zeros((height + top_pad + bottom_pad, width +
                   left_pad + right_pad, chan), np.uint8)
    big[top_pad:top_pad + height, left_pad:left_pad + width, :] = im
    # big[0:top_pad, left_pad:left_pad + width, :] = bg_color
    big[height + top_pad:, left_pad:left_pad + width, :] = bg_color
    # big[:, 0:left_pad, :] = bg_color
    big[:, width + left_pad:, :] = bg_color

    patch = im_to_torch(big[y1:y2 + 1, x1:x2 + 1, :])

    return F.interpolate(patch.unsqueeze(0), size=(target_size, target_size), mode='nearest')


def get_scale_size(h, w):
    # hc = h + (h + w)/2
    # wc = w + (h + w)/2
    # s = sqrt(hc * wc)

    return math.sqrt((3 * h + w) * (3 * w + h))/2


def TrackingStart(model, im, device='cpu'):
    # (Pdb) type(im), im.min(), im.max(), im.shape
    # (<class 'numpy.ndarray'>, 0, 255, (480, 854, 3))

    model.set_image_size(im.shape[0], im.shape[1])

    bg_color = np.mean(im, axis=(0, 1))
    model.set_background(bg_color)
    # array([ 96.94782641, 114.56385148, 141.78324551])

    # x = torch.from_numpy(im)
    # bg_color = x.mean(dim = 0, keepdim=False).mean(dim = 0, keepdim=False)

    target_e = get_scale_size(model.target_h, model.target_w)

    # initialize the exemplar

    z_crop = get_subwindow(im, model.target_rc, model.target_cc, model.template_size, target_e, bg_color)

    # (Pdb) z_crop.shape -- torch.Size([1, 3, 127, 127]), format range: [0, 255]

    model.set_template(z_crop.to(device))



def TrackingDoing(model, im, device='cpu'):
    bg_color = model.background

    # target_e -- target extend
    target_e = get_scale_size(model.target_h, model.target_w)

    scale_x = model.template_size / target_e
    # target_e -- 457.27， scale_x -- 0.2777325006938416

    # p.instance_size -- 255, p.template_size -- 127
    d_search = (model.instance_size - model.template_size) / 2
    pad = d_search / scale_x
    target_e = target_e + 2 * pad

    x_crop = get_subwindow(im, model.target_rc, model.target_cc, model.instance_size, target_e, bg_color)
    # (Pdb) pp x_crop.shape -- torch.Size([1, 3, 255, 255])

    score, bbox, mask = model.track_mask(x_crop.to(device))
    # score.size()-- (torch.Size([1, 10, 25, 25]),
    # mask.size() --  torch.Size([1, 3969, 25, 25]))
    # bbox.size() -- torch.Size([1, 20, 25, 25])
    # score.shape -- (3125,)

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    # Size penalty
    # For scale_x=model.template_size/target_e, so template_h/w is virtual template size
    template_h = model.target_h * scale_x
    template_w = model.target_w * scale_x
    bbox_h = bbox[3, :]
    bbox_w = bbox[2, :]
    s_c = change(sz(bbox_w, bbox_h) / (sz(template_h, template_w)))  # scale penalty
    r_c = change(model.target_w / model.target_h) / (bbox_w / bbox_h)  # ratio penalty
    penalty = np.exp(-(r_c * s_c - 1) * 0.04)  #penalty_k == 0.04
    penalty_score = penalty * score

    # Smooth penalty score ...
    window_influence = 0.4
    penalty_score = penalty_score * (1 - window_influence) + model.window.numpy() * window_influence

    best_id = np.argmax(penalty_score)
    lr = penalty[best_id] * score[best_id]  # lr for OTB

    # for Mask Branch
    best_anchor = np.unravel_index(best_id, (model.anchor_num, model.score_size, model.score_size))
    anchor_r, anchor_c = best_anchor[1], best_anchor[2]

    # pp model.template_size -- 127
    mask = model.track_refine((anchor_r, anchor_c)).to(device).squeeze().view(
        model.template_size, model.template_size).cpu().data.numpy()
    #  pp anchor_r, anchor_c -- (13, 12), mask.shape -- (127, 127)

    def crop_back(mask, bbox, padding=-1):
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
        a = (x2 - x1)/(W - 1)
        c = (x1 + x2 - W + 1)/(W - 1)
        b = (y2 - y1)/(H - 1)
        d = (y1 + y2 - H + 1)/(H - 1)
        theta = torch.FloatTensor([[a, 0, c], [0, b, d]]).unsqueeze(0)

        H = int(model.image_height)
        W = int(model.image_width)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)

        input = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        output = F.grid_sample(input, grid, mode='bilinear', align_corners=True, padding_mode="zeros")
        output = output.squeeze(0).squeeze(0).numpy()

        return output
        # a = W / bbox[2]   # width
        # b = H / bbox[3]   # height
        # c = -a * bbox[0]    # x
        # d = -b * bbox[1]    # y

        # mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        # crop = cv2.warpAffine(mask, mapping, (W, H),
        #                       flags=cv2.INTER_LINEAR,
        #                       borderMode=cv2.BORDER_CONSTANT,
        #                       borderValue=padding)
        # return crop

    s = target_e / model.instance_size
    # e-target center: x, y format
    e_center = [model.target_cc - target_e/2, model.target_rc - target_e/2]
    # Anchor e_box center
    anchor_dr = (anchor_r - model.anchors["base_size"] / 2) * model.anchors["stride"]
    anchor_dc = (anchor_c - model.anchors["base_size"] / 2) * model.anchors["stride"]
    # Foreground box
    fg_box = [e_center[0] + anchor_dc * s, e_center[1] + anchor_dr * s, 
            s * model.template_size, s * model.template_size]

    s = model.instance_size / target_e
    bg_box = [-fg_box[0] * s, -fg_box[1] * s, model.image_width * s, model.image_height * s]

    mask_in_img = crop_back(mask, bg_box)
    # mask.shape -- (127, 127)
    # (Pdb) mask_in_img.shape -- (480, 854)
    target_mask = (mask_in_img > model.segment_threshold).astype(np.uint8)

    # Update model target
    # bbox format: x, y, w, h
    best_bbox = bbox[:, best_id] / scale_x
    model.set_target(model.target_rc + best_bbox[1], 
                    model.target_cc + best_bbox[0],
                    model.target_h * (1 - lr) + best_bbox[3] * lr,
                    model.target_w * (1 - lr) + best_bbox[2] * lr)
    model.target_clamp()

    return target_mask
