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
    y2 = y1 + d - 1

    pad1 = max(0, -y1)
    pad2 = max(0, y2 - maxy + 1)

    y1 = y1 + pad1
    y2 = y2 + pad1

    return y1, y2, pad1, pad2


def get_subwindow(im, target_rc, target_cc, target_size, search_size, avg_chans):
    # (Pdb) type(im) -- <class 'numpy.ndarray'>, (Pdb) im.shape -- (480, 854, 3), range: [0, 255], uint8

    height, width, chan = im.shape

    x1, x2, left_pad, right_pad = get_range_pad(target_cc, search_size, width)
    y1, y2, top_pad, bottom_pad = get_range_pad(target_rc, search_size, height)

    big = np.zeros((height + top_pad + bottom_pad, width +
                   left_pad + right_pad, chan), np.uint8)
    big[top_pad:top_pad + height, left_pad:left_pad + width, :] = im
    # big[0:top_pad, left_pad:left_pad + width, :] = avg_chans
    big[height + top_pad:, left_pad:left_pad + width, :] = avg_chans
    # big[:, 0:left_pad, :] = avg_chans
    big[:, width + left_pad:, :] = avg_chans

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

    state = dict()

    model.set_image_size(im.shape[0], im.shape[1])

    avg_chans = np.mean(im, axis=(0, 1))
    model.set_background(avg_chans)

    # x = torch.from_numpy(im)
    # avg_chans = x.mean(dim = 0, keepdim=False).mean(dim = 0, keepdim=False)

    s_z = round(get_scale_size(model.target_h, model.target_w))

    # initialize the exemplar

    z_crop = get_subwindow(im, model.target_rc, model.target_cc, model.template_size, s_z, avg_chans)

    # (Pdb) pp z_crop.shape -- torch.Size([1, 3, 127, 127]), format range: [0, 255]

    model.set_template(z_crop.to(device))

    # (Pdb) avg_chans
    # array([ 96.94782641, 114.56385148, 141.78324551])

    return state


def TrackingDoing(model, state, im, device='cpu'):
    avg_chans = model.background

    # mask_enable = True
    s_x = get_scale_size(model.target_h, model.target_w)

    scale_x = model.template_size / s_x
    # s_x -- 457.27， scale_x -- 0.2777325006938416

    # p.instance_size -- 255, p.template_size -- 127
    d_search = (model.instance_size - model.template_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [model.target_cc - round(s_x) / 2, model.target_rc - round(s_x) / 2, round(s_x), round(s_x)]
    # (Pdb) crop_box -- [-69.0, -219.0, 918, 918]

    x_crop = get_subwindow(im, model.target_rc, model.target_cc, model.instance_size, round(s_x), avg_chans)
    # (Pdb) pp x_crop.shape -- torch.Size([1, 3, 255, 255])

    score, delta, mask = model.track_mask(x_crop.to(device))
    # score.size()-- (torch.Size([1, 10, 25, 25]),
    # delta.size() -- torch.Size([1, 20, 25, 25])
    # mask.size() --  torch.Size([1, 3969, 25, 25]))

    delta = delta.permute(1, 2, 3, 0).contiguous().view(
        4, -1).data.cpu().numpy()
    # score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
    #         1].cpu().numpy()
    score = score.view(2, -1).data[1].cpu()
    score = F.softmax(score, dim=0).numpy()

    # delta.shape -- (4, 3125)
    # score.shape -- (3125,)

    delta[0, :] = delta[0, :] * model.anchor[:, 2] + model.anchor[:, 0]
    delta[1, :] = delta[1, :] * model.anchor[:, 3] + model.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * model.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * model.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop_h = model.target_h * scale_x
    target_sz_in_crop_w = model.target_w * scale_x
    # s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz(target_sz_in_crop_h, target_sz_in_crop_w)))  # scale penalty
    r_c = change((target_sz_in_crop_w / target_sz_in_crop_h) / (delta[2, :] / delta[3, :]))  # ratio penalty

    # p.penalty_k -- 0.04
    penalty = np.exp(-(r_c * s_c - 1) * model.penalty_k)
    pscore = penalty * score

    # pp p.window_influence -- 0.4
    window_influence = 0.4
    pscore = pscore * (1 - window_influence) + model.window.numpy() * window_influence

    best_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_id] / scale_x
    lr = penalty[best_id] * score[best_id]  # lr for OTB

    # bbox format: x, y, w, h
    # update model target pos, size via prediction
    model.target_rc += pred_in_crop[1]
    model.target_cc += pred_in_crop[0]
    model.target_w = model.target_w * (1 - lr) + pred_in_crop[2] * lr
    model.target_h = model.target_h * (1 - lr) + pred_in_crop[3] * lr


    # for Mask Branch
    best_id_mask = np.unravel_index(best_id, (5, model.score_size, model.score_size))
    delta_y, delta_x = best_id_mask[1], best_id_mask[2]

    # pp model.template_size -- 127
    mask = model.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
        model.template_size, model.template_size).cpu().data.numpy()
    #  pp delta_y, delta_x -- (13, 12), mask.shape -- (127, 127)

    # out_sz -- (model.image_width, model.image_height)
    def crop_back(mask, bbox, out_sz, padding=-1):
        W = out_sz[0]
        H = out_sz[1]

        a = W / bbox[2]   # width
        b = H / bbox[3]   # height
        c = -a * bbox[0]    # x
        d = -b * bbox[1]    # y

        # Transform matrix:
        # w-a  0    x-c
        # 0    h-b  y-d

        # theta = torch.FloatTensor([[1.0/a, 0, -c/W], [0, 1.0/b, -d/H]]).unsqueeze(0)
        # grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
        # grid_0 = grid[:, :, :, 0]
        # grid_0 = grid_0 - grid_0.mean()
        # grid_0 = grid_0/(grid_0.max() + 1e-6)

        # grid_1 = grid[:, :, :, 1]
        # grid_1 = grid_1 - grid_1.mean()
        # grid_1 = grid_1/(grid_1.max() + 1e-6)

        # grid = torch.stack((grid_0, grid_1), dim=3)

        # # grid = grid - grid.mean()
        # # grid = grid/(grid.max() + 1e-6)

        # input = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        # output = F.grid_sample(input, grid, mode='bilinear', align_corners=True, padding_mode="zeros")
        # output = output.squeeze(0).squeeze(0).numpy()

        # return output

        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(mask, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)

        # (Pdb) pp mask.shape -- (127, 127)
        # (Pdb) pp bbox -- [-44.83, -11.16, 237.22, 133.33], x, y, w, h
        # (Pdb) pp out_sz -- (854, 480) w, h
        # pp crop.shape -- (480, 854), h, w
        # ==> (Pdb) pp a, b, c, d -- (3.6, 3.60, 161.4(x?), 40.20)

        return crop
    # pp p.instance_size -- 255
    # width/height is same for crop_box
    s = crop_box[2] / model.instance_size

    sub_box = [crop_box[0] + (delta_x - model.anchors["base_size"] / 2) * model.anchors["stride"] * s,
               crop_box[1] + (delta_y - model.anchors["base_size"] /
                              2) * model.anchors["stride"] * s,
               s * model.template_size, s * model.template_size]

    s = model.template_size / sub_box[2]
    back_box = [-sub_box[0] * s, -sub_box[1] * s,
                model.image_width * s, model.image_height * s]

    mask_in_img = crop_back(
        mask, back_box, (model.image_width, model.image_height))
    # mask.shape -- (127, 127)
    # (Pdb) back_box -- [-44.8, -3.1, 237.2, 133.3]
    # (Pdb) mask_in_img.shape -- (480, 854), here (480 -- state['image_height'], 854 -- width)
    target_mask = (mask_in_img > model.segment_threshold).astype(np.uint8)


    # Update for next prediction
    model.target_rc = max(0, min(model.image_height, model.target_rc))
    model.target_cc = max(0, min(model.image_width, model.target_cc))
    model.target_h = max(10, min(model.image_height, model.target_h))
    model.target_w = max(10, min(model.image_width, model.target_w))

    state['score'] = score[best_id]
    state['mask'] = target_mask

    # state['ploygon'] = rbox_in_img
    return state
