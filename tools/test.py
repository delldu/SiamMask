# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
# import logging
import numpy as np
import cv2
from PIL import Image
# from os import makedirs
from os.path import join, isdir, isfile

# from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import math
import torch
# from torch.autograd import Variable
import torch.nn.functional as F

# from utils.anchors import Anchors
# from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config

import pdb


parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask', action='store_true', help='whether use mask output')
parser.add_argument('--refine', action='store_true', help='whether use mask refine output')
parser.add_argument('--dataset', dest='dataset', default='VOT2018', choices=dataset_zoo,
                    help='datasets')
parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
parser.add_argument('--save_mask', action='store_true', help='whether use save mask for davis')
parser.add_argument('--gt', action='store_true', help='whether use gt rect for davis (Oracle)')
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

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    # type(im) -- <class 'numpy.ndarray'>, im.shape -- (480, 854, 3)

    # avg_chans = np.mean(im, axis=(0, 1))
    height, width, chan = im.shape

    # (Pdb) type(im) -- <class 'numpy.ndarray'>, (Pdb) im.shape -- (480, 854, 3), range: [0, 255], uint8

    x1, x2, left_pad, right_pad = get_range_pad(pos[0], original_sz, width)
    y1, y2, top_pad, bottom_pad = get_range_pad(pos[1], original_sz, height)

    if any([top_pad, bottom_pad, left_pad, right_pad]):
        # print("CheckPoint 1 ...")
        te_im = np.zeros((height + top_pad + bottom_pad, width + left_pad + right_pad, chan), np.uint8)
        te_im[top_pad:top_pad + height, left_pad:left_pad + width, :] = im
        # if top_pad:
        #     te_im[0:top_pad, left_pad:left_pad + width, :] = avg_chans
        # if bottom_pad:
        #     te_im[height + top_pad:, left_pad:left_pad + width, :] = avg_chans
        # if left_pad:
        #     te_im[:, 0:left_pad, :] = avg_chans
        # if right_pad:
        #     te_im[:, width + left_pad:, :] = avg_chans

        te_im[0:top_pad, left_pad:left_pad + width, :] = avg_chans
        te_im[height + top_pad:, left_pad:left_pad + width, :] = avg_chans
        te_im[:, 0:left_pad, :] = avg_chans
        te_im[:, width + left_pad:, :] = avg_chans

        im_patch_original = te_im[y1:y2 + 1, x1:x2 + 1, :]
    else:
        # print("CheckPoint 2 ...")
        im_patch_original = im[y1:y2 + 1, x1:x2 + 1, :]

    # np.array_equal(model_sz, original_sz) -- False
    # if not np.array_equal(model_sz, original_sz):
    #     im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    # else:
    #     im_patch = im_patch_original
    im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))

    # (Pdb) out_mode -- 'torch'
    # (Pdb) im_patch.shape -- (127, 127, 3),  (Pdb) type(im_patch) -- <class 'numpy.ndarray'>

    # x = im_to_torch(im_patch)
    # x.size() -- torch.Size([3, 127, 127]),  (Pdb) type(x) -- <class 'torch.Tensor'>, rang: [0.0, 255.0]

    return im_to_torch(im_patch)


def get_scale_size(h, w):
    # hc = h + (h + w)/2
    # wc = w + (h + w)/2
    # s = sqrt(hc * wc)

    return math.sqrt((3 * h + w) * (3 * w + h))/2


def TrackingStart(model, im, target_pos, target_size, device='cpu'):
    # (Pdb) type(im), im.min(), im.max(), im.shape
    # (<class 'numpy.ndarray'>, 0, 255, (480, 854, 3))

    state = dict()
    state['image_height'] = im.shape[0]
    state['image_width'] = im.shape[1]

    avg_chans = np.mean(im, axis=(0, 1))
    # x = torch.from_numpy(im)
    # avg_chans = x.mean(dim = 0, keepdim=False).mean(dim = 0, keepdim=False)


    s_z = round(get_scale_size(target_size[0], target_size[1]))

 
    # initialize the exemplar

    # def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    z_crop = get_subwindow_tracking(im, target_pos, model.template_size, s_z, avg_chans)

    # pdb.set_trace()

    z = z_crop.unsqueeze(0)
    # (Pdb) pp z.shape -- torch.Size([1, 3, 127, 127]), format range: [0, 255]

    model.set_template(z.to(device))

    # (Pdb) avg_chans
    # array([ 96.94782641, 114.56385148, 141.78324551])

    state['avg_chans'] = avg_chans
    state['target_pos'] = target_pos
    state['target_size'] = target_size
    return state


def TrackingDoing(model, state, im, mask_enable=False, device='cpu'):
    avg_chans = state['avg_chans']
    # type(state['avg_chans']) -- <class 'numpy.ndarray'>
    # (Pdb) state['avg_chans'].shape -- (3,)

    window = model.window.numpy()

    target_pos = state['target_pos']
    # (Pdb) state['target_pos'] -- array([390., 240.])
    # (Pdb) state['target_pos'].shape -- (2,)

    target_size = state['target_size']
    # (Pdb) state['target_size'] -- array([180, 280])
    # (Pdb) state['target_size'].shape -- (2,)

    # mask_enable = True
    s_x = get_scale_size(target_size[0], target_size[1])

    scale_x = model.template_size / s_x
    # s_x -- 457.27， scale_x -- 0.2777325006938416

    # p.instance_size -- 255, p.exemplar_size -- 127
    d_search = (model.instance_size - model.template_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
    # (Pdb) crop_box -- [-69.0, -219.0, 918, 918]

    # def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    x_crop = get_subwindow_tracking(im, target_pos, model.instance_size, round(s_x), avg_chans).unsqueeze(0)
    # (Pdb) pp x_crop.shape -- torch.Size([1, 3, 255, 255])

    score, delta, mask = model.track_mask(x_crop.to(device))
    # (Pdb) pp score.size()-- (torch.Size([1, 10, 25, 25]),
    # delta.size() -- torch.Size([1, 20, 25, 25])
    # mask.size() --  torch.Size([1, 3969, 25, 25]))

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:,
            1].cpu().numpy()
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

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_size*scale_x
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))  # scale penalty
    r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    # p.penalty_k -- 0.04
    penalty = np.exp(-(r_c * s_c - 1) * model.penalty_k)
    pscore = penalty * score

    # pp p.window_influence -- 0.4
    window_influence = 0.4
    pscore = pscore * (1 - window_influence) + window * window_influence

    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id]  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_size[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_size[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_size = np.array([res_w, res_h])

    # for Mask Branch
    # pp mask_enable -- True
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, model.score_size, model.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        # pp model.template_size -- 127
        mask = model.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
            model.template_size, model.template_size).cpu().data.numpy()

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop
        # pp p.instance_size -- 255
        s = crop_box[2] / model.instance_size
        # pp p.base_size -- 8
        # (Pdb) pp p.total_stride -- 8
        # pp p.exemplar_size -- 127

        sub_box = [crop_box[0] + (delta_x - model.anchors["base_size"] / 2) * model.anchors["stride"] * s,
                   crop_box[1] + (delta_y - model.anchors["base_size"] / 2) * model.anchors["stride"] * s,
                   s * model.template_size, s * model.template_size]
        s = model.template_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['image_width'] * s, state['image_height'] * s]
        mask_in_img = crop_back(mask, back_box, (state['image_width'], state['image_height']))
        # mask.shape -- (127, 127)
        # (Pdb) back_box -- [-44.833333333333336, -3.1666666666666683, 237.22222222222223, 133.33333333333334]
        # (Pdb) mask_in_img.shape -- (480 -- state['image_height'], 854 -- width)

        # pp p.segment_threshold -- 0.35
        target_mask = (mask_in_img > model.segment_threshold).astype(np.uint8)

        # cv2.__version__ -- '4.4.0' ==> cv2.__version__[-5] == '4'
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])

    # type(state['image_width']) -- <class 'int'>
    target_pos[0] = max(0, min(state['image_width'], target_pos[0]))
    target_pos[1] = max(0, min(state['image_height'], target_pos[1]))
    target_size[0] = max(10, min(state['image_width'], target_size[0]))
    target_size[1] = max(10, min(state['image_height'], target_size[1]))

    state['target_pos'] = target_pos
    state['target_size'] = target_size
    state['score'] = score[best_pscore_id]
    state['mask'] = mask_in_img

    state['ploygon'] = rbox_in_img
    return state
