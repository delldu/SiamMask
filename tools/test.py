# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
from os import makedirs
from os.path import join, isdir, isfile

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import load_dataset, dataset_zoo

import math
import torch
# from torch.autograd import Variable
import torch.nn.functional as F

from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig

from utils.config_helper import load_config

import pdb


thrs = np.arange(0.3, 0.5, 0.05)

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


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    # H, W, C --> C, H, W, Range: 0-255
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    # pdb.set_trace()

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    # (Pdb) type(im) -- <class 'numpy.ndarray'>, (Pdb) im.shape -- (480, 854, 3), range: [0, 255], uint8

    # pdb.set_trace()

    # any([top_pad, bottom_pad, left_pad, right_pad]) -- False
    # [top_pad, bottom_pad, left_pad, right_pad] -- [0, 0, 0, 0]
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        # print("CheckPoint 1 ...")
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        # print("CheckPoint 2 ...")
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    # np.array_equal(model_sz, original_sz) -- False
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    # (Pdb) out_mode -- 'torch'
    # (Pdb) im_patch.shape -- (127, 127, 3),  (Pdb) type(im_patch) -- <class 'numpy.ndarray'>

    # x = im_to_torch(im_patch)
    # x.size() -- torch.Size([3, 127, 127]),  (Pdb) type(x) -- <class 'torch.Tensor'>, rang: [0.0, 255.0]

    return im_to_torch(im_patch)


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


def get_scale_size(h, w):
    # hc = h + (h + w)/2
    # wc = w + (h + w)/2
    # s = sqrt(hc * wc)

    return math.sqrt((3 * h + w) * (3 * w + h))/2



def TrackingStart(model, im, target_pos, target_size, hp=None, device='cpu'):
    # hp = {'instance_size': 255, 
    #     'base_size': 8, 
    #     'out_size': 127, 
    #     'seg_thr': 0.35, 
    #     'penalty_k': 0.04, 
    #     'window_influence': 0.4,
    #     'lr': 1.0}

    # (Pdb) type(im), im.min(), im.max(), im.shape
    # (<class 'numpy.ndarray'>, 0, 255, (480, 854, 3))

    state = dict()
    state['image_height'] = im.shape[0]
    state['image_width'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)
    p.renew()
    # p.scales = model.anchors['scales']
    # p.ratios = model.anchors['ratios']
    # p.anchor_num = model.anchor_num
    p.anchor = generate_anchor(model.anchors, p.score_size)

    # (Pdb) print(p.__dict__)
    # {'instance_size': 255, 
    # 'base_size': 8, 
    # 'out_size': 127, 
    # 'seg_thr': 0.35, 'penalty_k': 0.04, 
    # 'window_influence': 0.4, 'lr': 1.0, 
    # 'total_stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 
    # 'scales': [8], 'round_dight': 0, 
    # 'score_size': 25, 'anchor_num': 5, 
    # 'anchor': array([[-96., -96., 104.,  32.],
    #        [-88., -96., 104.,  32.],
    #        [-80., -96., 104.,  32.],
    #        ...,
    #        [ 80.,  96.,  32.,  96.],
    #        [ 88.,  96.,  32.,  96.],
    #        [ 96.,  96.,  32.,  96.]], dtype=float32)}
    # pp p.exemplar_size -- 127

    avg_chans = np.mean(im, axis=(0, 1))
    # pdb.set_trace()

    # # pp p.context_amount -- 0.5
    # wc_z = target_size[0] + p.context_amount * sum(target_size)
    # hc_z = target_size[1] + p.context_amount * sum(target_size)
    # # (Pdb) target_size -- array([180, 280])
    # # (Pdb) wc_z, hc_z -- (410.0, 510.0)
    # s_z = round(np.sqrt(wc_z * hc_z))

    s_z = round(get_scale_size(target_size[0], target_size[1]))

    # pp s_z -- 457
 
    # initialize the exemplar

    # def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    # pdb.set_trace()

    z = z_crop.unsqueeze(0)
    # (Pdb) pp z.shape -- torch.Size([1, 3, 127, 127]), format range: [0, 255]

    model.template(z.to(device))

    # p.windowing == 'cosine'
    # if p.windowing == 'cosine':
    window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    # p window.shape -- (3125,)
    # (Pdb) pp window.max(), window.min(), window.mean() -- (1.0, 0.0, 0.2304)

    # (Pdb) avg_chans
    # array([ 96.94782641, 114.56385148, 141.78324551])

    state['p'] = p
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_size'] = target_size
    return state


def TrackingDoing(model, state, im, mask_enable=False, device='cpu'):
    p = state['p']
    # (Pdb) print(state['p'])
    # <utils.tracker_config.TrackerConfig object at 0x7f4b95a0bd30>
    # (Pdb) print(state['p'].__dict__)
    # {'instance_size': 255, 'base_size': 8, 'out_size': 127, 'seg_thr': 0.35, 'penalty_k': 0.04, 'window_influence': 0.4, 'lr': 1.0, 'total_stride': 8, 'ratios': [0.33, 0.5, 1, 2, 3], 'scales': [8], 'round_dight': 0, 'score_size': 25, 'anchor_num': 5, 'anchor': array([[-96., -96., 104.,  32.],
    #        [-88., -96., 104.,  32.],
    #        [-80., -96., 104.,  32.],
    #        ...,
    #        [ 80.,  96.,  32.,  96.],
    #        [ 88.,  96.,  32.,  96.],
    #        [ 96.,  96.,  32.,  96.]], dtype=float32)}

    avg_chans = state['avg_chans']
    # type(state['avg_chans']) -- <class 'numpy.ndarray'>
    # (Pdb) state['avg_chans'].shape -- (3,)

    window = state['window']
    # (Pdb) state['window'] -- array([0., 0., 0., ..., 0., 0., 0.])
    # (Pdb) state['window'].shape -- (3125,)

    target_pos = state['target_pos']
    # (Pdb) state['target_pos'] -- array([390., 240.])
    # (Pdb) state['target_pos'].shape -- (2,)

    target_size = state['target_size']
    # (Pdb) state['target_size'] -- array([180, 280])
    # (Pdb) state['target_size'].shape -- (2,)

    # mask_enable = True
    # wc_x = target_size[1] + p.context_amount * sum(target_size)
    # hc_x = target_size[0] + p.context_amount * sum(target_size)
    # s_x = np.sqrt(wc_x * hc_x)
    s_x = get_scale_size(target_size[0], target_size[1])

    scale_x = p.exemplar_size / s_x
    # s_x -- 457.27， scale_x -- 0.2777325006938416

    # p.instance_size -- 255, p.exemplar_size -- 127
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
    # (Pdb) crop_box -- [-69.0, -219.0, 918, 918]

    # def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0)
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

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

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
    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    # pp p.window_influence -- 0.4
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_size[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_size[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_size = np.array([res_w, res_h])

    # for Mask Branch
    # pp mask_enable -- True
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(best_pscore_id, (5, p.score_size, p.score_size))
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        # pp p.out_size -- 127
        mask = model.track_refine((delta_y, delta_x)).to(device).sigmoid().squeeze().view(
            p.out_size, p.out_size).cpu().data.numpy()

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
        s = crop_box[2] / p.instance_size
        # pp p.base_size -- 8
        # (Pdb) pp p.total_stride -- 8
        # pp p.exemplar_size -- 127

        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['image_width'] * s, state['image_height'] * s]
        mask_in_img = crop_back(mask, back_box, (state['image_width'], state['image_height']))
        # mask.shape -- (127, 127)
        # (Pdb) back_box -- [-44.833333333333336, -3.1666666666666683, 237.22222222222223, 133.33333333333334]
        # (Pdb) mask_in_img.shape -- (480 -- state['image_height'], 854 -- width)

        # pp p.seg_thr -- 0.35
        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
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
