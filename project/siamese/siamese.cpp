/************************************************************************************
***
***    Copyright Dell 2021, All Rights Reserved.
***
***    File Author: Dell, 2021年 07月 17日
***
************************************************************************************/

#include <iostream>
#include <torch/extension.h>

// Anchor layout: 25x25x5
#define ANCHOR_NUMBERS 25

#define TEMPLATE_SIZE 127
#define INSTANCE_SIZE 255

namespace F = torch::nn::functional;
#define CheckPoint(fmt, arg...)                                                \
  printf("# CheckPoint: %d(%s): " fmt "\n", (int)__LINE__, __FILE__, ##arg)

float get_scale_size(float h, float w) {
  double pad = (w + h) * 0.5;
  double sz2 = (w + pad) * (h + pad);

  return (float)sqrt(sz2);
}

torch::Tensor get_scale_tensor(const torch::Tensor &h, const torch::Tensor &w) {
  torch::Tensor pad = (h + w) * 0.5;
  torch::Tensor sz2 = (h + pad) * (w + pad);
  return torch::sqrt(sz2);
}

torch::Tensor get_max_change(const torch::Tensor &r) {
  return torch::max(r, 1.0 / r);
}

std::vector<int64_t> get_range_pad(int64_t y, int64_t d, int64_t max) {
  std::vector<int64_t> result;
  int64_t y1 = y - d / 2;
  int64_t y2 = y1 + d - 1;
  int64_t pad1 = (-y1 > 0) ? -y1 : 0;
  int64_t pad2 = (y2 - max + 1 > 0) ? y2 - max + 1 : 0;
  y1 += pad1;
  y2 += pad1;

  result.push_back(y1);
  result.push_back(y2);
  result.push_back(pad1);
  result.push_back(pad2);

  return result;
}

torch::Tensor sub_window(const torch::Tensor &image,
                         const torch::Tensor &target) {
  // 1. image.dim() == 4 && kNearest mode
  // 2. target target.dim() == 1 with 4 elements, rc, cc, h, w

  float *target_data = target.data_ptr<float>();
  int64_t rc = (int64_t)target_data[0];
  int64_t cc = (int64_t)target_data[1];
  int64_t e = (int64_t)get_scale_size(target_data[2], target_data[3]); // h, w

  int64_t height = (int64_t)image.size(2);
  int64_t width = (int64_t)image.size(3);

  // y1, y2, pad1, pad2
  std::vector<int64_t> top_bottom_pad = get_range_pad(rc, e, height);
  std::vector<int64_t> left_right_pad = get_range_pad(cc, e, width);

  // Padding, F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
  // ==> mode='constant', value=0
  std::vector<int64_t> padding;
  padding.push_back(left_right_pad.at(2)); // left
  padding.push_back(left_right_pad.at(3)); // right
  padding.push_back(top_bottom_pad.at(2)); // top
  padding.push_back(top_bottom_pad.at(3)); // bottom
  torch::Tensor pad_data =
      F::pad(image, F::PadFuncOptions(padding).mode(torch::kConstant));

  // Slice, pad_data[:, :, y1 : y2 + 1, x1 : x2 + 1]
  int64_t y1 = top_bottom_pad.at(0);
  int64_t y2 = top_bottom_pad.at(1);
  int64_t x1 = left_right_pad.at(0);
  int64_t x2 = left_right_pad.at(1);

  return pad_data.slice(2, y1, y2 + 1).slice(3, x1, x2 + 1);
}

torch::Tensor anchor_bbox(const torch::Tensor &image,
                          const torch::Tensor &target,
                          const torch::Tensor &anchor) {
  float image_height = (float)image.size(2);
  float image_width = (float)image.size(3);

  float *target_data = target.data_ptr<float>();
  float target_rc = target_data[0];
  float target_cc = target_data[1];
  float target_e = get_scale_size(target_data[2] /*h*/, target_data[3] /*w*/);

  float *anchor_data = anchor.data_ptr<float>();
  float anchor_dr =
      (anchor_data[0] - 4.0) * 8.0 / INSTANCE_SIZE; // delta row, ratio
  float anchor_dc =
      (anchor_data[1] - 4.0) * 8.0 / INSTANCE_SIZE; // delta col, ratio

  // 0.5 move center from [0, 1.0] --> [-0.5, 0.5]
  float rc = target_rc / target_e + anchor_dr - 0.5;
  float cc = target_cc / target_e + anchor_dc - 0.5;

  torch::Tensor bbox = torch::zeros({4}, torch::dtype(torch::kFloat32));
  float *bbox_data = bbox.data_ptr<float>();

  bbox_data[0] = -cc * INSTANCE_SIZE;                     // x
  bbox_data[1] = -rc * INSTANCE_SIZE;                     // y
  bbox_data[2] = image_width / target_e * INSTANCE_SIZE;  // w
  bbox_data[3] = image_height / target_e * INSTANCE_SIZE; // h

  return bbox;
}

torch::Tensor affine_theta(const torch::Tensor &mask,
                           const torch::Tensor &bbox) {
  //     https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/utils/net_utils.py
  //     affine input: (x1,y1,x2,y2)
  //     [  x2-x1             x1 + x2 - W + 1  ]
  //     [a=-----      0    c=---------------  ]
  //     [  W - 1                  W - 1       ]
  //     [                                     ]
  //     [           y2-y1    y1 + y2 - H + 1  ]
  //     [    0    b=-----  d=---------------  ]
  //     [           H - 1         H - 1      ]
  //

  float *bbox_data = bbox.data_ptr<float>();
  float x1 = bbox_data[0];
  float y1 = bbox_data[1];
  float x2 = x1 + bbox_data[2];
  float y2 = y1 + bbox_data[3];
  float H = mask.size(2) - 1.0;
  float W = mask.size(3) - 1.0;

  torch::Tensor theta = torch::zeros({2, 3}, torch::dtype(torch::kFloat32));
  float *theta_data = theta.data_ptr<float>();
  theta_data[0] = (x2 - x1) / W;
  theta_data[1] = 0.0;
  theta_data[2] = (x1 + x2 - W) / W;
  theta_data[3] = 0.0;
  theta_data[4] = (y2 - y1) / H;
  theta_data[5] = (y1 + y2 - H) / H;

  return theta;
}

torch::Tensor best_anchor(const torch::Tensor &score, const torch::Tensor &bbox,
                          torch::Tensor &target) {

  /* bbox format: bbox.size() -- [4, 3125]
   *    x --- x1, x2, ...
   *    y --- y1, y2, ...
   *    w --- w1, w2, ...
   *    h --- h1, h2, ...
   */
  torch::Tensor bbox_h = bbox.slice(0, 3, 4); // bbox[3, :]
  torch::Tensor bbox_w = bbox.slice(0, 2, 3); // bbox[2, :]

  float *target_data = target.data_ptr<float>();
  float target_r = target_data[2] / target_data[3];
  float target_e = get_scale_size(target_data[2] /*h*/, target_data[3] /*w*/);
  float template_h = TEMPLATE_SIZE * (target_data[2] / target_e);
  float template_w = TEMPLATE_SIZE * (target_data[3] / target_e);
  float template_e = get_scale_size(template_h, template_w);

  // bbox size change from template
  torch::Tensor s_c =
      get_max_change(get_scale_tensor(bbox_h, bbox_w) / template_e);
  // bbox h/w ratio change from target
  torch::Tensor r_c = get_max_change((bbox_h / bbox_w) / target_r);

  torch::Tensor penalty =
      torch::exp(-(s_c * r_c - 1.0) * 0.04); // penalty_k == 0.04
  torch::Tensor penalty_score = (penalty * score).cpu();

  torch::Tensor argmax = torch::argmax(penalty_score).cpu();
  int64_t *argmax_data = argmax.data_ptr<int64_t>();
  float *penalty_score_data = penalty_score.data_ptr<float>();

  int64_t best_id = argmax_data[0];
  float best_lr = penalty_score_data[best_id];

  int64_t left = best_id % (ANCHOR_NUMBERS * ANCHOR_NUMBERS);
  int64_t anchor_r = (int64_t)(left / ANCHOR_NUMBERS);
  int64_t anchor_c = (int64_t)(left % ANCHOR_NUMBERS);

  // Scale best bbox for new target
  torch::Tensor best_bbox =
      bbox.slice(1, best_id, best_id + 1).cpu(); // bbox[:, bestid]
  torch::Tensor scale_best_bbox = best_bbox / TEMPLATE_SIZE * target_e;
  float *scale_best_bbox_data = scale_best_bbox.data_ptr<float>();

  float new_target_rc = target_data[0] + scale_best_bbox_data[1]; // rc
  float new_target_cc = target_data[1] + scale_best_bbox_data[0]; // cc
  float new_target_h =
      target_data[2] * (1.0 - best_lr) + scale_best_bbox_data[3] * best_lr; // h
  float new_target_w =
      target_data[3] * (1.0 - best_lr) + scale_best_bbox_data[2] * best_lr; // w
  if (new_target_h < 10)
    new_target_h = 10;
  if (new_target_w < 10)
    new_target_w = 10;

  // Save anchor
  torch::Tensor anchor = torch::zeros({6}, torch::dtype(torch::kFloat32));
  float *anchor_data = anchor.data_ptr<float>();
  anchor_data[0] = (float)anchor_r;
  anchor_data[1] = (float)anchor_c;

  // For new target
  anchor_data[2] = new_target_rc;
  anchor_data[3] = new_target_cc;
  anchor_data[4] = new_target_h;
  anchor_data[5] = new_target_w;

  return anchor;
}

torch::Tensor anchor_patch(const std::vector<torch::Tensor> &full_feature,
                           const torch::Tensor &corr_feature,
                           const torch::Tensor &anchor, int no) {
  float *anchor_data = anchor.data_ptr<float>();
  int64_t anchor_r = (int64_t)anchor_data[0];
  int64_t anchor_c = (int64_t)anchor_data[1];

  if (no == 0) {
    torch::Tensor p0_pad =
        F::pad(full_feature.at(0),
               F::PadFuncOptions({16, 16, 16, 16}).mode(torch::kReplicate));
    torch::Tensor p0 = p0_pad.slice(2 /*dim*/, 4 * anchor_r, 4 * anchor_r + 61)
                           .slice(3 /*dim*/, 4 * anchor_c, 4 * anchor_c + 61);
    return p0;
  }

  if (no == 1) {
    torch::Tensor p1_pad = F::pad(
        full_feature.at(1), F::PadFuncOptions({8, 8, 8, 8}).mode(torch::kReplicate));
    torch::Tensor p1 = p1_pad.slice(2 /*dim*/, 2 * anchor_r, 2 * anchor_r + 31)
                           .slice(3 /*dim*/, 2 * anchor_c, 2 * anchor_c + 31);
    return p1;
  }

  if (no == 2) {
    torch::Tensor p2_pad = F::pad(
        full_feature.at(2), F::PadFuncOptions({4, 4, 4, 4}).mode(torch::kReplicate));
    torch::Tensor p2 = p2_pad.slice(2 /*dim*/, 1 * anchor_r, 1 * anchor_r + 15)
                           .slice(3 /*dim*/, 1 * anchor_c, 1 * anchor_c + 15);
    return p2;
  }

  // no == 3
  torch::Tensor p3 = corr_feature.slice(2 /*dim*/, anchor_r, anchor_r + 1)
                         .slice(3 /*dim*/, anchor_c, anchor_c + 1);

  return p3;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sub_window", sub_window, "Subwindow Function");
  m.def("anchor_bbox", anchor_bbox, "Get Anchor Bunding Box");
  m.def("affine_theta", affine_theta, "Get Affine Theta");
  m.def("best_anchor", best_anchor, "Find Best Anchor");
  m.def("anchor_patch", anchor_patch, "Get Anchor's Pyramid Features");
}
