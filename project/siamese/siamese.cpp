#include <iostream>
#include <torch/extension.h>

#define TEMPLATE_SIZE 127
#define INSTANCE_SIZE 255

int64_t get_scale_size(int64_t h, int64_t w) {
  double pad = (w + h) * 0.5;
  double sz2 = (w + pad) * (h + pad);

  return (int64_t)sqrt(sz2);
}

std::vector<int64_t> get_range_pad(int64_t y, int64_t d, int64_t max) {
  std::vector<int64_t> result;
  int64_t y1 = y - d / 2;
  int64_t y2 = y1 + d - 1;
  int64_t pad1 = (-y1 > 0) ? -y1 : 0;
  int64_t pad2 = (y2 - max + 1 > 0) ? y2 - max + 1 : 0;
  y1 += pad1;
  y2 += pad2;

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

  float *data = target.data_ptr<float>();
  int64_t rc = (int64_t)data[0];
  int64_t cc = (int64_t)data[1];
  int64_t h = (int64_t)data[2];
  int64_t w = (int64_t)data[3];
  int64_t e = get_scale_size(h, w);

  int64_t height = (int64_t)image.size(2);
  int64_t width = (int64_t)image.size(2);

  // y1, y2, pad1, pad2
  std::vector<int64_t> top_bottom_pad = get_range_pad(rc, e, height);
  std::vector<int64_t> left_right_pad = get_range_pad(cc, e, width);

  // Padding, F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
  namespace F = torch::nn::functional;
  std::vector<int64_t> padding;
  padding.push_back(left_right_pad.at(2)); // left
  padding.push_back(left_right_pad.at(3)); // right
  padding.push_back(top_bottom_pad.at(2)); // top
  padding.push_back(top_bottom_pad.at(3)); // bottom
  torch::Tensor pad_data =
      F::pad(image, F::PadFuncOptions(padding).mode(torch::kReplicate));

  // Slice, pad_data[:, :, y1 : y2 + 1, x1 : x2 + 1]
  int64_t y1 = top_bottom_pad.at(0);
  int64_t y2 = top_bottom_pad.at(1);
  int64_t x1 = left_right_pad.at(0);
  int64_t x2 = left_right_pad.at(1);
  torch::Tensor patch = pad_data.slice(2, y1, y2 + 1).slice(3, x1, x2 + 1);

  return patch;
}

// def anchor_bgbox(self, anchor_r, anchor_c, target_e):
//     s = target_e / self.instance_size
//     # e-target center: x, y format
//     e_center = [self.target_cc - target_e / 2, self.target_rc - target_e / 2]
//     # Anchor e_box center
//     base_size = 8  # self.config["base_size"]
//     config_stride = 8  # self.config["stride"]
//     anchor_dr = (anchor_r - base_size / 2) * config_stride
//     anchor_dc = (anchor_c - base_size / 2) * config_stride
//     # Foreground box
//     fg_box = [
//         e_center[0] + anchor_dc * s,
//         e_center[1] + anchor_dr * s,
//         s * self.template_size,
//         s * self.template_size,
//     ]

//     s = self.instance_size / target_e
//     bg_box = [
//         int(-fg_box[0] * s),
//         int(-fg_box[1] * s),
//         int(self.image_width * s),
//         int(self.image_height * s),
//     ]
//     return bg_box
torch::Tensor anchor_bgbox(const torch::Tensor &anchor,
                           const torch::Tensor &target) {
  torch::Tensor bgbox = torch::zeros({4}, torch::dtype(torch::kFloat32));

  return bgbox;
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

  torch::Tensor theta = torch::zeros({2, 6}, torch::dtype(torch::kFloat32));

  float *bbox_data = bbox.data_ptr<float>();
  float x1 = bbox_data[0];
  float y1 = bbox_data[1];
  float x2 = x1 + bbox_data[2];
  float y2 = y1 + bbox_data[3];

  float H = mask.size(2) - 1.0;
  float W = mask.size(3) - 1.0;
  float a = (x2 - x1) / W;
  float c = (x1 + x2 - W) / W;
  float b = (y2 - y1) / H;
  float d = (y1 + y2 - H) / H;

  return torch::Tensor([ a, 0.0, c, 0.0, b, d ]).view(2, 3);
}

// # xxxx6666, siamese::best_anchor(score, bbox, target) ==> [anchor_tensor,
// new_target] def best_anchor(self, score, bbox, scale_x):
//     # Size penalty
//     # For scale_x=template_size/target_e, so template_h/w is virtual template
//     size template_h = int(self.target_h * scale_x) template_w =
//     int(self.target_w * scale_x) bbox_w = bbox[2, :] bbox_h = bbox[3, :] s_c
//     = get_max_change(
//         get_scale_tensor(bbox_w, bbox_h) / (get_scale_size(template_h,
//         template_w))
//     )  # scale penalty
//     r_c = get_max_change(
//         (self.target_w / self.target_h) / (bbox_w / bbox_h)
//     )  # ratio penalty
//     penalty = torch.exp(-(r_c * s_c - 1) * 0.04)  # penalty_k == 0.04
//     penalty_score = penalty * score

//     best_id = torch.argmax(penalty_score)
//     lr = penalty[best_id] * score[best_id]

//     # Mask Branch
//     left = best_id % (self.score_size * self.score_size)
//     anchor_r = int(left // self.score_size)
//     anchor_c = int(left % self.score_size)
//     best_bbox = bbox[:, best_id] / scale_x

//     # anchor_r, anchor_c -- (12, 13), mask.size -- (127, 127)

//     return anchor_r, anchor_c, lr, best_bbox

std::vector<torch::Tensor> best_anchor(const torch::Tensor &score,
                                      const torch::Tensor &bbox,
                                      const torch::Tensor &target) {
  torch::Tensor anchor = torch::zeros({2}, torch::dtype(torch::kFloat32));
  torch::Tensor new_target = torch::zeros({4}, torch::dtype(torch::kFloat32));

  return {anchor, new_target};
}

// siameses::anchor_patchs(full_feature, anchor_tensor) ==> [p0, p1, p2, p3]
std::vector<torch::Tensor>
anchor_patchs(const std::vector<torch::Tensor> &full_feature,
              const torch::Tensor &corr_feature, const torch::Tensor &anchor) {
  // torch::Tensor anchor = torch::zeros({2}, torch::dtype(torch::kFloat32));
  // torch::Tensor new_target = torch::zeros({4},
  // torch::dtype(torch::kFloat32));

  // p0 = F.pad(f[0], [16, 16, 16, 16])[
  //   :, :, 4 * anchor_r : 4 * anchor_r + 61, 4 * anchor_c : 4 * anchor_c + 61]
  // p1 = F.pad(f[1], [8, 8, 8, 8])[
  //     :, :, 2 * anchor_r : 2 * anchor_r + 31, 2 * anchor_c : 2 * anchor_c +
  //     31]
  // p2 = F.pad(f[2], [4, 4, 4, 4])[
  //     :, :, anchor_r : anchor_r + 15, anchor_c : anchor_c + 15]

  // p3 = corr_feature[:, :, anchor_r, anchor_c].view(-1, 256, 1, 1)

  float *anchor_data = anchor.data_ptr<float>();
  int64_t anchor_r = (int64_t)anchor_data[0];
  int64_t anchor_c = (int64_t)anchor_data[1];

  torch::Tensor p0_pad =
      F::pad(full_feature.at(0),
             F::PadFuncOptions([ 16, 16, 16, 16 ]).mode(torch::kReplicate));
  torch::Tensor p0 = p0_pad.slice(2 /*dim*/, 4 * anchor_r, 4 * anchor_r + 61)
                         .slice(3 /*dim*/, 4 * anchor_c, 4 * anchor_c + 61);

  torch::Tensor p1_pad =
      F::pad(full_feature.at(1),
             F::PadFuncOptions([ 8, 8, 8, 8 ]).mode(torch::kReplicate));
  torch::Tensor p1 = p1_pad.slice(2 /*dim*/, 2 * anchor_r, 2 * anchor_r + 31)
                         .slice(3 /*dim*/, 2 * anchor_c, 2 * anchor_c + 31);

  torch::Tensor p2_pad =
      F::pad(full_feature.at(2),
             F::PadFuncOptions([ 4, 4, 4, 4 ]).mode(torch::kReplicate));
  torch::Tensor p2 = p2_pad.slice(2 /*dim*/, 1 * anchor_r, 1 * anchor_r + 15)
                         .slice(3 /*dim*/, 1 * anchor_c, 1 * anchor_c + 15);

  torch::Tensor p3 =
      corr_feature.slice(2 /*dim*/, 1 * anchor_r, 1 * anchor_r + 15)
          .slice(3 /*dim*/, 1 * anchor_c, 1 * anchor_c + 15)
          .view(-1, 256, 1, 1);

  return {p0, p1, p2, p3};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sub_window", sub_window, "Subwindow Function");
  m.def("anchor_bgbox", anchor_bgbox, "Get Anchor Background Box");
  m.def("affine_theta", affine_theta, "Get Affine Theta");
  m.def("best_anchor", best_anchor, "Find Best Anchor");
  m.def("anchor_patchs", anchor_patchs, "Get Anchor's Pyramid Features");
}
