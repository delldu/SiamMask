#include <iostream>
#include <torch/extension.h>

int64_t get_scale_size(int64_t h, int64_t w)
{
    double pad = (w + h) * 0.5;
    double sz2 = (w + pad) * (h + pad);

    return (int64_t)sqrt(sz2);
}

std::vector<int64_t> get_range_pad(int64_t y, int64_t d, int64_t max)
{
    std::vector<int64_t> result;
    int64_t y1 = y - d/2;
    int64_t y2 = y1 + d - 1;
    int64_t pad1 = (-y1 > 0)? -y1 : 0;
    int64_t pad2 = (y2 - max + 1 > 0)? y2 - max + 1: 0;
    y1 += pad1;
    y2 += pad2;

    result.push_back(y1);
    result.push_back(y2);
    result.push_back(pad1);
    result.push_back(pad2);

    return result;
}

torch::Tensor sub_window(const torch::Tensor &image, torch::Tensor target)
{
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
  torch::Tensor pad_data = F::pad(image, F::PadFuncOptions(padding).mode(torch::kReplicate));

  // Slice, pad_data[:, :, y1 : y2 + 1, x1 : x2 + 1]
  int64_t y1 = top_bottom_pad.at(0);
  int64_t y2 = top_bottom_pad.at(1);
  int64_t x1 = left_right_pad.at(0);
  int64_t x2 = left_right_pad.at(1);
  torch::Tensor patch = pad_data.slice(2, y1, y2 + 1).slice(3, x1, x2 + 1);

  return patch;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sub_window", sub_window, "Subwindow Function");
}
