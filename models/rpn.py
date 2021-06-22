# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        # (Pdb) out.size() -- torch.Size([1, 10, 25, 25])

        return out

if __name__ == '__main__':
    model = DepthCorr(256, 256, 10)
    script_model = torch.jit.script(model)

    print("--------------------------------------------------")
    print(script_model.code)
    print("--------------------------------------------------")
    print(script_model.graph)
