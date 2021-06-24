# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


if __name__ == '__main__':
    model = DepthCorr(256, 256, 10)
    script_model = torch.jit.script(model)

    print("--------------------------------------------------")
    print(script_model.code)
    print("--------------------------------------------------")
    print(script_model.graph)
