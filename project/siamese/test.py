"""Test siamese."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 07月 17日
# ***
# ************************************************************************************/
#
import torch
import argparse
import os
import pdb  # For debug
import time

import numpy as np
import onnx
import onnxruntime

import torch
from torch import nn
from torch.autograd import Function


# Our module!
import siamese_cpp

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print(
        "Onnx Model Engine: ",
        onnx_model.get_providers(),
        "Device: ",
        onnxruntime.get_device(),
    )

    return onnx_model


def onnx_forward(onnx_model, input):
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


class SubWindowFunction(Function):
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)        
        output = siamese_cpp.sub_window(input, target)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        # Set grad as 1.0
        grad_input = torch.ones_like(input)
        grad_pos = torch.ones_like(target)

        return (grad_input, grad_pos)

    @staticmethod
    def symbolic(g, input, target):
        return g.op("siamese::sub_window", input, target) 


class SubWindow(torch.nn.Module):
    # def __init__(self):
    #     super(SubWindow, self).__init__()

    def forward(self, input, target):
        output =  SubWindowFunction.apply(input, target)
        return output


if __name__ == "__main__":
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output folder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = SubWindow()
    model.eval()

    onnx_file_name = "/tmp/test.onnx"

    dummy_input = torch.randn(1, 3, 960, 1024)
    dummy_target = torch.Tensor([240, 390, 127, 255])

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    #
    def export_onnx():
        """Export onnx model."""

        # 1. Create and load model.

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input", "target"]
        output_names = ["output"]


        torch.onnx.export(
            model,
            (dummy_input, dummy_target),
            onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            export_params=True,
        )

        # 3. Optimize model
        print("Checking model ...")
        onnx_model = onnx.load(onnx_file_name)
        onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('/tmp/test.onnx')"


    def verify_onnx():
        """Verify onnx model."""

        onnxruntime_engine = onnx_load(onnx_file_name)

        with torch.no_grad():
            torch_output = model(dummy_input, dummy_target)

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input),
            onnxruntime_engine.get_inputs()[1].name: to_numpy(dummy_target),
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print(
            "Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(
                onnx_file_name
            )
        )

    export_onnx()
    verify_onnx()
