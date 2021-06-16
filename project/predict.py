"""Model predict."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 16日 星期三 18:41:24 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import get_model, model_device

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', type=str, default="models/image_siammask.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="tennis/*.png", help="input image")
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    print(model)
    pdb.set_trace()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total = len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()

        toimage(output_tensor.cpu()).save("{}/{}".format(args.output, os.path.basename(filename)))
