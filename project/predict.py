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
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import get_model, model_device, model_setenv

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', type=str, default="models/image_siammask.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="tennis/*.jpg", help="input image")
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    model_setenv()
    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    # print(model)
    # script_model = torch.jit.script(model)

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total = len(image_filenames))

    spend_time = 0
    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")

        start_time = time.time()
        input_tensor = totensor(image).unsqueeze(0).to(device)

        input_tensor = input_tensor * 255.0

        if index == 0:
            x, y, h, w = 300, 100, 280, 180
            r, c = y + h/2, x + w/2
            model.set_reference(input_tensor, r, c, h, w)
            continue
        else:
            with torch.no_grad():
                mask = model(input_tensor).squeeze()

            input_tensor[:, 0, :, :] = (mask > 0) * 255.0 + (mask == 0) * input_tensor[:, 0, :, :]

        input_tensor = input_tensor / 255.0
        input_tensor = input_tensor.squeeze(0).cpu()
        spend_time += (time.time() - start_time)

        toimage(input_tensor).save("{}/{}".format(args.output, os.path.basename(filename)))
    
    spend_time = spend_time*1000/len(image_filenames)
    print("Per frame spend {} ms, fps: {}".format(spend_time, 1000.0/spend_time))