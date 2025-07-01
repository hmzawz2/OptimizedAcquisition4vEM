# from https://github.com/myungsub/CAIN/blob/master/utils.py,
# but removed the errenous normalization and quantization steps from computing the PSNR.

import math
import os
import shutil

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

from pytorch_msssim import ssim as calc_ssim
from pathlib import Path
import numpy as np
from einops import rearrange
from tifffile import imread, imwrite


def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims


def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

        ssim = calc_ssim(
            output[b].unsqueeze(0).clamp(0, 1),
            gt[b].unsqueeze(0).clamp(0, 1),
            data_range=1.0,
        )
        ssims.update(ssim)


def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split("+")
    for l in loss_list:
        _, loss_type = l.split("*")
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics["total"] = AverageMeter()
    return loss_specifics


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def save_checkpoint(state, directory, is_best, filename="checkpoint.pth"):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, "model_best.pth"))


def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode="train"):
    writer.add_scalar("Loss/%s/%s" % mode, loss, timestep)
    writer.add_scalar("PSNR/%s" % mode, psnr, timestep)
    writer.add_scalar("SSIM/%s" % mode, ssim, timestep)
    if mode == "train":
        writer.add_scalar("lr", lr, timestep)


def compute_gini(distances: torch.Tensor) -> torch.Tensor:
    """
    copy and modified from https://github.com/QY-H00/attention-interpolation-diffusion
    Compute the Gini index of the input distances.

    Args:
        distances: The input distances as a torch tensor.

    Returns:
        gini: The Gini index of the input distances as a torch tensor.
    """
    if not isinstance(distances, torch.Tensor):
        raise TypeError("Input must be a torch tensor")
    if distances.dim() != 1:
        raise ValueError("Input must be a one-dimensional torch tensor")
    if len(distances) < 2:
        return torch.tensor(0.0, dtype=distances.dtype, device=distances.device)  # Gini index is 0 for less than two elements

    # Sort the distances tensor
    sorted_distances, _ = torch.sort(distances)
    n = len(sorted_distances)
    mean_distance = torch.mean(sorted_distances)

    # Compute the sum of absolute differences
    sum_of_differences = torch.sum(torch.abs(sorted_distances.unsqueeze(0) - sorted_distances.unsqueeze(1)))

    # Normalize the sum of differences by the mean and the number of elements
    gini = sum_of_differences / (2 * n * n * mean_distance)
    return gini