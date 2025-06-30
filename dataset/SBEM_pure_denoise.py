import os
import random
from pathlib import Path

import numpy as np
import torch
from tifffile import imread, imwrite
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
from skimage.transform import rescale
from .base_dataset import BaseVolumeDataset, add_damage
import torch.utils.data.dataset as dataset
from scipy.ndimage import zoom, gaussian_filter, sobel

def R2R(vol, variance=0.05):
    vol = vol.astype("float32")
    split_strategy = random.choice([0, 1])

    if split_strategy == 0:
        hat = vol[:, 0::2, 0::2]
        tilde = vol[:, 1::2, 1::2]
    elif split_strategy == 1:
        hat = vol[:, 0::2, 1::2]
        tilde = vol[:, 1::2, 0::2]

    return hat, tilde

def add_structured_noise_numpy(img, gaussian_std_range=(0.05, 0.1), poisson_scale=255, smooth_thresh=0.05):
    gauss_std = np.random.uniform(*gaussian_std_range)
    grad_x = sobel(img, axis=1)
    grad_y = sobel(img, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = gaussian_filter(grad_mag, sigma=2)
    mask = (grad_mag < smooth_thresh).astype(np.float32)
    gauss_noise = np.random.randn(*img.shape) * gauss_std
    poisson_noise = np.random.poisson(img * poisson_scale) / poisson_scale - img
    combined_noise = gauss_noise + poisson_noise
    noisy_img = img + combined_noise * mask
    return np.clip(noisy_img, 0, 1)

def add_noise_to_slice(slice, gaussian_std_range):
    std = np.random.uniform(*gaussian_std_range)
    # gaussian noise
    gaussian_noise = np.random.normal(0, std, size=slice.shape).astype(np.float32)
    img_gaussian = slice + gaussian_noise
    img_gaussian = np.clip(img_gaussian, 0.0, 1.0)
    # poisson noise
    img_poisson = np.random.poisson(img_gaussian * 255) / 255.0
    img_poisson = np.clip(img_poisson, 0, 1)

    return img_poisson


def aug_slice_unbalanced(hat, gaussian_std_range=(0.02, 0.15)):
    hat_e = hat.copy()
    T, _, _ = hat.shape
    for t in range(T):
        hat_e[t] = add_noise_to_slice(hat_e[t], gaussian_std_range)
    return hat_e

class SBEM2_Pure_Denoise_Dataset(dataset.Dataset):
    def __init__(self,
                 data_dir,
                 is_training,
                 patch_y=256,
                 patch_x=256,
                 patch_t=32,
                 dataset_num=1000,
                 downsample_ratio=8,
                 ):
        
        self.noisy_volume = rearrange(imread(list(Path(data_dir).glob("*.tif"))[0]), "d h w -> d h w")
        self.is_training = is_training
        self.desc = "SBEM pure denoise Dataset"
        self.patch_y = patch_y * 2
        self.patch_x = patch_x * 2
        self.patch_t = patch_t
        self.downsample_ratio = downsample_ratio
        self.vol_shape = self.noisy_volume.shape

    def __getitem__(self, index):
        # crop
        st = np.random.randint(0, self.vol_shape[0] - self.patch_t)
        sy = np.random.randint(0, self.vol_shape[1] - self.patch_y)
        sx = np.random.randint(0, self.vol_shape[2] - self.patch_x)
        patch_coor = np.s_[st:st+self.patch_t, sy:sy+self.patch_y, sx:sx+self.patch_x]
        patch = self.noisy_volume[patch_coor]

        # augment
        if self.is_training:
            if random.random() >= 0.5:
                patch = patch[::-1]
            if random.random() >= 0.5:
                patch = patch[:, ::-1]
            if random.random() >= 0.5:
                patch = patch[:, :, ::-1]
            rotations = random.choice([0, 1, 2, 3])
            patch = np.rot90(patch, rotations, axes=(1, 2))
        
        # to float
        patch = patch.astype(np.float32) / 255.0

        if self.is_training:
            # split

            # patch_tilde = aug_slice_unbalanced(patch_tilde)

            patch_hat, patch_tilde = R2R(patch)
        else:
            patch_tilde, patch_hat = patch[:, 0::2, 0::2], patch[:, 1::2, 1::2]

        # to tensor
        input_volume = torch.from_numpy(np.ascontiguousarray(patch_tilde).astype(np.float32)).unsqueeze(0)
        gt_volume = torch.from_numpy(np.ascontiguousarray(patch_hat).astype(np.float32)).unsqueeze(0)
        
        return {"input_volume":input_volume, "gt_volume":gt_volume}

    def __len__(self):
        if self.is_training:
            return 540
        return 32



def get_SBEM_Pure_Denoise_loader(mode, 
                     data_root, 
                     shuffle, 
                     num_workers, 
                     dataset_num=1000,
                     patch_size=256,
                     patch_t=5,
                     ):
    if mode == "train":
        is_training = True
        dataset = SBEM2_Pure_Denoise_Dataset(data_root, 
                                            is_training=is_training, 
                                            patch_x=patch_size,
                                            patch_y=patch_size,
                                            patch_t=patch_t,
                                            dataset_num=dataset_num,
                                            )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        is_training = False
        dataset = SBEM2_Pure_Denoise_Dataset(data_root, 
                                            is_training=is_training, 
                                            patch_x=patch_size,
                                            patch_y=patch_size,
                                            patch_t=32,
                                            dataset_num=dataset_num,
                                            )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
