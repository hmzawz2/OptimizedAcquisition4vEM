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
from .base_dataset import BaseVolumeDataset
from scipy.ndimage import zoom

def R2R(vol, variance=0.05):
    vol_dtype = vol.dtype
    vol = vol.astype("float32")
    frame = vol.shape[0]

    def interpft(x, ny, dim=0):
        x = np.moveaxis(x, dim, 0)  # Bring the target axis to the front
        m = x.shape[0]
        n = x.shape[1:]
        fft_data = np.fft.fft(x, axis=0)
        nyquist = (m + 1) // 2

        # Create extended frequency array
        interp_fft = np.vstack(
            [
                fft_data[:nyquist],
                np.zeros((ny - m,) + n, dtype=fft_data.dtype),
                fft_data[nyquist:],
            ]
        )

        # Adjust for even-length input
        if m % 2 == 0:
            interp_fft[nyquist] /= 2
            interp_fft[nyquist + ny - m] = interp_fft[nyquist]

        # Perform inverse FFT and scale
        y = np.fft.irfft(interp_fft, n=ny, axis=0) * (ny / m)
        return np.moveaxis(y, 0, dim)  # Move axis back to original position

    def fourier_inter(image):

        # Original image dimensions
        x, y = image.shape

        # Calculate adjusted size
        n = 2  # Scaling factor
        sz = np.array([x, y]) - (np.array([x, y]) % 2)  # Ensure even dimensions
        half_sz = sz // 2
        idx_start = np.ceil(half_sz).astype(int)
        idx_end = idx_start + (sz * n - 1)

        # Pad image symmetrically
        pad_width = (half_sz // n).astype(int)
        img_padded = np.pad(
            image,
            ((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1])),
            mode="symmetric",
        )

        # New size for Fourier interpolation
        new_size = (np.array(img_padded.shape) * n).astype(int)

        # Perform Fourier interpolation along both axes
        img_rescaled = interpft(img_padded, new_size[0], dim=0)
        img_rescaled = interpft(img_rescaled, new_size[1], dim=1)

        # Crop the image back to 2x scaled size
        imgf1 = img_rescaled[
            idx_start[0] - 1 : idx_end[0], idx_start[1] - 1 : idx_end[1]
        ]

        # Ensure no negative values in the output
        imgf1 = np.clip(imgf1, 0, None)
        return imgf1

    hat = (vol[:, 0::2, 0::2] + vol[:, 1::2, 1::2]) / 2.0
    tilde = (vol[:, 1::2, 0::2] + vol[:, 0::2, 1::2]) / 2.0
    hat = np.array([fourier_inter(hat[i]).astype(np.float32) for i in range(frame)])
    tilde = np.array([fourier_inter(tilde[i]).astype(np.float32) for i in range(frame)])
    hat, tilde = hat.astype(vol_dtype), tilde.astype(vol_dtype)

    noise = np.random.normal(0, np.sqrt(variance), vol.shape)
    hat = hat + 0.5 * variance * noise
    tilde = tilde - 2 * variance * noise
    hat, tilde = np.clip(hat, 0, 1), np.clip(tilde, 0, 1)
    return hat, tilde

class EPFL_Dataset(BaseVolumeDataset):
    def __init__(self,
                 data_dir,
                 is_training,
                 patch_y=256,
                 patch_x=256,
                 patch_t=256,
                 overlap_factor=0.2,
                 dataset_num=1000,
                 downsample_ratio=8,
                 damage_ratio=0.0,
                 nbr_frames=4):
        super().__init__(
            data_dir,
            patch_y,
            patch_x,
            patch_t,
            overlap_factor,
            dataset_num,
            downsample_ratio,
        )
        self.is_training = is_training
        self.damage_ratio = damage_ratio
        self.nbr_frames = nbr_frames
        self.desc = "EPFL Dataset"

    def __getitem__(self, index):
        index = np.random.randint(0, len(self.patch_name_list))
        gt = super().__getitem__(index)

        if self.is_training:
            if random.random() >= 0.5:
                gt = gt[::-1]
            if random.random() >= 0.5:
                gt = gt[:, ::-1]
            if random.random() >= 0.5:
                gt = gt[:, :, ::-1]

            if random.random() >= 0.5:
                gt = rearrange(gt, 'd h w -> w d h')
            else:
                gt = rearrange(gt, 'd h w -> h d w')

            rotations = random.choice([0, 1, 2, 3])
            gt = np.rot90(gt, rotations, axes=(1, 2))
            
        gt = gt.astype(np.float32) / 255.0

        input_volume = []
        gt_volume = []

        input_original_depth = (self.nbr_frames-1)*self.depth_downsample_ratio+1
        start_idx = np.random.randint(0, gt.shape[0]-input_original_depth)
        stack = gt[start_idx:start_idx+input_original_depth]
        stack_tilde, stack_hat = R2R(stack)

        input_stack = stack_hat[::self.depth_downsample_ratio]
        gt_start = self.depth_downsample_ratio*int(self.nbr_frames/2 - 1)
        gt_end = self.depth_downsample_ratio*int(self.nbr_frames/2)
        gt_stack = stack_tilde[gt_start:gt_end]

        for i in range(0, self.depth_downsample_ratio):
            input_v = input_stack.copy()
            timestep = np.ones_like(input_v[0:1], dtype=np.float32)*float(i)/self.depth_downsample_ratio
            input_v = np.concatenate((input_v, timestep), axis=0)
            
            # if i == 0:
            #     slice = gt_stack[0:1].copy()
            #     slice_left = (slice[:, 0::2, 0::2] + slice[:, 1::2, 1::2]) / 2.0
            #     slice = input_v[self.nbr_frames // 2 - 1]
            #     slice = slice[np.newaxis, :, :]
            #     slice_right = (slice[:, 1::2, 0::2] + slice[:, 0::2, 1::2]) / 2.0
            #     slice_left, slice_right = zoom(slice_left, (1, 2, 2), order=3), zoom(slice_right, (1, 2, 2), order=3)
                
            #     slice_gt = slice_left
            #     input_v[self.nbr_frames // 2 - 1] = slice_right
            # else:
            slice_gt = gt_stack[i:i+1].copy()

            input_volume.append(input_v)
            gt_volume.append(slice_gt)

        input_volume = [torch.from_numpy(np.ascontiguousarray(v).astype(np.float32)) for v in input_volume]
        gt_volume = [torch.from_numpy(np.ascontiguousarray(v).astype(np.float32)) for v in gt_volume]  

        return {"input_volume":input_volume, "gt_volume":gt_volume}

    def __len__(self):
        return min(540, len(self.patch_name_list))


def get_EPFL_loader(mode, 
                    data_root, 
                    shuffle,
                    num_workers, 
                    dataset_num=1000,
                    damage_ratio=0.0,
                    nbr_frames=4):
    if mode == "train":
        is_training = True
        dataset = EPFL_Dataset(data_root, 
                               is_training=is_training, 
                               dataset_num=dataset_num,
                               damage_ratio=0.0,
                               nbr_frames=nbr_frames)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        is_training = False
        dataset = EPFL_Dataset(data_root, 
                               is_training=is_training, 
                               dataset_num=dataset_num,
                               nbr_frames=nbr_frames)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )