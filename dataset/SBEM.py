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

class SBEM2_Dataset(dataset.Dataset):
    def __init__(self,
                 data_dir,
                 is_training,
                 patch_y=256,
                 patch_x=256,
                 patch_t=32,
                 dataset_num=1000,
                 downsample_ratio=8,
                 damage_ratio=0.0,
                 nbr_frames=4,
                 use_cn=True):
        
        self.clean_volume = imread(list(Path(data_dir).glob("*.tif"))[0])
        self.use_cn = use_cn
        if use_cn is True:
            self.noisy_volume = imread(list(Path(data_dir).glob("cn-*.tif"))[0])
        else:
            self.noisy_volume = self.clean_volume
        print(f"Successfully load {data_dir} c-300 and cn-300 volume.")
        self.is_training = is_training
        self.nbr_frames = nbr_frames
        self.damage_ratio = damage_ratio
        self.desc = "SBEM Dataset"
        self.patch_y = patch_y
        self.patch_x = patch_x
        self.patch_t = patch_t
        self.downsample_ratio = downsample_ratio
        self.vol_shape = self.clean_volume.shape

    def __getitem__(self, index):
        
        st = np.random.randint(0, self.vol_shape[0] - self.patch_t)
        sy = np.random.randint(0, self.vol_shape[1] - self.patch_y)
        sx = np.random.randint(0, self.vol_shape[2] - self.patch_x)
        patch_coor = np.s_[st:st+self.patch_t, sy:sy+self.patch_y, sx:sx+self.patch_x]
        clean_patch = self.clean_volume[patch_coor]
        if self.use_cn:
            noisy_patch = self.noisy_volume[patch_coor]

        if self.is_training:
            if random.random() >= 0.5:
                clean_patch = clean_patch[::-1]
                if self.use_cn:
                    noisy_patch = noisy_patch[::-1]
            if random.random() >= 0.5:
                clean_patch = clean_patch[:, ::-1]
                if self.use_cn:
                    noisy_patch = noisy_patch[:, ::-1]
            if random.random() >= 0.5:
                clean_patch = clean_patch[:, :, ::-1]
                if self.use_cn:
                    noisy_patch = noisy_patch[:, :, ::-1]

            rotations = random.choice([0, 1, 2, 3])
            clean_patch = np.rot90(clean_patch, rotations, axes=(1, 2))
            if self.use_cn:
                noisy_patch = np.rot90(noisy_patch, rotations, axes=(1, 2))
            
        clean_patch = clean_patch.astype(np.float32) / 255.0
        if self.use_cn:
            noisy_patch = noisy_patch.astype(np.float32) / 255.0

        input_original_depth = (self.nbr_frames-1)*self.downsample_ratio+1
        start_idx = np.random.randint(0, clean_patch.shape[0]-input_original_depth)
        clean_stack = clean_patch[start_idx:start_idx+input_original_depth]
        if self.use_cn:
            noisy_stack = noisy_patch[start_idx:start_idx+input_original_depth]
        stack_tilde, stack_hat = R2R(clean_stack)

        if self.use_cn:
            # # slow-fast-slow denoise
            # input_stack = stack_hat[::self.downsample_ratio]

            # # Here, the +1 operation is for construct 1+3+1 patch
            # noisy_start = self.downsample_ratio*int(self.nbr_frames/2 - 1) + 1 
            # noisy_end = self.downsample_ratio*int(self.nbr_frames/2)
            # noisy_inpu = noisy_stack[noisy_start:noisy_end]
            # input_stack = np.concatenate((input_stack[:1], noisy_inpu, input_stack[-1:]), axis=0)
            # gt_stack = stack_tilde

            # for pure denoise
            input_stack = np.concatenate((input_stack[:1], noisy_inpu, input_stack[-1:]), axis=0)
            gt_stack = stack_tilde[1:-1]
            input_stack = noisy_stack
            gt_stack = clean_stack
        
        else:
            # interpolation
            input_stack = stack_hat[::self.downsample_ratio]
            
            # To denoise the first input layer, delete the +1 operation in use_cn branch
            noisy_start = self.downsample_ratio*int(self.nbr_frames/2 - 1)
            noisy_end = self.downsample_ratio*int(self.nbr_frames/2)
            gt_stack = stack_tilde[noisy_start:noisy_end]

        # volume train
        input_volume = torch.from_numpy(np.ascontiguousarray(input_stack).astype(np.float32)).unsqueeze(0)
        gt_volume = torch.from_numpy(np.ascontiguousarray(gt_stack).astype(np.float32)).unsqueeze(0)
        
        return {"input_volume":input_volume, "gt_volume":gt_volume}

    def __len__(self):
        return 540



def get_SBEM2_loader(mode, 
                     data_root, 
                     shuffle, 
                     num_workers, 
                     dataset_num=1000, 
                     damage_ratio=0.0,
                     nbr_frames=4,
                     patch_size=256,
                     downsample_ratio=4,
                     use_cn=True,):
    if mode == "train":
        is_training = True
        dataset = SBEM2_Dataset(data_root, 
                                is_training=is_training, 
                                patch_x=patch_size,
                                patch_y=patch_size,
                                patch_t=32,
                                dataset_num=dataset_num,
                                damage_ratio=damage_ratio,
                                nbr_frames=nbr_frames,
                                downsample_ratio=downsample_ratio,
                                use_cn=use_cn)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        is_training = False
        dataset = SBEM2_Dataset(data_root, 
                                is_training=is_training, 
                                patch_x=patch_size,
                                patch_y=patch_size,
                                patch_t=32,
                                dataset_num=dataset_num,
                                nbr_frames=nbr_frames,
                                downsample_ratio=downsample_ratio,
                                use_cn=use_cn)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
