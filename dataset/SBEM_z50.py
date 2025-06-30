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
import torch.utils.data.dataset as dataset
from scipy.ndimage import zoom

def add_noise_to_slice(slice, gaussian_std_range, jitter_range):
    std = np.random.uniform(*gaussian_std_range)
    # gaussian noise
    gaussian_noise = np.random.normal(0, std, size=slice.shape).astype(np.float32)
    img_gaussian = slice + gaussian_noise
    img_gaussian = np.clip(img_gaussian, 0.0, 1.0)
    # poisson noise
    img_poisson = np.random.poisson(img_gaussian * 255) / 255.0
    img_poisson = np.clip(img_poisson, 0, 1)
    # jitter
    img_jit = np.clip(np.random.uniform(*jitter_range) * img_poisson, 0, 1)

    return img_jit


def add_dropout(stack, p=0.5, lth=20):
    if np.random.rand() >= p:
        return stack
    frame_scores = np.std(stack, axis=(1, 2))
    drop_id = np.argmax(frame_scores)
    drop_type = random.choice([0, 1, 2])
    if drop_type == 0:
        # all black
        stack[drop_id] = np.random.normal(0, 0.02, size=stack[drop_id].shape).astype(np.float32)
    elif drop_type == 1:
        # all noise
        stack[drop_id] = np.random.normal(0, 0.5, size=stack[drop_id].shape).astype(np.float32)
    elif drop_type == 2:
        # part black part noisy content
        slice = stack[drop_id]
        sx = np.random.randint(0, slice.shape[1]-lth)
        sy = np.random.randint(0, slice.shape[0]-lth)
        slice[sy:sy+lth, sx:sx+lth] = 0
        slice = slice + np.random.normal(0, 0.2, size=slice.shape).astype(np.float32)
        slice = np.clip(slice, 0, 1)
    return stack

def aug_z50(image, gaussian_std_range=(0.05, 0.2), jitter_range=(0.95, 1.05), dropout_p=0.5):
    hat, tilde = image[:, 0::2, 0::2] , image[:, 1::2, 1::2]

    hat_e, tilde_e = hat.copy(), tilde.copy()

    hat_e, tilde_e = add_dropout(hat_e, dropout_p), add_dropout(tilde_e, dropout_p)

    T, _, _ = hat.shape

    for t in range(T):
        hat_e[t] = add_noise_to_slice(hat_e[t], gaussian_std_range, jitter_range)
        tilde_e[t] = add_noise_to_slice(tilde_e[t], gaussian_std_range, jitter_range)
    
    return hat, hat_e, tilde, tilde_e


class SBEM2_Dataset(dataset.Dataset):
    # 按照真实场景下的获取方案,一张慢扫,k张快扫
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
        
        image_name = list(Path(data_dir).glob("*.tif"))[0]
        print(f"Read image {str(image_name)}")
        self.clean_volume = imread(image_name)

        self.use_cn = use_cn
        if use_cn is True:
            self.noisy_volume = imread(image_name)
        else:
            self.noisy_volume = self.clean_volume
        # print(f"Successfully load {data_dir} c-300 and cn-300 volume.")
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
        # clean_patch = self.clean_volume[patch_coor]
        if self.use_cn:
            noisy_patch = self.noisy_volume[patch_coor]
            noisy_patch = noisy_patch.astype(np.float32) / 255.0
            input_original_depth = (self.nbr_frames-1)*self.downsample_ratio+1
            start_idx = np.random.randint(0, noisy_patch.shape[0]-input_original_depth)
            noisy_stack = noisy_patch[start_idx:start_idx+input_original_depth]
        
        if self.is_training:
            
            stack_hat, stack_hat_e, stack_tilde, stack_tilde_e = aug_z50(noisy_stack)

            # Randomly flip along the first axis
            if random.random() >= 0.5:
                stack_hat = stack_hat[::-1]
                stack_hat_e = stack_hat_e[::-1]
                stack_tilde = stack_tilde[::-1]
                stack_tilde_e = stack_tilde_e[::-1]

            # Randomly flip along the second axis
            if random.random() >= 0.5:
                stack_hat = stack_hat[:, ::-1]
                stack_hat_e = stack_hat_e[:, ::-1]
                stack_tilde = stack_tilde[:, ::-1]
                stack_tilde_e = stack_tilde_e[:, ::-1]

            # Randomly flip along the third axis
            if random.random() >= 0.5:
                stack_hat = stack_hat[:, :, ::-1]
                stack_hat_e = stack_hat_e[:, :, ::-1]
                stack_tilde = stack_tilde[:, :, ::-1]
                stack_tilde_e = stack_tilde_e[:, :, ::-1]

            # Apply the same random rotation to all stacks
            rotations = random.choice([0, 1, 2, 3])
            stack_hat = np.rot90(stack_hat, rotations, axes=(1, 2))
            stack_hat_e = np.rot90(stack_hat_e, rotations, axes=(1, 2))
            stack_tilde = np.rot90(stack_tilde, rotations, axes=(1, 2))
            stack_tilde_e = np.rot90(stack_tilde_e, rotations, axes=(1, 2))

            stack_hat = torch.from_numpy(np.ascontiguousarray(stack_hat).astype(np.float32)).unsqueeze(0)
            stack_hat_e = torch.from_numpy(np.ascontiguousarray(stack_hat_e).astype(np.float32)).unsqueeze(0)
            stack_tilde = torch.from_numpy(np.ascontiguousarray(stack_tilde).astype(np.float32)).unsqueeze(0)
            stack_tilde_e = torch.from_numpy(np.ascontiguousarray(stack_tilde_e).astype(np.float32)).unsqueeze(0)

            return {
                "stack_hat":stack_hat, 
                "stack_hat_e":stack_hat_e,
                "stack_tilde":stack_tilde,
                "stack_tilde_e":stack_tilde_e,
                }

        
        stack_hat, stack_tilde = noisy_stack[:, 0::2, 0::2], noisy_stack[:, 1::2, 1::2]
        input_volume = torch.from_numpy(np.ascontiguousarray(stack_hat).astype(np.float32)).unsqueeze(0)
        gt_volume = torch.from_numpy(np.ascontiguousarray(stack_tilde).astype(np.float32)).unsqueeze(0)
        return {"input_volume":input_volume, "gt_volume":gt_volume}

    def __len__(self):
        if self.is_training:
            return 540
        return 32



def get_SBEM2_z50_loader(mode, 
                        data_root, 
                        shuffle, 
                        num_workers, 
                        dataset_num=1000, 
                        damage_ratio=0.0,
                        nbr_frames=4,
                        patch_size=256,
                        downsample_ratio=4,
                        patch_t=32,
                        use_cn=True,):
    if mode == "train":
        is_training = True
        dataset = SBEM2_Dataset(data_root, 
                                is_training=is_training, 
                                patch_x=patch_size,
                                patch_y=patch_size,
                                patch_t=patch_t,
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
                                patch_t=patch_t,
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
