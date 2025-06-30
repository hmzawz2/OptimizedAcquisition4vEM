import os
import random
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from einops import rearrange
from tifffile import imread, imwrite
from torchvision import transforms

def get_gap_t(
    img_shape,
    stack_num,
    patch_x,
    patch_y,
    patch_t,
    gap_x,
    gap_y,
    dataset_num,
):
    whole_t, whole_y, whole_x = img_shape
    w_num = math.floor((whole_x - patch_x) / gap_x) + 1
    h_num = math.floor((whole_y - patch_y) / gap_y) + 1
    s_num = math.ceil(dataset_num / w_num / h_num / stack_num)
    if s_num == 1:
        raise ZeroDivisionError("Not enough size, please make dataset_num larger.")
    gap_t = math.floor((whole_t - patch_t) / (s_num - 1))
    return gap_t

def get_split_stacks(
    patch_y,
    patch_x,
    patch_t,
    overlap_factor,
    im_folder,
    dataset_num,
):
    gap_y = int(patch_y * (1 - overlap_factor))
    gap_x = int(patch_x * (1 - overlap_factor))

    name_list = []
    coordinate_dict = {}
    volume_index = []
    volume_im_all = []
    ind = 0
    print("\033[1;31mImage list -----> \033[0m")
    print("All files are in -----> ", im_folder)
    stack_list = list(Path(im_folder).glob("*.tif"))
    print("Total stack number -----> ", len(stack_list))

    print("Reading files...")
    for im_name in stack_list:
        im_dir = os.path.join(im_folder, im_name)
        volume_im = imread(im_dir)

        print(im_name, " -----> the shape is", volume_im.shape)
        pure_im_name = str(im_name.name).split(".", maxsplit=1)[0]

        gap_t = get_gap_t(
            volume_im.shape,
            len(stack_list),
            patch_x,
            patch_y,
            patch_t,
            gap_x,
            gap_y,
            dataset_num,
        )

        if gap_t <= 0:
            print(im_name, " -----> calculated_gap <= 0, ignore")
            continue

        volume_im_all.append(volume_im)

        whole_t, whole_y, whole_x = volume_im.shape
        for x in range(0, int((whole_y - patch_y + gap_y) / gap_y)):
            for y in range(0, int((whole_x - patch_x + gap_x) / gap_x)):
                for z in range(0, int((whole_t - patch_t + gap_t) / gap_t)):
                    single_coordinate = {
                        "init_h": 0,
                        "end_h": 0,
                        "init_w": 0,
                        "end_w": 0,
                        "init_s": 0,
                        "end_s": 0,
                    }
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                    init_s = gap_t * z
                    end_s = gap_t * z + patch_t
                    single_coordinate["init_h"] = init_h
                    single_coordinate["end_h"] = end_h
                    single_coordinate["init_w"] = init_w
                    single_coordinate["end_w"] = end_w
                    single_coordinate["init_s"] = init_s
                    single_coordinate["end_s"] = end_s
                    patch_name = f"{im_folder}_{pure_im_name}_x{x}_y{y}_z{z}"

                    name_list.append(patch_name)
                    coordinate_dict[patch_name] = single_coordinate
                    volume_index.append(ind)
        ind = ind + 1
    return name_list, volume_im_all, coordinate_dict, volume_index


def volume_transform(inpu):
    inpu = np.ascontiguousarray(inpu)
    return torch.from_numpy(inpu.astype(np.float32) / 255.0).to(torch.float32)


def aug(img, gt, img_up):
    if random.random() < 0.5:
        img = np.flip(img, axis=2)
        gt = np.flip(gt, axis=2)
        img_up = np.flip(img_up, axis=2)
    if random.random() < 0.5:
        img = np.rot90(img, k=1, axes=(1, 2))
        gt = np.rot90(gt, k=1, axes=(1, 2))
        img_up = np.rot90(img_up, k=1, axes=(1, 2))
    return img, gt, img_up


class BaseVolumeDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        patch_y=160,
        patch_x=160,
        patch_t=160,
        overlap_factor=0.2,
        dataset_num=1000,
        downsample_ratio=10,
    ):
        name_list, noise_im_all, coordinate_dict, stack_index = get_split_stacks(
            patch_y,
            patch_x,
            patch_t,
            overlap_factor,
            data_dir,
            dataset_num,
        )
        self.patch_y = patch_y
        self.patch_x = patch_x
        self.patch_t = patch_t
        self.patch_name_list = name_list
        self.noise_im_all = noise_im_all
        self.coordinate_dict = coordinate_dict
        self.patch_index = stack_index
        self.trans = volume_transform
        self.depth_downsample_ratio = downsample_ratio

    def __getitem__(self, index):
        patch_name = self.patch_name_list[index]
        patch_im_index = self.patch_index[index]
        noise = self.noise_im_all[patch_im_index]

        coordinate = self.coordinate_dict[patch_name]
        cur_patch = noise[
            coordinate["init_s"] : coordinate["end_s"],
            coordinate["init_h"] : coordinate["end_h"],
            coordinate["init_w"] : coordinate["end_w"],
        ]
        return cur_patch

    def __len__(self):
        return len(self.patch_name_list)

def add_damage(inputs, max_damage_num=5):
    for inpu in inputs:
        for i in [1, 2]:
            layer = inpu[i]
            num_squares = np.random.randint(1, max_damage_num)
            height, width = layer.shape
            for _ in range(num_squares):
                lth = np.random.randint(5, 25)
                x = np.random.randint(0, width - lth)
                y = np.random.randint(0, height - lth)
                seed = np.random.rand()
                if seed >= 0.5:
                    layer[y:y+lth, x:x+lth] = 1 - 1e-2*seed
                else:
                    layer[y:y+lth, x:x+lth] = 0 + 1e-2*seed
    return inputs