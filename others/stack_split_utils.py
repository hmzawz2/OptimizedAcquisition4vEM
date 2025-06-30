import numpy as np
import os
from tifffile import imread
from skimage import io
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from pathlib import Path
from torchvision.transforms import transforms
from einops import rearrange
import re
from math import floor, ceil

"""
Copy and modified from SRDTrans projects
https://github.com/cabooster/SRDTrans,
thanks for their great work.
"""

def get_split_stack_for_pred(
        patch_y,
        patch_x,
        patch_t,
        overlap_factor,
        volume_im,
        gap_t=None,
    ):

    if gap_t is None:
        gap_t = int(patch_t * (1 - overlap_factor))
    gap_x = int(patch_x * (1 - overlap_factor))
    gap_y = int(patch_y * (1 - overlap_factor))

    cut_w = floor((patch_x - gap_x)/2)
    cut_h = floor((patch_y - gap_y)/2)
    cut_s = floor((patch_t - gap_t)/2)

    assert cut_w >=0 and cut_h >= 0 and cut_s >= 0, "test cut size is negative!"

    name_list = []
    coordinate_dict={}

    if type(volume_im) is str:
        volume_im = imread(volume_im)
    input_data_type = volume_im.dtype

    whole_x = volume_im.shape[2]
    whole_y = volume_im.shape[1]
    whole_t = volume_im.shape[0]

    num_w = math.ceil((whole_x-patch_x+gap_x)/gap_x)
    num_h = math.ceil((whole_y-patch_y+gap_y)/gap_y)
    num_s = math.ceil((whole_t-patch_t+gap_t)/gap_t)

    for z in range(0, num_s):
        for x in range(0,num_h):
            for y in range(0,num_w):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_y*x
                    end_h = gap_y*x + patch_y
                elif x == (num_h-1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w-1):
                    init_w = gap_x*y
                    end_w = gap_x*y + patch_x
                elif y == (num_w-1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s-1):
                    init_s = gap_t*z
                    end_s = gap_t*z + patch_t
                elif z == (num_s-1):
                    init_s = whole_t - patch_t
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y*gap_x
                    single_coordinate['stack_end_w'] = y*gap_x+patch_x-cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x-cut_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_x-patch_x+cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y*gap_x+cut_w
                    single_coordinate['stack_end_w'] = y*gap_x+patch_x-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x-cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x*gap_y
                    single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y-cut_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_y-patch_y+cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x*gap_y+cut_h
                    single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y-cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z*gap_t
                    single_coordinate['stack_end_s'] = z*gap_t+patch_t-cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t-cut_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_t-patch_t+cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t
                else:
                    single_coordinate['stack_start_s'] = z*gap_t+cut_s
                    single_coordinate['stack_end_s'] = z*gap_t+patch_t-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t-cut_s

                patch_name = 'volume' +'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                name_list.append(patch_name)
                coordinate_dict[patch_name] = single_coordinate

    return name_list, volume_im, coordinate_dict, input_data_type

def singlebatch_test_save(single_coordinate, output_image, z_scale=None):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    if z_scale is not None:
        stack_start_s *= z_scale
        stack_end_s *= z_scale
        patch_start_s *= z_scale
        patch_end_s *= z_scale

    output_block = output_image[:, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return output_block, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s

class predset(Dataset):
    def __init__(
            self,
            name_list,
            coordinate_dict,
            im,
        ):
        self.name_list = name_list
        self.coordinate_dict = coordinate_dict
        self.im = im
    
    @staticmethod
    def transform(patch):
        return torch.from_numpy(patch.astype(np.float32) / 255).to(torch.float32)

    def __getitem__(self, index):
        single_coordinate = self.coordinate_dict[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        patch = self.im[init_s:end_s, init_h:end_h, init_w:end_w]
        
        return predset.transform(patch), single_coordinate

    def __len__(self):
        return len(self.name_list)

def sort_paths(paths, position_from_end=1):
    def extract_digit_from_end(filename, position_from_end):
        numbers = re.findall(r'\d+', filename.stem)
        if numbers:
            return int(numbers[-position_from_end])
        raise NotImplementedError(f"Cannot match any digits in path: {filename.name}")

    return sorted(paths, key=lambda x: extract_digit_from_end(x, position_from_end))


if __name__ == "__main__":
    name_list, volume_im, coordinate_dict, input_data_type = get_split_stack_for_pred(160, 160, 16, 0.1, "/home/chenbh/data/SRDTrans/CREMI_tiny", 0)
    print(len(name_list))

    ans = np.zeros((volume_im.shape[0]*10, volume_im.shape[1], volume_im.shape[2]), dtype=np.uint8)

    testset = predset(name_list, coordinate_dict, volume_im)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    for iteration, (patch, coordinate) in enumerate(testloader):
        
        print(patch.shape)
        pred = torch.rand((patch.shape[0], patch.shape[1]*10, patch.shape[2], patch.shape[3]))
        
        pred = (pred.detach().cpu().numpy() * 255).astype(np.uint8).squeeze()

        output_block, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(coordinate, pred, 10)

        ans[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = output_block
    
