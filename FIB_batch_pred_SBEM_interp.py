import os
import sys
import time
import copy
import shutil
import random
import pdb
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from tifffile import imread, imwrite

# from configs.epfl_config import get_EPFL_config
# from configs.fib25_config import get_FIB25_config
# from configs.fib25_dbase_config import get_FIB25_dbase_config
# from configs.bootstrap_config import get_bootstrap_config
from configs.SBEM_interp_config import get_SBEM_interp_config
import myutils
from dataset.test_dataset.stack_split_utils import get_split_stack_for_pred, predset, singlebatch_test_save, sort_paths

from torch.utils.data import DataLoader
import torch.nn.functional as F

##### Parse CmdLine Arguments #####
# args = get_FIB25_config()
# args = get_EPFL_config()
# args = get_FIB25_dbase_config()
# args = get_bootstrap_config()
args = get_SBEM_interp_config()
print(args)

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# from model.myFLAVR import FLAVR_model
# print("Building model: %s"%args.model.lower())
# model = FLAVR_model(n_inputs=args.nbr_frame, # only interp, no timestep
#                     n_outputs=args.n_outputs, 
#                     joinType=args.joinType, 
#                     )
from model.shiftnet import GShiftNet
model = GShiftNet()


model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

def pred(model, 
         volume_im, 
         args, 
         patch_t=2, 
         patch_y=256,
         patch_x=256):
        
    name_list, volume_im, coordinate_dict, input_data_type = get_split_stack_for_pred(patch_y, patch_x, patch_t, 0.1, volume_im, 1)
    pred_shape_t, pred_shape_h, pred_shape_w = volume_im.shape
    pred_shape_t = pred_shape_t * args.sr_ratio
    pred_volume = np.zeros((pred_shape_t, pred_shape_h, pred_shape_w), dtype=np.float32)

    pred_set = predset(name_list, coordinate_dict, volume_im)
    pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False)

    with torch.no_grad():
        for pidx, (patch, coordinate) in enumerate(tqdm(pred_loader)):
            patch = patch.cuda().unsqueeze(0)

            # for interpolation
            # patch = F.upsample(patch, scale_factor=(1.5, 1, 1), mode='trilinear', align_corners=True)
            
            _, _, d, _, _ = patch.shape
            d_2 = int(d/2)
            # # timestep_outs = [patch[:, :, d_2-1:d_2]] + timestep_outs
            # pred = torch.cat(timestep_outs, dim=2)
            pred = model(patch).detach().cpu()
            pred = pred.squeeze().numpy()

            #TODO: change to a better solution
            output_block, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(coordinate, pred, args.sr_ratio)
            if stack_start_s == 0:
                stack_start_s = (d_2 - 1) * args.sr_ratio
            pred_volume[stack_start_s:stack_start_s+args.sr_ratio, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = output_block

    pred_volume = np.clip(pred_volume, 0, 1)
    res = (pred_volume * 255).astype(np.uint8)
    return res


""" Entry Point """
def main(args):
    assert args.load_from is not None
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    model.eval()

    # ------------------
    # Create saving dir
    # ------------------
    model_dir_split = str(args.load_from).split("/")
    model_dir = f"{model_dir_split[-3]}/{model_dir_split[-2]}"
    data_dir_split = str(args.pred_volume_dir).split("/")
    pred_save_dir = Path(args.save_to) / model_dir / data_dir_split[-1]
    if not pred_save_dir.exists():
        pred_save_dir.mkdir(parents=True, exist_ok=False)
    print(f"Pred results will save to {str(pred_save_dir)}")

    # ------------------
    # Pred and save
    # ------------------
    volume_names = list(Path(args.pred_volume_dir).glob("*.tif"))
    if len(volume_names) > 1:
        volume_names = sort_paths(volume_names)
    for i in range(len(volume_names)):
        print(f"Pred: {volume_names[i].name}")
        volume_im = imread(volume_names[i])[:40][::2]
        pred_volume = pred(model, volume_im, args, 2)
        imwrite(pred_save_dir/f"{volume_names[i].name}", pred_volume)
    
    print("Done.")

if __name__ == "__main__":
    main(args)
