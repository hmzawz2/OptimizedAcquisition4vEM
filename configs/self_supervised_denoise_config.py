import os
from .config import *

def get_config():
    args, _ = get_args()

    # Dataset info
    args.dataset = "SBEM3_Z25"
    args.exp_name = "pure_fast"
    args.data_root = "SBEM3-Z50/train"
    args.val_data_root = "SBEM3-Z50/train"

    # Training info
    args.loss = "1*L1+1*SSIM"
    args.max_epoch = 100
    args.lr = 0.0002
    # args.pretrained = "checkpoint.pth"

    # Others
    args.checkpoint_dir = "ckpt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Inference info
    args.load_from = "ckpt/checkpoint.pth"
    args.pred_volume_dir = "SBEM_type2/SBEM3-Z50"
    args.save_to = "output"

    return args