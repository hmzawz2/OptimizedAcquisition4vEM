import os
from .config import *

def get_config():
    args, _ = get_args()

    # Dataset info
    args.dataset = "SBEM2_Z50"
    args.exp_name = "pure_fast"
    args.data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/train"
    args.val_data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/train"
    args.use_fast_slow = 1

    # Training info
    args.loss = "1*L1+1*SSIM"
    args.max_epoch = 100
    args.lr = 0.0002
    # args.pretrained = "ckpt/saved_models_final/SBEM3_Z25/SBEM3_Z25_diag_fz/checkpoint.pth"

    # Others
    args.checkpoint_dir = "ckpt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Inference info
    args.load_from = "ckpt/SBEM3_Z25/SBEM3_Z50_diag_2_fz_R2R_new_loss/checkpoint.pth"
    args.pred_volume_dir = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/SBEM3-Z25/Z25-inpu"
    args.save_to = "output"

    return args