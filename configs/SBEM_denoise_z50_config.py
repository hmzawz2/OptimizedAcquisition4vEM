import os
from .config import *

def get_SBEM_denoise_z50_config():
    args, _ = get_args()

    args.sr_ratio = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args.batch_size = 1
    args.nbr_frame = 2
    args.test_batch_size = 1
    args.dataset = "SBEM_act_fast_slow"
    args.loss = "1*L1"
    args.max_epoch = 300
    args.lr = 0.0002
    args.data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/SBEM3-Z50/train"
    args.val_data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/SBEM3-Z50/train"
    args.n_outputs = 5
    args.exp_name = "SBEM3-Z50_wlt_0d1_wl1_diag_fz"
    args.checkpoint_dir = "ckpt"
    # args.pretrained = "zhuhai/wffl_wol1_drop0d1_wssim_wlt0d2/checkpoint.pth"

    # test
    args.load_from = "ckpt/saved_models_final/SBEM_act_fast_slow/E_z50_noisy_lt_0d2_wssim/checkpoint.pth" #"ckpt/saved_models_final/SBEM_act_fast_slow/E_z50_noisy_lt_0d6_wl1_wtv1e5/checkpoint.pth"
    args.pred_volume_dir = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/pure_noisy"
    args.save_to = "output"

    return args
