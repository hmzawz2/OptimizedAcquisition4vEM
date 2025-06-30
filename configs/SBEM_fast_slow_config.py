import os
from .config import *

def get_SBEM_denoise_config():
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
    args.data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/norm_pure_noisy"
    args.val_data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/norm_pure_noisy"
    args.n_outputs = 5
    args.exp_name = "pure_noisy_z50_lt_0d5_lcont_0d1_wo_lssim"
    args.checkpoint_dir = "ckpt"
    args.pretrained = "ckpt/saved_models_final/SBEM_act_fast_slow/norm_pure_noisy_z50_ubr2r_lt_0d5/model_best.pth"

    # test
    args.load_from = "ckpt/saved_models_final/SBEM_act_fast_slow/norm_pure_noisy_z50_ubr2r_lt_1/model_best.pth"
    args.pred_volume_dir = "/home/chenbh/data/huayunfeng/SBEM_type2/pure_noisy/norm"
    args.save_to = "output"

    return args
