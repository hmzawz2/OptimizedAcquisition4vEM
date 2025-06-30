import os
from .config import *

def get_SBEM_denoise_config():
    args, _ = get_args()

    args.sr_ratio = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args.batch_size = 1
    args.nbr_frame = 2
    args.test_batch_size = 1
    args.dataset = "SBEM3_Z25"
    args.loss = "1*L1"
    args.max_epoch = 300
    args.lr = 0.0002
    args.data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/SBEM3-Z50/train"
    args.val_data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/SBEM3-Z50/train"
    args.n_outputs = 5
    args.exp_name = "SBEM3_Z50_diag_2_fz_R2R"
    args.checkpoint_dir = "ckpt"
    # args.pretrained = "ckpt/saved_models_final/SBEM3_Z25/SBEM3_Z25_diag_fz/checkpoint.pth"

    # test
    args.load_from = "ckpt/saved_models_final/SBEM3_Z25/SBEM3_Z25_diag_moreR2R/checkpoint.pth"
    args.pred_volume_dir = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/SBEM3-Z25/Z25-inpu"
    args.save_to = "output"

    return args
