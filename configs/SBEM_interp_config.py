import os
from .config import *

def get_SBEM_interp_config():
    args, _ = get_args()

    args.sr_ratio = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args.batch_size = 1
    args.nbr_frame = 2
    args.test_batch_size = 1
    args.dataset = "SBEM_clean_vol_gshift"
    args.loss = "1*L1"
    args.max_epoch = 60
    args.lr = 0.0002
    args.data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/train"
    args.val_data_root = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/train"
    args.n_outputs = 3
    args.exp_name = "SBEM_clean_interp_volume_Gshift_i2o2_n"
    args.checkpoint_dir = "ckpt"
    args.pretrained = "ckpt/saved_models_final/SBEM_clean_vol_gshift/SBEM_clean_interp_volume_Gshift/model_best.pth"
    # args.damage_ratio = 0.5

    # test
    args.load_from = "ckpt/saved_models_final/SBEM_clean_vol_gshift/SBEM_clean_interp_volume_Gshift_i2o2_n/model_best.pth"
    args.pred_volume_dir = "/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/pure_clear" #"/home/chenbohao/cbh_3p2/data/huayunfeng/SBEM_type2/pure_clear" #"/home/chenbohao/cbh_3p2/data/FIB25/test-90" #"/mnt/Ext001/chenbohao/cbh_3p2/data/FIB25/test_512/inpu"
    # args.save_to = "/mnt/Ext001/chenbohao/cbh_3p2/python/I2VEM/output" #"/mnt/Ext001/chenbohao/cbh_3p2/data/FIB25/test_512/FLAVR_enhanced"

    return args
