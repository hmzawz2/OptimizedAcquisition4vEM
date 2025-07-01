import os
from .config import *

def get_config():
    args, _ = get_args()

    # Dataset info
    args.dataset = "used dataset name string"
    args.exp_name = "the experiment name string"
    args.data_root = "for supervised: the path where put fast and slow tifs; for self-supervised: the path where put fast tif"
    args.val_data_root = "for supervised: the path where put fast and slow tifs; for self-supervised: the path where put fast tif"
    args.use_fast_slow = "only used in supervised mode, whether to open fast-slow mode"

    # Training info
    args.loss = "loss function string, example:1*L1+1*SSIM"
    args.max_epoch = 100
    args.lr = 0.0002
    args.pretrained = "the path where to load a pretrained model"

    # Others
    args.checkpoint_dir = "ckpt"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Inference info
    args.load_from = "the path where to load a pretrained model"
    args.pred_volume_dir = "the dir contains noisy tifs, all noisy tifs will be processed"
    args.save_to = "output"

    return args