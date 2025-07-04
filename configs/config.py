import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--dataset', type=str)
data_arg.add_argument('--data_root', type=str, help="training data root")
data_arg.add_argument('--val_data_root', type=str, help="validation data root")

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--nbr_frame' , type=int , default=4)
model_arg.add_argument('--nbr_width' , type=int , default=1)

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--loss', type=str, default='1*L1')
learn_arg.add_argument('--lr', type=float, default=2e-4)
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.99)
learn_arg.add_argument('--batch_size', type=int, default=1)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=200)
learn_arg.add_argument('--resume', action='store_true')
learn_arg.add_argument('--resume_exp', type=str, default=None)
learn_arg.add_argument('--checkpoint_dir', type=str ,default="./ckpt")
learn_arg.add_argument('--tensorboard_dir', type=str, default="./log")
learn_arg.add_argument('--use_fast_slow', type=int, default=1)
learn_arg.add_argument("--pretrained" , type=str, help="point to a pretrained model path (.pth file)")

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--exp_name', type=str, default='exp')
misc_arg.add_argument('--log_iter', type=int, default=60)
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=3407)
misc_arg.add_argument('--num_workers', type=int, default=16)
misc_arg.add_argument('--use_tensorboard', action='store_true')
misc_arg.add_argument('--val_freq', type=int, default=1)

# Pred
pred_arg = add_argument_group('Pred')
pred_arg.add_argument('--pred_volume_dir', type=str, default="./")
pred_arg.add_argument('--load_from', type=str, default=None)
pred_arg.add_argument('--save_to', type=str, default="./output")

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
