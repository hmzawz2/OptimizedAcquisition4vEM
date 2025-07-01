from utils.stack_split_utils import *
import numpy as np
from tifffile import imread, imwrite
import time
from tqdm import tqdm
from model.shiftnet import GShiftNet
from configs.self_supervised_denoise_config import get_config
args = get_config()

def run_model(
    model,
    test_volume,
    image_shape,
):
    # ----------
    #  Crop Subvolumes
    # ----------
    name_list, test_volume, coordinate_dict, input_data_type = get_split_stack_for_pred(image_shape[1], image_shape[2], image_shape[0], 0.1, test_volume)
    patch_set = predset(name_list, coordinate_dict, test_volume)
    patch_loader = DataLoader(patch_set, batch_size=1, shuffle=False)

    pred_volume = np.zeros((test_volume.shape[0], test_volume.shape[1], test_volume.shape[2]), dtype=np.float32)

    with torch.no_grad():
        for patch, coordinates in tqdm(patch_loader):

            # ----------
            #  Inference on Subvolumes
            # ----------
            patch = patch.unsqueeze(1)
            patch = patch.to(device)
            pred = model(patch)
            pred = pred.squeeze().unsqueeze(0)
            pred = pred.detach().cpu().numpy().astype(np.float32)

            # ----------
            #  Put Back
            # ----------
            output_block, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(coordinates, pred.squeeze())
            pred_volume[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = output_block

    
    # ----------
    #  Change Type & Return Final Result
    # ----------
    pred_volume = np.clip(pred_volume, 0.0, 1.0)
    res = (pred_volume * 255).astype(np.uint8)
    return res


if __name__ == "__main__":

    # ----------
    #  Preparing
    # ----------
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # ----------
    #  Init Model
    # ----------
    model = GShiftNet()
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    model.eval()

    # ------------------
    # Create Output Dir
    # ------------------
    model_dir_split = str(args.load_from).split("/")
    model_dir = f"{model_dir_split[-3]}/{model_dir_split[-2]}"
    data_dir_split = str(args.pred_volume_dir).split("/")
    pred_save_dir = Path(args.save_to) / model_dir / data_dir_split[-1]
    if not pred_save_dir.exists():
        pred_save_dir.mkdir(parents=True, exist_ok=False)
    print(f"Pred results will save to {str(pred_save_dir)}")

    # ----------
    #  Model Inference
    # ----------
    tic = time.time()
    
    # For every tif file in input dir
    volume_name_list = list(Path(args.pred_volume_dir).glob("*.tif"))
    # denoise, and
    for volume_name in volume_name_list:
        input_volume = imread(volume_name).astype(np.float32)
        pred_np = run_model(
            model,
            input_volume,
            (5, 256, 256), # 256x256, 5 slices
        )
        # save to output dir
        imwrite(pred_save_dir / str(volume_name.name), pred_np)
    
    tac = time.time()
    print(f"Inference costs {(tac - tic)/60:.2f} min.")