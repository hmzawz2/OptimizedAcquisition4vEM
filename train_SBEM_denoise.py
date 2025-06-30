import os
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.shiftnet import GShiftNet
from configs.SBEM_denoise_config import get_SBEM_denoise_config
from pytorch_msssim import ssim
from tifffile import imwrite
import torch.nn.functional as F
from utils import utils

##### Parse CmdLine Arguments #####
args = get_SBEM_denoise_config()
print(args)

save_loc = os.path.join(
    args.checkpoint_dir, "saved_models_final", args.dataset, args.exp_name
)
if not os.path.exists(save_loc):
    os.makedirs(save_loc)
opts_file = os.path.join(save_loc, "opts.txt")
with open(opts_file, "w") as fh:
    fh.write(str(args))


##### TensorBoard & Misc Setup #####
writer_loc = os.path.join(
    args.tensorboard_dir, "tensorboard_logs_%s_final/%s" % (args.dataset, args.exp_name)
)
writer = SummaryWriter(writer_loc)

device = torch.device("cuda" if args.cuda else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

from dataset import get_SBEM_Pure_Denoise_loader

train_loader = get_SBEM_Pure_Denoise_loader(
    "train",
    args.data_root,
    shuffle=True,
    num_workers=args.num_workers,
    dataset_num=6000,
    patch_size=128,
    patch_t=5,
)
test_loader = get_SBEM_Pure_Denoise_loader(
    "test",
    args.val_data_root,
    shuffle=False,
    num_workers=args.num_workers,
    dataset_num=1000,
    patch_size=128,
    patch_t=5,
)


print("Building model: %s" % args.model.lower())

model = GShiftNet()
model = torch.nn.DataParallel(model).to(device)

##### Define Loss & Optimizer #####

## ToDo: Different learning rate schemes for different parameters

optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)


def train(args, epoch):
    losses, psnrs, ssims, lpipss = utils.init_meters(args.loss)

    model.train()
    # criterion.train()

    t = time.time()
    for bidx, data_dict in enumerate(train_loader):

        # Build input batch
        images = data_dict["input_volume"]
        gts = data_dict["gt_volume"]

        images = images.cuda()
        gts = gts.cuda()

        # Forward
        optimizer.zero_grad()
        out = model(images)
        
        # out = rearrange(out, 'n c d h w -> n (c d) h w')
        # gts = rearrange(gts, 'n c d h w -> n (c d) h w')
        # loss, loss_specific = criterion(out, gts)

        # # Save loss values
        # for k, v in losses.items():
        #     if k != "total":
        #         v.update(loss_specific[k].item())
        # losses["total"].update(loss.item())

        l1_loss = torch.nn.L1Loss()(out, gts)
        ssim_loss = 1 - ssim(out, gts, data_range=1.0)
        loss = l1_loss + ssim_loss

        loss.backward()
        optimizer.step()

        # Calc metrics & print logs
        if bidx % args.log_iter == 0:
            # utils.eval_metrics(out, gts, psnrs, ssims, lpipss)

            print(
                f"Train Epoch: {epoch} [{bidx}/{len(train_loader)}]\tL1Loss: {l1_loss.item():.4f}\tSSIM Loss: {ssim_loss.item():.4f}",
                flush=True
            )

            # Log to TensorBoard
            # timestep = epoch * len(train_loader) + bidx
            # writer.add_scalar("Loss/train", loss.item(), timestep)
            # writer.add_scalar("PSNR/train", psnrs.avg, timestep)
            # writer.add_scalar("SSIM/train", ssims.avg, timestep)
            # writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], timestep)

            # Reset metrics
            # losses, psnrs, ssims, lpipss = utils.init_meters(args.loss)
            # t = time.time()


def test(args, epoch):
    print("Evaluating for epoch = %d" % epoch)
    losses, psnrs, ssims, lpipss = utils.init_meters(args.loss)
    model.eval()

    t = time.time()
    with torch.no_grad():
        for bidx, data_dict in enumerate(tqdm(test_loader)):
            images = data_dict["input_volume"]
            gts = data_dict["gt_volume"]

            # Build input batch
            images = images.cuda()
            gts = gts.cuda()

            out = model(images)  ## images is a list of neighboring frames

            l1_loss = torch.nn.L1Loss()(out, gts)
            ssim_loss = ssim(out, gts, data_range=1.0)

            utils.eval_metrics(out, gts, psnrs, ssims, lpipss)

    # Print progress
    print(
        f"Train Epoch: {epoch} [{bidx}/{len(train_loader)}]\tL1Loss: {l1_loss.item():.4f}\tSSIM: {ssim_loss.item():.4f}",
        flush=True
    )

    inpu_patch = np.clip(images.detach().squeeze().cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)
    pred_patch = np.clip(out.detach().squeeze().cpu().numpy().squeeze() * 255, 0, 255).astype(np.uint8)
    imwrite(f"{writer_loc}/epoch_{epoch}_inpu.tif", inpu_patch)
    imwrite(f"{writer_loc}/epoch_{epoch}_pred.tif", pred_patch)
    print(f"Epoch {epoch} log done. \n")

    # Save psnr & ssim
    save_fn = os.path.join(save_loc, "results.txt")
    with open(save_fn, "a") as f:
        f.write("For epoch=%d\t" % epoch)
        f.write("L1: %f, SSIM Loss: %f \n" % (l1_loss.item(), ssim_loss.item()))

    # Log to TensorBoard
    # timestep = epoch + 1
    # writer.add_scalar("Loss/test", loss.item(), timestep)
    # writer.add_scalar("PSNR/test", psnrs.avg, timestep)
    # writer.add_scalar("SSIM/test", ssims.avg, timestep)

    return losses["total"].avg, psnrs.avg, ssims.avg


""" Entry Point """


def main(args):

    if args.pretrained:
        ## For low data, it is better to load from a supervised pretrained model
        loadStateDict = torch.load(args.pretrained)["state_dict"]
        modelStateDict = model.state_dict()

        for k, v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading ", k)
                modelStateDict[k] = v
            else:
                print("Not loading", k)

        model.load_state_dict(modelStateDict)

    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        train(args, epoch)

        test_loss, psnr, _ = test(args, epoch)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "lr": optimizer.param_groups[-1]["lr"],
            },
            save_loc,
            is_best,
            args.exp_name,
        )

        # update optimizer policy
        scheduler.step(test_loss)


if __name__ == "__main__":
    main(args)
