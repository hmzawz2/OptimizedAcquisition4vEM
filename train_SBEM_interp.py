import os
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn

import myutils
from loss import Loss
# from model.myFLAVR import FLAVR_model
from model.shiftnet import GShiftNet
from configs.SBEM_interp_config import get_SBEM_interp_config
from pytorch_msssim import ssim
from tifffile import imwrite
import torch.nn.functional as F


def load_checkpoint(args, model, optimizer, path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


##### Parse CmdLine Arguments #####
args = get_SBEM_interp_config()
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

from dataset import get_SBEM2_loader

train_loader = get_SBEM2_loader(
    "train",
    args.data_root,
    shuffle=True,
    num_workers=args.num_workers,
    dataset_num=6000,
    damage_ratio=args.damage_ratio,
    nbr_frames=args.nbr_frame,
    patch_size=256,
    downsample_ratio=args.sr_ratio,
    use_cn=False,
)
test_loader = get_SBEM2_loader(
    "test",
    args.val_data_root,
    shuffle=False,
    num_workers=args.num_workers,
    dataset_num=1000,
    nbr_frames=args.nbr_frame,
    patch_size=256,
    downsample_ratio=args.sr_ratio,
    use_cn=False,
)


print("Building model: %s" % args.model.lower())
# model = FLAVR_model(
#     n_inputs=args.nbr_frame, # no timestep
#     n_outputs=args.n_outputs,
#     joinType=args.joinType,
#     upmode=args.upmode,
# )
model = GShiftNet()
model = torch.nn.DataParallel(model).to(device)

##### Define Loss & Optimizer #####
criterion = Loss(args)

## ToDo: Different learning rate schemes for different parameters

optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)


def train(args, epoch):
    losses, psnrs, ssims, lpipss = myutils.init_meters(args.loss)

    model.train()
    # criterion.train()

    t = time.time()
    for bidx, data_dict in enumerate(tqdm(train_loader)):

        # Build input batch
        images = data_dict["input_volume"]
        gts = data_dict["gt_volume"]

        images = images.cuda()
        gts = gts.cuda()

        # for interpolation
        # images = F.upsample(images, scale_factor=(2, 1, 1), mode='trilinear', align_corners=True)

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
            # myutils.eval_metrics(out, gts, psnrs, ssims, lpipss)

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
            # losses, psnrs, ssims, lpipss = myutils.init_meters(args.loss)
            # t = time.time()


def test(args, epoch):
    print("Evaluating for epoch = %d" % epoch)
    losses, psnrs, ssims, lpipss = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()

    t = time.time()
    with torch.no_grad():
        for bidx, data_dict in enumerate(tqdm(test_loader)):
            images = data_dict["input_volume"]
            gts = data_dict["gt_volume"]

            # Build input batch
            images = images.cuda()
            gts = gts.cuda()

            # for interpolation
            # images = F.upsample(images, scale_factor=(2, 1, 1), mode='trilinear', align_corners=True)

            out = model(images)  ## images is a list of neighboring frames

            l1_loss = torch.nn.L1Loss()(out, gts)
            ssim_value = ssim(out, gts, data_range=1.0)

            # out = rearrange(out, 'n c d h w -> n (c d) h w')
            # gts = rearrange(gts, 'n c d h w -> n (c d) h w')

            # Save loss values
            # loss, loss_specific = criterion(out, gts)
            # for k, v in losses.items():
            #     if k != "total":
            #         v.update(loss_specific[k].item())
            # losses["total"].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gts, psnrs, ssims, lpipss)

    # Print progress
    print(
        f"Train Epoch: {epoch} [{bidx}/{len(train_loader)}]\tL1Loss: {l1_loss.item():.4f}\tSSIM: {ssim_value.item():.4f}",
        flush=True
    )

    inpu_patch = (images.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred_patch = (out.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
    imwrite(f"{writer_loc}/epoch_{epoch}_inpu.tif", inpu_patch)
    imwrite(f"{writer_loc}/epoch_{epoch}_pred.tif", pred_patch)
    print(f"Epoch {epoch} log done. \n")

    # Save psnr & ssim
    save_fn = os.path.join(save_loc, "results.txt")
    with open(save_fn, "a") as f:
        f.write("For epoch=%d\t" % epoch)
        f.write("L1: %f, SSIM Loss: %f \n" % (l1_loss.item(), ssim_value.item()))

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
        myutils.save_checkpoint(
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
