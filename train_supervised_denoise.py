# import os
# import time
# from pathlib import Path

# import numpy as np
# import torch
# from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from model.shiftnet import GShiftNet
# from configs.self_supervised_denoise_config import get_config
# from pytorch_msssim import ssim
# from tifffile import imwrite
# from utils import utils
# from utils.loss import Loss



# ##### Parse CmdLine Arguments #####
# args = get_config()
# print(args)
# device = torch.device("cuda" if args.cuda else "cpu")

# criterion = Loss(args).to(device).eval()

# save_loc = os.path.join(
#     args.checkpoint_dir, args.dataset, args.exp_name
# )
# if not os.path.exists(save_loc):
#     os.makedirs(save_loc)
# opts_file = os.path.join(save_loc, "opts.txt")
# with open(opts_file, "w") as fh:
#     fh.write(str(args))


# ##### TensorBoard & Misc Setup #####
# writer_loc = os.path.join(
#     args.tensorboard_dir, args.dataset, f"tensorboard_logs_{args.exp_name}"
# )
# writer = SummaryWriter(writer_loc)


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

# torch.manual_seed(args.random_seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.random_seed)

# from dataset import get_self_supervised_dataloader

# train_loader = get_self_supervised_dataloader(
#     Path(args.data_root),
#     "train",
#     patch_size = (5, 128, 128),
#     num_samples_per_epoch = 540,
# )
# test_loader = get_self_supervised_dataloader(
#     Path(args.val_data_root),
#     "test",
#     patch_size = (5, 128, 128),
#     num_samples_per_epoch = 32,
# )


# print("Building model: %s" % args.model.lower())

# model = GShiftNet()
# model = torch.nn.DataParallel(model).to(device)

# ##### Define Loss & Optimizer #####

# ## ToDo: Different learning rate schemes for different parameters

# optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=5, verbose=True
# )

# def train(args, epoch):

#     model.train()
#     loss_specifics = utils.init_losses(args.loss)

#     for bidx, data_dict in enumerate(train_loader):

#         # Build input batch
#         images = data_dict["source"]
#         gts = data_dict["target"]
    
#         images = images.cuda()
#         gts = gts.cuda()

#         # Forward
#         optimizer.zero_grad()
#         out1 = model(images)
#         out2 = model(gts)
        
#         # l1_loss = torch.nn.L1Loss()(out1, gts)
#         # ssim_loss = 1 - ssim(out1, gts, data_range=1.0)

#         total_loss, individual_loss = criterion(out1, gts)

#         total_loss.backward()
#         optimizer.step()


#         # Save loss values
#         for k, v in loss_specifics.items():
#             if k != "total":
#                 v.update(individual_loss[k].item())
#         loss_specifics["total"].update(total_loss.item())


#         # Calc metrics & print logs
#         if bidx % args.log_iter == 0:
#             # utils.eval_metrics(out, gts, psnrs, ssims, lpipss)

#             log_prefix = f"Epoch: {epoch} [{bidx}/{len(train_loader)}]"
#             log_items = [f"{name}: {tracker.val:.4f} ({tracker.avg:.4f})" for name, tracker in loss_specifics.items()]
#             log_string = f"{log_prefix}\t" + "\t".join(log_items)
#             print(log_string, flush=True)

#             # Log to TensorBoard
#             # timestep = epoch * len(train_loader) + bidx
#             # writer.add_scalar("Loss/train", loss.item(), timestep)
#             # writer.add_scalar("PSNR/train", psnrs.avg, timestep)
#             # writer.add_scalar("SSIM/train", ssims.avg, timestep)
#             # writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], timestep)

#     # Reset metrics
#     # losses, psnrs, ssims, lpipss = utils.init_meters(args.loss)



# def test(args, epoch):
#     print("Evaluating for epoch = %d" % epoch)
#     losses, psnrs, ssims = utils.init_meters(args.loss)
#     model.eval()

#     with torch.no_grad():
#         for bidx, data_dict in enumerate(tqdm(test_loader)):
            
#             images = data_dict["source"]
#             gts = data_dict["target"]
        
#             images = images.cuda()
#             gts = gts.cuda()

#             out = model(images)  ## images is a list of neighboring frames

#             l1_loss = torch.nn.L1Loss()(out, gts)
#             ssim_loss = ssim(out, gts, data_range=1.0)

#             # fulls = model(fulls)

#             utils.eval_metrics(out, gts, psnrs, ssims)

#     # Print progress
#     print(
#         f"Train Epoch: {epoch} [{bidx}/{len(train_loader)}]\tL1Loss: {l1_loss.item():.4f}\tSSIM: {ssim_loss.item():.4f}",
#         flush=True
#     )

#     inpu_patch = (np.clip(images.detach().squeeze().cpu().numpy().squeeze(), 0, 1) * 255).astype(np.uint8)
#     pred_patch = (np.clip(out.detach().squeeze().cpu().numpy().squeeze(), 0, 1) * 255).astype(np.uint8)
#     # clean_patch = (np.clip(cleans.detach().squeeze().cpu().numpy().squeeze(), 0, 1) * 255).astype(np.uint8)
#     # full_patch = (np.clip(fulls.detach().squeeze().cpu().numpy().squeeze(), 0, 1) * 255).astype(np.uint8)
#     imwrite(f"{writer_loc}/epoch_{epoch}_inpu.tif", inpu_patch)
#     imwrite(f"{writer_loc}/epoch_{epoch}_pred.tif", pred_patch)
#     # imwrite(f"{writer_loc}/epoch_{epoch}_clean.tif", clean_patch)
#     # imwrite(f"{writer_loc}/epoch_{epoch}_full.tif", full_patch)
#     print(f"Epoch {epoch} log done. \n")

#     # Save psnr & ssim
#     save_fn = os.path.join(save_loc, "results.txt")
#     with open(save_fn, "a") as f:
#         f.write("For epoch=%d\t" % epoch)
#         f.write("L1: %f, SSIM Loss: %f \n" % (l1_loss.item(), ssim_loss.item()))

#     # Log to TensorBoard
#     # timestep = epoch + 1
#     # writer.add_scalar("Loss/test", loss.item(), timestep)
#     # writer.add_scalar("PSNR/test", psnrs.avg, timestep)
#     # writer.add_scalar("SSIM/test", ssims.avg, timestep)

#     return losses["total"].avg, psnrs.avg, ssims.avg


# """ Entry Point """


# def main(args):

#     if args.pretrained:
#         ## For low data, it is better to load from a supervised pretrained model
#         loadStateDict = torch.load(args.pretrained)["state_dict"]
#         modelStateDict = model.state_dict()

#         for k, v in loadStateDict.items():
#             if v.shape == modelStateDict[k].shape:
#                 print("Loading ", k)
#                 modelStateDict[k] = v
#             else:
#                 print("Not loading", k)

#         model.load_state_dict(modelStateDict)

#     best_psnr = 0
#     for epoch in range(args.start_epoch, args.max_epoch):
#         train(args, epoch)

#         test_loss, psnr, _ = test(args, epoch)

#         # save checkpoint
#         is_best = psnr > best_psnr
#         best_psnr = max(psnr, best_psnr)
#         utils.save_checkpoint(
#             {
#                 "epoch": epoch,
#                 "state_dict": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "best_psnr": best_psnr,
#                 "lr": optimizer.param_groups[-1]["lr"],
#             },
#             save_loc,
#             is_best,
#             args.exp_name,
#         )

#         # update optimizer policy
#         scheduler.step(test_loss)


# if __name__ == "__main__":
#     main(args)


import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tifffile import imwrite

# Assume these are your project's modules
from model.shiftnet import GShiftNet
from configs.self_supervised_denoise_config import get_config
from dataset import get_self_supervised_dataloader
from utils import utils
from utils.loss import Loss

##### Parse CmdLine Arguments & Setup #####
args = get_config()
print(args)
device = torch.device("cuda" if args.cuda else "cpu")

criterion = Loss(args).to(device)

# Use pathlib for cleaner path management
save_loc = Path(args.checkpoint_dir) / args.dataset / args.exp_name
save_loc.mkdir(parents=True, exist_ok=True)
with open(save_loc / "opts.txt", "w") as fh:
    fh.write(str(args))

writer_loc = Path(args.tensorboard_dir) / args.dataset / f"tensorboard_{args.exp_name}"
writer = SummaryWriter(log_dir=writer_loc)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.random_seed)

##### Dataloaders #####
train_loader = get_self_supervised_dataloader(
    Path(args.data_root),
    "train",
    patch_size = (5, 128, 128),
    num_samples_per_epoch = 120,
)
val_loader = get_self_supervised_dataloader(
    Path(args.val_data_root),
    "val",
    patch_size = (5, 128, 128),
    num_samples_per_epoch = 32,
)

##### Model, Optimizer, Scheduler #####
print(f"Building model: {args.model}")
model = GShiftNet()
model = torch.nn.DataParallel(model).to(device)

optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)


def train(epoch: int, global_step: int):
    model.train()
    loss_trackers = utils.init_losses(args.loss)

    for i, data_batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}"), 1):
        source = data_batch['source'].to(device)
        target = data_batch['target'].to(device)

        # Forward pass
        optimizer.zero_grad()
        prediction = model(source)

        # Loss calculation and backward pass
        total_loss, individual_losses = criterion(prediction, target)
        total_loss.backward()
        optimizer.step()

        # Update loss trackers
        batch_size = source.size(0)
        for k, v in loss_trackers.items():
            if k != "total":
                v.update(individual_losses[k].item())
        loss_trackers["total"].update(total_loss.item())

        # Log to console and TensorBoard
        if i % args.log_iter == 0:
            current_step = global_step + i
            log_prefix = f"Epoch: {epoch} [{i}/{len(train_loader)}]"
            log_items = [f"{name}: {tracker.avg:.4f}" for name, tracker in loss_trackers.items()]
            tqdm.write(f"{log_prefix}\t" + "\t".join(log_items))

            for name, tracker in loss_trackers.items():
                writer.add_scalar(f"Loss/{name}_train", tracker.avg, current_step)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], current_step)
    
    return len(train_loader)


def validate(epoch: int):
    print(f"--- Validating for epoch {epoch} ---")
    model.eval()

    losses, psnrs, ssims = utils.init_meters(args.loss)

    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch}"), 1):
            source = data_batch['source'].to(device)
            target = data_batch['target'].to(device)

            # Forward pass
            prediction = model(source)
            total_loss, individual_losses = criterion(prediction, target)

            batch_size = source.size(0)
            for k, v in losses.items():
                if k != "total":
                    v.update(individual_losses[k].item())
            losses["total"].update(total_loss.item())
            utils.eval_metrics(target, prediction, psnrs, ssims)

            # Save visualization for the first batch only for consistent comparison
            if i == 0:
                vis_path = writer_loc / "visualizations"
                vis_path.mkdir(exist_ok=True)

                source_vis = (np.clip(source.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                target_vis = (np.clip(target.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                pred_vis = (np.clip(prediction.squeeze().cpu().numpy(), 0, 1) * 255).astype(np.uint8)

                imwrite(vis_path / f"epoch_{epoch:04d}_source.tif", source_vis)
                imwrite(vis_path / f"epoch_{epoch:04d}_prediction.tif", pred_vis)
                imwrite(vis_path / f"epoch_{epoch:04d}_target.tif", target_vis)

    # Print summary and log to TensorBoard
    # print(f"--- Epoch {epoch} Validation Summary ---")
    # print(f"Avg Loss: {avg_val_loss:.4f} | Avg PSNR: {avg_psnr:.4f} | Avg SSIM: {avg_ssim:.4f}")
    # writer.add_scalar("Loss/total_val", avg_val_loss, epoch)
    # writer.add_scalar("Metrics/PSNR_val", avg_psnr, epoch)
    # writer.add_scalar("Metrics/SSIM_val", avg_ssim, epoch)

    # Save average metrics to a text file
    with open(save_loc / "results.txt", "a") as f:
        f.write(f"Epoch: {epoch}\tPSNR: {psnrs.avg:.4f}\tSSIM: {ssims.avg:.4f}\n")

    return losses["total"].avg, psnrs.avg, ssims.avg


def main(args):
    # Load pretrained model if specified
    if args.pretrained:
        print(f"Loading pretrained model from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        # Add robust model loading logic here if needed
        model.load_state_dict(checkpoint['state_dict'])

    best_psnr = 0.0
    global_step = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        steps_this_epoch = train(epoch, global_step)
        global_step += steps_this_epoch

        val_loss, val_psnr, val_ssim = validate(epoch)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save checkpoint based on performance
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            print(f"*** New best PSNR: {best_psnr:.4f} at epoch {epoch} ***")
        
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_psnr,
            },
            save_loc,
            is_best,
        )

if __name__ == "__main__":
    main(args)