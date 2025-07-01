import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tifffile import imwrite
from model.shiftnet import GShiftNet
from configs.supervised_denoise_config import get_config
from dataset import get_supervised_dataloader
from utils import utils
from utils.loss import Loss

##### Parse CmdLine Arguments & Setup #####
args = get_config()
print(args)
device = torch.device("cuda" if args.cuda else "cpu")

criterion = Loss(args).to(device)

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
train_loader = get_supervised_dataloader(
    Path(args.data_root),
    Path(args.data_root),
    "train",
    patch_size = (5, 128, 128),
    num_samples_per_epoch = 540,
    use_fast_slow_emulation = (args.use_fast_slow == 1),
)
val_loader = get_supervised_dataloader(
    Path(args.val_data_root),
    Path(args.val_data_root),
    "val",
    patch_size = (5, 128, 128),
    num_samples_per_epoch = 32,
    use_fast_slow_emulation = (args.use_fast_slow == 1),
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
        for i, data_batch in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch}")):
            source = data_batch['source'].to(device)
            target = data_batch['target'].to(device)

            # Forward pass
            prediction = model(source)
            total_loss, individual_losses = criterion(prediction, target)

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
    print(f"--- Epoch {epoch} Validation Summary ---")
    print(f"Avg Loss: {losses['total'].avg:.4f} | Avg PSNR: {psnrs.avg:.4f} | Avg SSIM: {ssims.avg:.4f}")
    writer.add_scalar("Loss/total_val", losses["total"].avg, epoch)
    writer.add_scalar("Metrics/PSNR_val", psnrs.avg, epoch)
    writer.add_scalar("Metrics/SSIM_val", ssims.avg, epoch)

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