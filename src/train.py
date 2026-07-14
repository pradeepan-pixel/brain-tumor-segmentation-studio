"""
train.py — Train the 3D UNet on BraTS patches.

Memory-saving techniques used (all matter on a 4-6GB RTX 3050):
  1. Patch-based data (see data_loader.py) — never load a full volume into the model
  2. Mixed precision (torch.cuda.amp) — roughly halves activation memory + speeds up conv3d
  3. Small batch size (default 1) with gradient accumulation to simulate a larger
     effective batch without the memory cost
  4. torch.cuda.empty_cache() is NOT needed in the loop — PyTorch's caching
     allocator handles reuse; calling it repeatedly actually slows things down

Run:
    python src/train.py --dataset_path dataset/archive/BraTS2021 --epochs 50 --patch_size 96 96 96
"""

import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from model import UNet3D
from data_loader import BraTSDataset
from utils import DiceBCELoss, dice_score, save_checkpoint, load_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                         help="Path to folder containing per-patient BraTS subfolders")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1,
                         help="Keep this at 1-2 on a 4-6GB GPU with patch_size 96")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                         help="Effective batch size = batch_size * accumulation_steps")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--base_channels", type=int, default=16,
                         help="Lower (e.g. 8) if you still OOM; raise (e.g. 32) if you have headroom")
    parser.add_argument("--use_checkpointing", action="store_true",
                         help="Enable gradient checkpointing to save more VRAM at the cost of speed")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--samples_per_volume", type=int, default=2,
                         help="Random patches drawn per patient per epoch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint .pth file to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    dataset = BraTSDataset(
        args.dataset_path,
        patch_size=tuple(args.patch_size),
        mode="train",
        samples_per_volume=args.samples_per_volume,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = UNet3D(
        in_channels=4, out_channels=1,
        base_channels=args.base_channels,
        use_checkpointing=args.use_checkpointing,
    ).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer, map_location=device)
        print(f"[train] Resumed from {args.resume} at epoch {start_epoch}, best_dice={best_dice:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        t0 = time.time()

        optimizer.zero_grad()

        for step, (image, mask) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(image)
                loss = criterion(logits, mask)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                batch_dice = dice_score(logits.float(), mask)

            loss_meter.update(loss.item() * args.accumulation_steps, n=image.size(0))
            dice_meter.update(batch_dice, n=image.size(0))

            if step % 10 == 0:
                print(f"  epoch {epoch+1}/{args.epochs} step {step}/{len(loader)} "
                      f"loss={loss_meter.avg:.4f} dice={dice_meter.avg:.4f}")

        scheduler.step()
        elapsed = time.time() - t0
        print(f"[epoch {epoch+1}] loss={loss_meter.avg:.4f} dice={dice_meter.avg:.4f} "
              f"time={elapsed:.1f}s lr={scheduler.get_last_lr()[0]:.2e}")

        # Always save the latest checkpoint (for resuming)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
                "base_channels": args.base_channels,
            },
            args.checkpoint_dir,
            filename="last.pth",
        )

        # Save best checkpoint separately
        if dice_meter.avg > best_dice:
            best_dice = dice_meter.avg
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "base_channels": args.base_channels,
                },
                args.checkpoint_dir,
                filename="best.pth",
            )
            print(f"  -> new best model saved (dice={best_dice:.4f})")

    print("[train] Training complete.")


if __name__ == "__main__":
    main()
