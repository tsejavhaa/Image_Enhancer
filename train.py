"""
train.py
────────
Fine-tune any SwinIR task on your own paired dataset.

Directory structure expected:
    data/
      train/
        hr/    high-resolution ground-truth images
        lr/    (optional) pre-degraded low-resolution images
      val/
        hr/
        lr/    (optional)

Usage examples:
    # Fine-tune classical SR x4, using synthesised LQ
    python train.py --task classical_sr --variant x4

    # Fine-tune colour denoiser noise-25 on custom data
    python train.py --task color_dn --variant noise25 --epochs 50 --batch_size 4

    # Resume from checkpoint
    python train.py --task classical_sr --variant x2 --resume checkpoints/classical_sr_x2_epoch10.pth
"""

import argparse
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import MODEL_CONFIGS, MODELS_DIR, TRAIN_DEFAULTS, DATA_DIR
from core.model_manager import build_model, download_model
from utils.metrics import calculate_psnr, calculate_ssim
from utils.logger import setup_logger

logger = setup_logger("train")


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class SwinIRDataset(Dataset):
    """
    Paired HR/LR dataset.  If no LR folder is provided, degradation
    is synthesised on-the-fly matching the target task.
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, split: str, task: str, cfg: dict, patch_size: int = 64):
        self.task = task
        self.cfg = cfg
        self.patch_size = patch_size
        self.scale = cfg["scale"]

        split_dir = Path(DATA_DIR) / split
        self.hr_dir = split_dir / "hr"
        self.lr_dir = split_dir / "lr" if (split_dir / "lr").exists() else None

        self.hr_files = sorted(
            p for p in self.hr_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS
        )
        if not self.hr_files:
            raise FileNotFoundError(f"No HR images found in {self.hr_dir}")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx: int):
        hr_path = self.hr_files[idx]
        hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # ── synthesise or load LR ──────────────────────────────
        if self.lr_dir:
            lr_path = self.lr_dir / hr_path.name
            if lr_path.exists():
                lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            else:
                lr = self._degrade(hr)
        else:
            lr = self._degrade(hr)

        # ── random paired crop ─────────────────────────────────
        lr, hr = self._random_crop(lr, hr)

        # ── augment ───────────────────────────────────────────
        lr, hr = self._augment(lr, hr)

        # ── to tensor ─────────────────────────────────────────
        lr_t = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1)))
        hr_t = torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1)))
        return lr_t, hr_t

    # ── helpers ────────────────────────────────────────────────
    def _degrade(self, hr: np.ndarray) -> np.ndarray:
        task = self.task
        if task in ("classical_sr", "lightweight_sr", "real_sr"):
            h, w = hr.shape[:2]
            s = self.scale
            lr = cv2.resize(hr, (w // s, h // s), interpolation=cv2.INTER_CUBIC)
            return np.clip(lr, 0, 1)
        elif task == "gray_dn":
            gray = cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255.0
            noise = np.random.normal(0, self.cfg["noise"] / 255.0, gray.shape).astype(np.float32)
            lr = (gray + noise)[:, :, np.newaxis]
            return np.repeat(lr, 3, axis=2)   # keep 3-ch for simplicity
        elif task == "color_dn":
            noise = np.random.normal(0, self.cfg["noise"] / 255.0, hr.shape).astype(np.float32)
            return np.clip(hr + noise, 0, 1)
        elif task in ("jpeg_car", "color_jpeg_car"):
            hr_u8 = (hr * 255).astype(np.uint8)
            q = self.cfg["jpeg"]
            _, enc = cv2.imencode(".jpg", hr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            lr = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            return lr
        return hr.copy()

    def _random_crop(self, lr: np.ndarray, hr: np.ndarray):
        ps = self.patch_size
        s = self.scale
        lr_ps = ps // s if self.task in ("classical_sr", "lightweight_sr", "real_sr") else ps

        lh, lw = lr.shape[:2]
        if lh < lr_ps or lw < lr_ps:
            lr = cv2.resize(lr, (max(lw, lr_ps), max(lh, lr_ps)))
            hr = cv2.resize(hr, (max(lw, lr_ps) * s, max(lh, lr_ps) * s)) \
                if s > 1 else cv2.resize(hr, (max(lw, ps), max(lh, ps)))
            lh, lw = lr.shape[:2]

        y = random.randint(0, lh - lr_ps)
        x = random.randint(0, lw - lr_ps)
        lr_crop = lr[y: y + lr_ps, x: x + lr_ps]

        if s > 1:
            hr_crop = hr[y * s: (y + lr_ps) * s, x * s: (x + lr_ps) * s]
        else:
            hr_crop = hr[y: y + ps, x: x + ps]

        return lr_crop, hr_crop

    @staticmethod
    def _augment(lr: np.ndarray, hr: np.ndarray):
        if random.random() > 0.5:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)
        if random.random() > 0.5:
            lr = np.flipud(lr)
            hr = np.flipud(hr)
        k = random.randint(0, 3)
        lr = np.rot90(lr, k)
        hr = np.rot90(hr, k)
        return lr.copy(), hr.copy()


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── validate task/variant ─────────────────────────────────
    if args.task not in MODEL_CONFIGS:
        raise ValueError(f"Unknown task: {args.task}")
    if args.variant not in MODEL_CONFIGS[args.task]:
        raise ValueError(f"Unknown variant '{args.variant}' for task '{args.task}'")

    cfg = MODEL_CONFIGS[args.task][args.variant]

    # ── build model ────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # Load pretrained weights (download if needed)
    if args.pretrained:
        model_path = download_model(args.task, args.variant)
        state = torch.load(model_path, map_location=device)
        param_key = cfg.get("param_key", "params")
        state = state[param_key] if param_key in state else state
        model.load_state_dict(state, strict=True)
        logger.info(f"Loaded pretrained weights from {model_path}")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from epoch {start_epoch}: {args.resume}")

    # ── datasets & loaders ────────────────────────────────────
    train_ds = SwinIRDataset("train", args.task, cfg, patch_size=args.patch_size)
    val_ds   = SwinIRDataset("val",   args.task, cfg, patch_size=args.patch_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
    )

    logger.info(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ── loss / optimizer / scheduler ─────────────────────────
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_psnr = 0.0

    # ── training loop ─────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            out = model(lr_batch)
            loss = criterion(out, hr_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        logger.info(
            f"── Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        # ── evaluation ────────────────────────────────────────
        if (epoch + 1) % args.eval_every == 0:
            psnr, ssim = evaluate(model, val_loader, cfg, device)
            logger.info(f"   Val PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

            if psnr > best_psnr:
                best_psnr = psnr
                best_path = os.path.join(
                    args.checkpoint_dir,
                    f"{args.task}_{args.variant}_best.pth"
                )
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "psnr": psnr, "ssim": ssim}, best_path)
                logger.info(f"   ★ New best saved: {best_path}")

        # ── periodic checkpoint ───────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"{args.task}_{args.variant}_epoch{epoch+1}.pth"
            )
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)
            logger.info(f"   Checkpoint saved: {ckpt_path}")

    logger.info(f"Training complete. Best Val PSNR: {best_psnr:.2f} dB")


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: dict,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    scale = cfg["scale"]
    window_size = cfg["window_size"]

    psnr_list, ssim_list = [], []

    for lr_t, hr_t in loader:
        lr_t = lr_t.to(device)
        hr_t = hr_t.to(device)

        _, _, h_old, w_old = lr_t.shape
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        lr_t = torch.cat([lr_t, torch.flip(lr_t, [2])], 2)[:, :, : h_old + h_pad, :]
        lr_t = torch.cat([lr_t, torch.flip(lr_t, [3])], 3)[:, :, :, : w_old + w_pad]

        out = model(lr_t)
        out = out[..., : h_old * scale, : w_old * scale]

        # Convert to uint8 for metric calculation
        def to_u8(t):
            arr = t.squeeze().float().cpu().clamp_(0, 1).numpy()
            if arr.ndim == 3:
                arr = np.transpose(arr[[2, 1, 0]], (1, 2, 0))
            return (arr * 255.0).round().astype(np.uint8)

        out_u8 = to_u8(out)
        gt_u8  = to_u8(hr_t)

        psnr_list.append(calculate_psnr(out_u8, gt_u8, crop_border=scale))
        ssim_list.append(calculate_ssim(out_u8, gt_u8, crop_border=scale))

    model.train()
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    d = TRAIN_DEFAULTS
    p = argparse.ArgumentParser(description="Fine-tune SwinIR on custom data")

    p.add_argument("--task",    required=True,
                   help="Task: classical_sr | real_sr | gray_dn | color_dn | jpeg_car | color_jpeg_car")
    p.add_argument("--variant", required=True,
                   help="Variant, e.g. x2 / x4 / noise25 / q40 …")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Initialise from official pretrained weights (default: True)")
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pth to resume from")
    p.add_argument("--epochs",        type=int,   default=d["epochs"])
    p.add_argument("--batch_size",    type=int,   default=d["batch_size"])
    p.add_argument("--patch_size",    type=int,   default=d["patch_size"])
    p.add_argument("--lr",            type=float, default=d["lr"])
    p.add_argument("--lr_min",        type=float, default=d["lr_min"])
    p.add_argument("--eval_every",    type=int,   default=d["eval_every"])
    p.add_argument("--save_every",    type=int,   default=d["save_every"])
    p.add_argument("--num_workers",   type=int,   default=d["num_workers"])
    p.add_argument("--checkpoint_dir", type=str,  default="checkpoints")
    p.add_argument("--seed",          type=int,   default=d["seed"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)
