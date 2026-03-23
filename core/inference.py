"""
core/inference.py
─────────────────
Handles image preprocessing, tiled/whole-image inference, and
postprocessing for every SwinIR task.

KEY RULE: The user's uploaded image is ALWAYS treated as the LQ (input)
directly. We never artificially degrade it before inference — that was
only correct for benchmark evaluation against paired datasets.
"""

import cv2
import numpy as np
import torch

from config import DEFAULT_TILE, DEFAULT_TILE_OVERLAP


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def run_inference(
    img_path: str,
    model: torch.nn.Module,
    task: str,
    cfg: dict,
    tile: int = None,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    device: torch.device = None,
) -> np.ndarray:
    """
    End-to-end pipeline: load image → preprocess → infer → postprocess.

    Returns
    -------
    (out_uint8, None)
    out_uint8 : np.ndarray  uint8 HxWxC (BGR) or HxW (gray) ready for cv2.imwrite.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window_size = cfg["window_size"]
    scale       = cfg["scale"]

    img_lq = _load_image(img_path, task, cfg)

    # ── HWC-BGR / HW1 → CHW-RGB tensor ─────────────────────────────────────
    if img_lq.ndim == 2:
        # Pure grayscale HW → 1HW
        img_tensor = torch.from_numpy(img_lq[np.newaxis, ...]).float().unsqueeze(0)
    elif img_lq.shape[2] == 1:
        # Single-channel HW1 → 1CHW
        img_tensor = torch.from_numpy(img_lq[:, :, 0][np.newaxis, ...]).float().unsqueeze(0)
    else:
        # Colour HWC-BGR → CHW-RGB
        img_tensor = torch.from_numpy(
            img_lq[:, :, [2, 1, 0]].transpose(2, 0, 1)
        ).float().unsqueeze(0)

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # Pad so H and W are multiples of window_size
        _, _, h_old, w_old = img_tensor.shape
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, : h_old + h_pad, :]
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, : w_old + w_pad]

        output = _tile_or_whole(img_tensor, model, tile, tile_overlap, window_size, scale)
        output = output[..., : h_old * scale, : w_old * scale]

    # ── tensor → uint8 numpy ────────────────────────────────────────────────
    out_np = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if out_np.ndim == 3:
        # CHW-RGB → HWC-BGR
        out_np = np.transpose(out_np[[2, 1, 0], :, :], (1, 2, 0))
    out_uint8 = (out_np * 255.0).round().astype(np.uint8)

    return out_uint8, None   # img_gt is always None in user-facing inference


# ──────────────────────────────────────────────────────────────
# Image loading — input is always treated as LQ directly
# ──────────────────────────────────────────────────────────────

def _load_image(img_path: str, task: str, cfg: dict) -> np.ndarray:
    """
    Load the uploaded image as the LQ input for the model.
    Returns float32 numpy array:
      - Colour tasks  → HWC BGR,  values in [0, 1]
      - Gray tasks    → HW1,      values in [0, 1]
    """

    # ── Super Resolution (all variants) ─────────────────────────────────────
    # The uploaded image is the LQ; the model will upscale it.
    if task in ("classical_sr", "lightweight_sr", "real_sr"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        return img.astype(np.float32) / 255.0

    # ── Colour Denoising ─────────────────────────────────────────────────────
    # The uploaded image is the noisy input; the model cleans it.
    elif task == "color_dn":
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        return img.astype(np.float32) / 255.0

    # ── Grayscale Denoising ──────────────────────────────────────────────────
    elif task == "gray_dn":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        return img.astype(np.float32)[:, :, np.newaxis] / 255.0   # HW1

    # ── Colour JPEG Artifact Reduction ───────────────────────────────────────
    # The uploaded image is assumed to be a JPEG-compressed image.
    elif task == "color_jpeg_car":
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        return img.astype(np.float32) / 255.0

    # ── Grayscale JPEG Artifact Reduction ────────────────────────────────────
    elif task == "jpeg_car":
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)[:, :, np.newaxis] / 255.0   # HW1

    else:
        raise ValueError(f"Unknown task: {task}")


# ──────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────

def _tile_or_whole(
    img_tensor: torch.Tensor,
    model: torch.nn.Module,
    tile: int,
    tile_overlap: int,
    window_size: int,
    scale: int,
) -> torch.Tensor:
    """Run model on the whole image, or tile-by-tile to save memory."""
    if tile is None:
        return model(img_tensor)

    b, c, h, w = img_tensor.shape
    tile = min(tile, h, w)
    # Snap tile down to the nearest multiple of window_size (e.g. 500 → 496 for ws=8)
    tile = max(window_size, (tile // window_size) * window_size)

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    E = torch.zeros(b, c, h * scale, w * scale, dtype=img_tensor.dtype, device=img_tensor.device)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch  = img_tensor[..., h_idx: h_idx + tile, w_idx: w_idx + tile]
            out_patch = model(in_patch)
            out_mask  = torch.ones_like(out_patch)
            hs, he = h_idx * scale, (h_idx + tile) * scale
            ws, we = w_idx * scale, (w_idx + tile) * scale
            E[..., hs:he, ws:we].add_(out_patch)
            W[..., hs:he, ws:we].add_(out_mask)

    return E.div_(W)