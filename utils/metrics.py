"""
utils/metrics.py
────────────────
PSNR and SSIM metrics used during inference evaluation and training.
Based on the reference implementation from BasicSR / SwinIR.
"""

import cv2
import numpy as np


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, crop_border: int = 0,
                   test_y_channel: bool = False) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) in dB.

    Parameters
    ----------
    img1, img2 : np.ndarray  uint8 H×W or H×W×C
    crop_border : int  pixels to crop from each border before computing
    test_y_channel : bool  if True and input is color, convert to YCbCr and use Y only
    """
    assert img1.shape == img2.shape, "Images must have the same shape"
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel and img1.ndim == 3:
        img1 = _to_y_channel(img1)
        img2 = _to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, crop_border: int = 0,
                   test_y_channel: bool = False) -> float:
    """
    SSIM (Structural Similarity Index).

    Parameters same as calculate_psnr.
    For multi-channel images, SSIM is averaged over channels.
    """
    assert img1.shape == img2.shape
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel and img1.ndim == 3:
        img1 = _to_y_channel(img1)
        img2 = _to_y_channel(img2)

    if img1.ndim == 2:
        return _ssim_single(img1, img2)
    else:
        scores = [_ssim_single(img1[..., ch], img2[..., ch]) for ch in range(img1.shape[2])]
        return float(np.mean(scores))


def _ssim_single(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def _to_y_channel(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR image to Y channel (uint8)."""
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img_ycbcr[:, :, 0:1]


def compute_metrics(output: np.ndarray, gt: np.ndarray,
                    crop_border: int = 0) -> dict:
    """
    Compute full metric suite for a single image pair.
    Both arrays should be uint8.
    """
    results = {}
    results["psnr"] = calculate_psnr(output, gt, crop_border=crop_border)
    results["ssim"] = calculate_ssim(output, gt, crop_border=crop_border)

    if gt.ndim == 3:
        results["psnr_y"] = calculate_psnr(output, gt, crop_border=crop_border,
                                           test_y_channel=True)
        results["ssim_y"] = calculate_ssim(output, gt, crop_border=crop_border,
                                           test_y_channel=True)
    return results
