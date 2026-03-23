"""
Central configuration for SwinIR Image Enhancement App.
All task definitions, model URLs, and app-wide settings live here.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "pretrained_models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Base URL for downloading pretrained models
SWINIR_RELEASE_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"

# ─────────────────────────────────────────────
# Model registry: task → variant → config
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    # ── 1. Classical Super-Resolution ────────
    "classical_sr": {
        "x2": {
            "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
            "scale": 2,
            "in_chans": 3,
            "img_size": 48,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "x3": {
            "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth",
            "scale": 3,
            "in_chans": 3,
            "img_size": 48,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "x4": {
            "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
            "scale": 4,
            "in_chans": 3,
            "img_size": 48,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "x8": {
            "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
            "scale": 8,
            "in_chans": 3,
            "img_size": 48,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "param_key": "params",
        },
    },

    # ── 2. Real-World Super-Resolution ───────
    "real_sr": {
        "x4": {
            "filename": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
            "scale": 4,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "nearest+conv",
            "resi_connection": "1conv",
            "param_key": "params_ema",
            "large_model": False,
        },
        "x4_large": {
            "filename": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            "scale": 4,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6],
            "embed_dim": 240,
            "num_heads": [8, 8, 8, 8, 8, 8, 8, 8, 8],
            "mlp_ratio": 2,
            "upsampler": "nearest+conv",
            "resi_connection": "3conv",
            "param_key": "params_ema",
            "large_model": True,
        },
    },

    # ── 3. Grayscale Denoising ───────────────
    "gray_dn": {
        "noise15": {
            "filename": "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
            "scale": 1,
            "noise": 15,
            "in_chans": 1,
            "img_size": 128,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "noise25": {
            "filename": "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
            "scale": 1,
            "noise": 25,
            "in_chans": 1,
            "img_size": 128,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "noise50": {
            "filename": "004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth",
            "scale": 1,
            "noise": 50,
            "in_chans": 1,
            "img_size": 128,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
    },

    # ── 4. Color Denoising ───────────────────
    "color_dn": {
        "noise15": {
            "filename": "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
            "scale": 1,
            "noise": 15,
            "in_chans": 3,
            "img_size": 128,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "noise25": {
            "filename": "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
            "scale": 1,
            "noise": 25,
            "in_chans": 3,
            "img_size": 128,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "noise50": {
            "filename": "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth",
            "scale": 1,
            "noise": 50,
            "in_chans": 3,
            "img_size": 128,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
    },

    # ── 5. JPEG Artifact Reduction (Gray) ────
    "jpeg_car": {
        "q10": {
            "filename": "006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth",
            "scale": 1,
            "jpeg": 10,
            "in_chans": 1,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "q20": {
            "filename": "006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth",
            "scale": 1,
            "jpeg": 20,
            "in_chans": 1,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "q30": {
            "filename": "006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth",
            "scale": 1,
            "jpeg": 30,
            "in_chans": 1,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "q40": {
            "filename": "006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
            "scale": 1,
            "jpeg": 40,
            "in_chans": 1,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
    },

    # ── 6. JPEG Artifact Reduction (Color) ───
    "color_jpeg_car": {
        "q10": {
            "filename": "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth",
            "scale": 1,
            "jpeg": 10,
            "in_chans": 3,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "q20": {
            "filename": "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth",
            "scale": 1,
            "jpeg": 20,
            "in_chans": 3,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "q30": {
            "filename": "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth",
            "scale": 1,
            "jpeg": 30,
            "in_chans": 3,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
        "q40": {
            "filename": "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
            "scale": 1,
            "jpeg": 40,
            "in_chans": 3,
            "img_size": 126,
            "window_size": 7,
            "img_range": 255.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "",
            "resi_connection": "1conv",
            "param_key": "params",
        },
    },
}

# ─────────────────────────────────────────────
# Flask / app settings
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {
    # JPEG variants
    "jpg", "jpeg", "jpe", "jfif",
    # PNG
    "png",
    # BMP / DIB
    "bmp", "dib",
    # TIFF variants
    "tif", "tiff",
    # WebP
    "webp",
    # Portable bitmap family (OpenCV-supported)
    "pbm", "pgm", "ppm", "pnm",
    # High-dynamic range
    "hdr", "pic", "exr",
    # JPEG 2000
    "jp2", "j2k",
    # GIF (read-only, first frame)
    "gif",
    # Windows icon
    "ico",
    # AVIF / HEIF (requires appropriate OpenCV build)
    "avif", "heic", "heif",
}
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB

# Tile inference: use tiles for large images to save memory
DEFAULT_TILE = 512
DEFAULT_TILE_OVERLAP = 32

# Training defaults
TRAIN_DEFAULTS = {
    "batch_size": 8,
    "patch_size": 64,
    "lr": 2e-4,
    "lr_min": 1e-6,
    "epochs": 100,
    "save_every": 10,
    "eval_every": 5,
    "num_workers": 4,
    "seed": 42,
}