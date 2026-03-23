"""
utils/image_utils.py
────────────────────
Utility functions for image loading, saving, format conversion, and
building side-by-side comparison images.
"""

import os
import uuid
import cv2
import numpy as np
from PIL import Image


ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "jpe", "jfif",
    "png",
    "bmp", "dib",
    "tif", "tiff",
    "webp",
    "pbm", "pgm", "ppm", "pnm",
    "hdr", "pic", "exr",
    "jp2", "j2k",
    "gif",
    "ico",
    "avif", "heic", "heif",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_unique_filename(original: str) -> str:
    ext = original.rsplit(".", 1)[-1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


def read_image_rgb(path: str) -> np.ndarray:
    """Read image as uint8 RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img_bgr: np.ndarray, save_path: str) -> None:
    """Save a BGR uint8 numpy array to disk as PNG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def make_comparison(orig_path: str, output: np.ndarray,
                    max_width: int = 1024) -> np.ndarray:
    """
    Create a side-by-side comparison image (original | enhanced).
    Handles both grayscale (HW or HW1) and colour (HWC BGR) outputs.
    """
    orig_bgr = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    out_h, out_w = output.shape[:2]

    # ── Normalise output to 3-channel BGR ───────────────────────────────────
    if output.ndim == 2:
        # Pure grayscale HW → HWC BGR
        out_bgr = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif output.shape[2] == 1:
        # Single-channel HW1 → HWC BGR
        out_bgr = cv2.cvtColor(output[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        out_bgr = output

    # ── Resize original to output height ────────────────────────────────────
    orig_h, orig_w = orig_bgr.shape[:2]
    new_orig_w = max(1, int(orig_w * out_h / orig_h))
    orig_resized = cv2.resize(orig_bgr, (new_orig_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    # If output is grayscale, also convert original to grayscale for a fair visual
    if output.ndim == 2 or (output.ndim == 3 and output.shape[2] == 1):
        orig_resized = cv2.cvtColor(
            cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )

    comparison = np.hstack([orig_resized, out_bgr])

    # Downscale if too wide
    if comparison.shape[1] > max_width:
        scale = max_width / comparison.shape[1]
        new_w = max_width
        new_h = max(1, int(comparison.shape[0] * scale))
        comparison = cv2.resize(comparison, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return comparison


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR uint8 image to a base64 PNG data URI."""
    import base64
    _, buffer = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_image_info(path: str) -> dict:
    """Return basic metadata about an image file."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {}
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    size_kb = round(os.path.getsize(path) / 1024, 1)
    return {"width": w, "height": h, "channels": channels, "size_kb": size_kb}