"""
core/model_manager.py
─────────────────────
Downloads pretrained SwinIR weights on demand, instantiates models,
and caches loaded models in memory to avoid repeated disk I/O.
"""

import os
import logging
import requests
import torch

from config import MODEL_CONFIGS, MODELS_DIR, SWINIR_RELEASE_URL
from models.network_swinir import SwinIR

logger = logging.getLogger(__name__)

# In-process model cache:  (task, variant) → torch.nn.Module
_model_cache: dict = {}


def get_model_path(task: str, variant: str) -> str:
    """Return the local path for a pretrained model file."""
    cfg = _get_cfg(task, variant)
    return os.path.join(MODELS_DIR, cfg["filename"])


def _get_cfg(task: str, variant: str) -> dict:
    if task not in MODEL_CONFIGS:
        raise ValueError(f"Unknown task '{task}'. Valid: {list(MODEL_CONFIGS)}")
    task_cfg = MODEL_CONFIGS[task]
    if variant not in task_cfg:
        raise ValueError(f"Unknown variant '{variant}' for task '{task}'. "
                         f"Valid: {list(task_cfg)}")
    return task_cfg[variant]


def download_model(task: str, variant: str, progress_cb=None) -> str:
    """
    Download the pretrained weights if not already present.
    Returns the local file path.
    progress_cb(downloaded_bytes, total_bytes) is called during download.
    """
    cfg = _get_cfg(task, variant)
    filename = cfg["filename"]
    local_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(local_path):
        logger.info(f"Model already cached: {local_path}")
        return local_path

    os.makedirs(MODELS_DIR, exist_ok=True)
    url = f"{SWINIR_RELEASE_URL}/{filename}"
    logger.info(f"Downloading model: {url}")

    tmp_path = local_path + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total:
                        progress_cb(downloaded, total)
        os.rename(tmp_path, local_path)
        logger.info(f"Saved to {local_path}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    return local_path


def build_model(cfg: dict) -> SwinIR:
    """Instantiate a SwinIR model from a config dict."""
    model = SwinIR(
        upscale=cfg["scale"],
        in_chans=cfg["in_chans"],
        img_size=cfg["img_size"],
        window_size=cfg["window_size"],
        img_range=cfg["img_range"],
        depths=cfg["depths"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        upsampler=cfg["upsampler"],
        resi_connection=cfg["resi_connection"],
    )
    return model


def load_model(task: str, variant: str, device: torch.device = None) -> SwinIR:
    """
    Return a loaded, eval-mode SwinIR model (with in-memory caching).
    Will download weights automatically if missing.
    """
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    cache_key = (task, variant)
    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key]

    # Download if needed
    local_path = download_model(task, variant)

    cfg = _get_cfg(task, variant)
    model = build_model(cfg)

    # Load weights
    param_key = cfg.get("param_key", "params")
    try:
        state_dict = torch.load(local_path, map_location=device, weights_only=True)
    except TypeError:
        # weights_only not supported in this torch version
        state_dict = torch.load(local_path, map_location=device)
    if param_key in state_dict:
        state_dict = state_dict[param_key]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    _model_cache[cache_key] = model
    logger.info(f"Loaded model ({task}/{variant}) on {device}")
    return model


def clear_cache():
    """Free all cached models from memory."""
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared.")


def list_downloaded_models() -> list[dict]:
    """Return info about every model file currently on disk."""
    results = []
    for task, variants in MODEL_CONFIGS.items():
        for variant, cfg in variants.items():
            path = os.path.join(MODELS_DIR, cfg["filename"])
            results.append({
                "task": task,
                "variant": variant,
                "filename": cfg["filename"],
                "downloaded": os.path.exists(path),
                "size_mb": round(os.path.getsize(path) / 1e6, 1) if os.path.exists(path) else 0,
            })
    return results