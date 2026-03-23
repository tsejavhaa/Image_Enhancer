"""
app.py
──────
Flask web application for SwinIR image enhancement.
Serves the UI, handles uploads, triggers inference, and streams progress.
"""

import json
import os
import threading
import time
import uuid

try:
    import psutil as _psutil
    # Prime the baseline counters at import time so the first real
    # measurement (200 ms later) has a valid delta to compare against.
    if _psutil:
        _psutil.cpu_percent(interval=None)
        _psutil.cpu_percent(interval=None, percpu=True)
except ImportError:
    _psutil = None

JOB_TIMEOUT = 20 * 60   # 20 minutes hard ceiling per job
from pathlib import Path

import cv2
import torch
from flask import (Flask, jsonify, render_template, request,
                   send_from_directory, Response, stream_with_context)

from config import (BASE_DIR, MODEL_CONFIGS, UPLOADS_DIR, OUTPUTS_DIR,
                    DEFAULT_TILE, DEFAULT_TILE_OVERLAP, MAX_CONTENT_LENGTH)
from core.inference import run_inference
from core.model_manager import (load_model, download_model,
                                list_downloaded_models, clear_cache)
from utils.image_utils import (allowed_file, secure_unique_filename,
                               save_image, make_comparison, image_to_base64,
                               get_image_info)
from utils.metrics import compute_metrics
from utils.logger import setup_logger

logger = setup_logger("app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOADS_DIR
app.config["OUTPUT_FOLDER"] = OUTPUTS_DIR

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

# ── in-progress job tracker ───────────────────────────────────
_jobs: dict[str, dict] = {}        # job_id → live status dict
_history_lock = threading.Lock()   # guards history.json reads/writes


def _load_history() -> list[dict]:
    """Load persisted job history from disk."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list[dict]) -> None:
    """Persist job history to disk (caller must hold _history_lock)."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")


def _append_to_history(entry: dict) -> None:
    """Thread-safe append of a completed job to history.json."""
    with _history_lock:
        history = _load_history()
        # Deduplicate by job_id
        history = [h for h in history if h.get("job_id") != entry.get("job_id")]
        history.insert(0, entry)      # newest first
        history = history[:200]       # cap at 200 entries
        _save_history(history)


# ─────────────────────────────────────────────────────────────
# Static pages
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/uploads/<path:filename>")
def upload_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)


# ─────────────────────────────────────────────────────────────
# API: task/variant catalogue
# ─────────────────────────────────────────────────────────────

@app.route("/api/tasks")
def api_tasks():
    """Return the full task/variant registry as JSON."""
    result = {}
    for task, variants in MODEL_CONFIGS.items():
        result[task] = {}
        for variant, cfg in variants.items():
            result[task][variant] = {
                "scale": cfg.get("scale", 1),
                "noise": cfg.get("noise"),
                "jpeg": cfg.get("jpeg"),
                "filename": cfg["filename"],
            }
    return jsonify(result)


@app.route("/api/models")
def api_models():
    """Return download status for every model."""
    return jsonify(list_downloaded_models())


# ─────────────────────────────────────────────────────────────
# API: image upload
# ─────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "files" not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist("files")
    uploaded = []

    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"File type not allowed: {f.filename}"}), 400

        fname = secure_unique_filename(f.filename)
        save_path = os.path.join(UPLOADS_DIR, fname)
        f.save(save_path)

        info = get_image_info(save_path)
        uploaded.append({
            "id": fname,
            "original_name": f.filename,
            "path": f"/uploads/{fname}",
            "info": info,
        })

    if not uploaded:
        return jsonify({"error": "No valid files uploaded"}), 400

    return jsonify({"files": uploaded})


# ─────────────────────────────────────────────────────────────
# API: start enhancement job
# ─────────────────────────────────────────────────────────────

@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    file_ids  = data.get("file_ids", [])
    task      = data.get("task")
    variant   = data.get("variant")
    use_tile  = data.get("use_tile", True)
    tile_size = int(data.get("tile_size", DEFAULT_TILE))
    device_id = data.get("device", "auto")   # "cpu" | "mps" | "cuda" | "auto"

    if not file_ids:
        return jsonify({"error": "No file_ids provided"}), 400
    if not task or not variant:
        return jsonify({"error": "task and variant are required"}), 400
    if task not in MODEL_CONFIGS or variant not in MODEL_CONFIGS[task]:
        return jsonify({"error": f"Unknown task/variant: {task}/{variant}"}), 400

    job_id = uuid.uuid4().hex
    # Map file_id → original filename for history display
    orig_names = {f["file_id"]: f.get("original_name", f["file_id"])
                  for f in (data.get("file_meta") or [])}

    _jobs[job_id] = {
        "status": "queued",
        "detail": "",           # human-readable sub-status
        "progress": 0,
        "total": len(file_ids),
        "results": [],
        "errors": [],
        "created_at": time.time(),
        "started_at": None,
        "elapsed_sec": 0,
        "orig_names":  orig_names,
    }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, file_ids, task, variant, use_tile, tile_size, device_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _run_job(job_id, file_ids, task, variant, use_tile, tile_size, device_id='auto'):
    """Background worker: loads model once, processes all files."""
    job = _jobs[job_id]
    job["status"]     = "loading_model"
    job["detail"]     = "Downloading / loading model weights…"
    job["started_at"] = time.time()

    # ── elapsed-time ticker (runs in its own daemon thread) ──────────────────
    def _tick():
        while job["status"] not in ("done", "error"):
            job["elapsed_sec"] = int(time.time() - job["started_at"])
            # Hard timeout guard
            if job["elapsed_sec"] > JOB_TIMEOUT:
                job["status"] = "error"
                job["errors"].append("Job timed out after 20 minutes.")
                logger.error(f"Job {job_id} timed out.")
                break
            time.sleep(1)
    threading.Thread(target=_tick, daemon=True).start()

    try:
        import torch as _torch
        if device_id == "mps" and hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            device = _torch.device("mps")
        elif device_id == "cuda" and _torch.cuda.is_available():
            device = _torch.device("cuda")
        elif device_id == "cpu":
            device = _torch.device("cpu")
        else:
            # auto: prefer MPS on Apple Silicon, then CUDA, then CPU
            if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                device = _torch.device("mps")
            elif _torch.cuda.is_available():
                device = _torch.device("cuda")
            else:
                device = _torch.device("cpu")
        job["detail"] = f"Loading model on {device}…"
        model = load_model(task, variant, device)
        cfg   = MODEL_CONFIGS[task][variant]
    except Exception as e:
        job["status"] = "error"
        job["detail"] = str(e)
        job["errors"].append(str(e))
        logger.error(f"Job {job_id}: model load failed – {e}")
        return

    job["status"] = "processing"
    tile = tile_size if use_tile else None

    for i, file_id in enumerate(file_ids):
        src_path = os.path.join(UPLOADS_DIR, file_id)
        if not os.path.exists(src_path):
            job["errors"].append(f"File not found: {file_id}")
            continue

        orig_name_display = job.get("orig_names", {}).get(file_id, Path(file_id).name)
        job["detail"] = f"Inferring {orig_name_display} ({i+1}/{len(file_ids)})…"
        logger.info(f"Job {job_id}: starting inference on {orig_name_display}")

        t_start = time.time()   # ← start timer before inference

        try:
            out, img_gt = run_inference(
                img_path=src_path,
                model=model,
                task=task,
                cfg=cfg,
                tile=tile,
                tile_overlap=DEFAULT_TILE_OVERLAP,
                device=device,
            )
            proc_sec = round(time.time() - t_start, 1)   # ← seconds for this image

            # Normalise output to 3-channel BGR for saving/display.
            # gray_dn returns HW (2-D); jpeg_car gray returns HW too.
            if out.ndim == 2:
                out_save = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            elif out.ndim == 3 and out.shape[2] == 1:
                out_save = cv2.cvtColor(out[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                out_save = out

            # Save enhanced image
            stem = Path(file_id).stem
            out_name = f"{stem}_{task}_{variant}.png"
            out_path = os.path.join(OUTPUTS_DIR, out_name)
            save_image(out_save, out_path)

            # Build side-by-side comparison
            cmp = make_comparison(src_path, out)   # make_comparison handles gray itself
            cmp_name = f"{stem}_{task}_{variant}_cmp.png"
            cmp_path = os.path.join(OUTPUTS_DIR, cmp_name)
            save_image(cmp, cmp_path)

            # Metrics (only when ground-truth is available)
            metrics = {}
            if img_gt is not None:
                gt_u8 = (img_gt.squeeze() * 255.0).round().astype("uint8")
                try:
                    metrics = compute_metrics(out_save, gt_u8, crop_border=cfg["scale"])
                except Exception:
                    pass

            out_info = get_image_info(out_path)
            orig_name = job.get("orig_names", {}).get(file_id, file_id)
            job["results"].append({
                "file_id":        file_id,
                "orig_name":      orig_name,
                "original_url":   f"/uploads/{file_id}",
                "output_url":     f"/outputs/{out_name}",
                "comparison_url": f"/outputs/{cmp_name}",
                "output_info":    out_info,
                "metrics":        metrics,
                "proc_sec":       proc_sec,   # ← per-image processing time
            })
            logger.info(f"Job {job_id}: {orig_name_display} done in {proc_sec}s")

        except Exception as e:
            job["errors"].append(f"{file_id}: {e}")
            logger.error(f"Job {job_id}: failed on {file_id} – {e}", exc_info=True)

        job["progress"] = i + 1

    job["status"] = "done"
    logger.info(f"Job {job_id} complete. {len(job['results'])} OK, "
                f"{len(job['errors'])} errors.")

    # Persist to history (include original filenames stored in job)
    _append_to_history({
        "job_id":      job_id,
        "task":        task,
        "variant":     variant,
        "status":      "done",
        "results":     job["results"],
        "errors":      job["errors"],
        "total":       job["total"],
        "elapsed_sec": job.get("elapsed_sec", 0),
        "completed_at": time.time(),
        "orig_names":  job.get("orig_names", {}),
    })


# ─────────────────────────────────────────────────────────────
# API: poll job status
# ─────────────────────────────────────────────────────────────

@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/job/<job_id>/stream")
def api_job_stream(job_id):
    """Server-Sent Events stream for live progress updates."""

    def generate():
        tick = 0
        while True:
            job = _jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                break
            yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in ("done", "error"):
                break
            time.sleep(1)
            tick += 1
            # Send an SSE comment every 15 s to prevent proxy/browser timeouts
            if tick % 15 == 0:
                yield ": keepalive\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
# API: model management
# ─────────────────────────────────────────────────────────────

@app.route("/api/models/download", methods=["POST"])
def api_download_model():
    data = request.get_json()
    task    = data.get("task")
    variant = data.get("variant")
    try:
        path = download_model(task, variant)
        return jsonify({"path": path, "ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/clear_cache", methods=["POST"])
def api_clear_cache():
    clear_cache()
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
# API: history
# ─────────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    """Return all completed jobs, newest first."""
    return jsonify(_load_history())


@app.route("/api/history/<job_id>", methods=["DELETE"])
def api_delete_job(job_id):
    """Remove a job entry from history (does NOT delete output files)."""
    with _history_lock:
        history = _load_history()
        before = len(history)
        history = [h for h in history if h.get("job_id") != job_id]
        _save_history(history)
    removed = before - len(history)
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/history/<job_id>/results/<path:filename>", methods=["DELETE"])
def api_delete_result(job_id, filename):
    """Delete a single result file (output + comparison) from disk and history."""
    deleted_files = []

    # Delete the output file and its companion comparison file
    for candidate in [filename, filename.replace(".png", "_cmp.png")]:
        fpath = os.path.join(OUTPUTS_DIR, candidate)
        if os.path.exists(fpath):
            os.remove(fpath)
            deleted_files.append(candidate)

    # Also remove the matching result entry from history
    with _history_lock:
        history = _load_history()
        for job in history:
            if job.get("job_id") == job_id:
                before = len(job.get("results", []))
                job["results"] = [
                    r for r in job.get("results", [])
                    if os.path.basename(r.get("output_url", "")) != filename
                ]
                # Remove the job entry entirely if no results remain
                if not job["results"] and before > 0:
                    history = [h for h in history if h.get("job_id") != job_id]
                break
        _save_history(history)

    return jsonify({"ok": True, "deleted_files": deleted_files})


@app.route("/api/history/clear", methods=["DELETE"])
def api_clear_history():
    """Wipe all history entries (files on disk are kept)."""
    with _history_lock:
        _save_history([])
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
# API: system monitor
# ─────────────────────────────────────────────────────────────

@app.route("/api/system")
def api_system():
    """Real-time system metrics.  Two separate cpu_percent calls with their
    own blocking intervals so each has a valid baseline."""
    GiB = 1024 ** 3
    data = {"gpu_pct": None, "cpu_pct": 0, "cpu_per_core": [],
            "cpu_cores": 1, "ram_pct": 0, "ram_used_gb": 0, "ram_total_gb": 0}

    if _psutil:
        try:
            vm = _psutil.virtual_memory()
            data["ram_pct"]      = round(vm.percent, 1)
            data["ram_used_gb"]  = round(vm.used  / GiB, 2)
            data["ram_total_gb"] = round(vm.total / GiB, 1)
            data["cpu_cores"]    = _psutil.cpu_count(logical=True) or 1
        except Exception as e:
            logger.warning(f"psutil RAM error: {e}")

        try:
            # Blocking call — sets its own start/end baseline internally.
            # 0.3 s is the minimum reliable window on macOS.
            cpu_total = _psutil.cpu_percent(interval=0.3)
            data["cpu_pct"] = round(cpu_total, 1)
        except Exception as e:
            logger.warning(f"psutil CPU total error: {e}")

        try:
            # Per-core: use a fresh 0.3 s interval too (NOT interval=None)
            # so it doesn't depend on a previous call's baseline.
            cores = _psutil.cpu_percent(interval=0.3, percpu=True)
            data["cpu_per_core"] = [round(x, 1) for x in cores]
        except Exception as e:
            logger.warning(f"psutil CPU per-core error: {e}")

    # GPU via torch (optional)
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            data["gpu_pct"]       = round(alloc / total * 100, 1)
            data["gpu_mem_used"]  = round(alloc / GiB, 2)
            data["gpu_mem_total"] = round(total / GiB, 1)
            data["gpu_name"]      = torch.cuda.get_device_properties(0).name
    except Exception:
        pass

    # Active job info
    active = [j for j in _jobs.values() if j["status"] not in ("done", "error")]
    data["active_jobs"]    = len(active)
    data["active_detail"]  = active[0].get("detail",      "") if active else ""
    data["active_elapsed"] = active[0].get("elapsed_sec",  0) if active else 0

    return jsonify(data)


# ─────────────────────────────────────────────────────────────
# API: available compute devices
# ─────────────────────────────────────────────────────────────

@app.route("/api/devices")
def api_devices():
    """Return which compute devices are available on this machine."""
    import torch
    devices = []

    # CPU — always available
    devices.append({
        "id":    "cpu",
        "label": "CPU",
        "available": True,
        "info": f"{_psutil.cpu_count(logical=False) or '?'} cores" if _psutil else "Available",
    })

    # MPS — Apple Silicon / Metal
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    devices.append({
        "id":    "mps",
        "label": "MPS",
        "available": mps_ok,
        "info": "Apple Metal" if mps_ok else "Not available",
    })

    # CUDA GPU
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        try:
            gpu_name = torch.cuda.get_device_properties(0).name
            vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            gpu_info = f"{gpu_name}  {vram_gb} GB"
        except Exception:
            gpu_info = "Available"
    else:
        gpu_info = "Not available"
    devices.append({
        "id":    "cuda",
        "label": "GPU",
        "available": cuda_ok,
        "info": gpu_info,
    })

    # Recommend the best device
    if cuda_ok:
        recommended = "cuda"
    elif mps_ok:
        recommended = "mps"
    else:
        recommended = "cpu"

    return jsonify({"devices": devices, "recommended": recommended})


# ─────────────────────────────────────────────────────────────
# DEBUG: raw psutil diagnostics — visit /api/system/debug in browser
# ─────────────────────────────────────────────────────────────

@app.route("/api/system/debug")
def api_system_debug():
    import sys, platform
    out = {
        "python": sys.version,
        "platform": platform.platform(),
        "psutil_available": _psutil is not None,
    }
    if not _psutil:
        return jsonify(out)

    out["psutil_version"] = _psutil.__version__

    # Test 1: raw virtual_memory
    vm = _psutil.virtual_memory()
    out["vm_total_bytes"]  = vm.total
    out["vm_used_bytes"]   = vm.used
    out["vm_percent"]      = vm.percent
    out["vm_used_gib"]     = round(vm.used  / (1024**3), 2)
    out["vm_total_gib"]    = round(vm.total / (1024**3), 1)

    # Test 2: cpu_percent with explicit interval
    import time
    t0 = time.time()
    cpu1 = _psutil.cpu_percent(interval=0.5)   # block 0.5 s
    elapsed = round(time.time() - t0, 3)
    out["cpu_interval_0_5"]        = cpu1
    out["cpu_interval_0_5_elapsed"] = elapsed

    # Test 3: per-core (uses same baseline as above)
    cores = _psutil.cpu_percent(interval=None, percpu=True)
    out["cpu_per_core"]  = cores
    out["cpu_core_count"] = len(cores)

    # Test 4: cpu_count
    out["cpu_count_logical"]  = _psutil.cpu_count(logical=True)
    out["cpu_count_physical"] = _psutil.cpu_count(logical=False)

    return jsonify(out)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, debug=a.debug)