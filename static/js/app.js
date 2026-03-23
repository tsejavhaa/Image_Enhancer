/* ─────────────────────────────────────────────
   SwinIR UI – app.js
───────────────────────────────────────────── */

"use strict";

/* ── State ────────────────────────────────── */
const state = {
  task: "classical_sr",
  variant: "x4",
  uploadedFiles: [],   // [{id, original_name, path, info}]
  tasksConfig: {},
};

/* ── DOM refs ─────────────────────────────── */
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ── Bootstrap ─────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupNavTabs();
  setupTaskButtons();
  setupDropZone();
  setupFileInput();
  setupTileToggle();
  setupRunButton();
  setupModelsPanel();
  setupModal();
  setupClearFiles();
  setupHistoryPanel();
  detectDevice();
  await loadTaskConfig();
  renderVariantPills();
  createToastContainer();
  if (window._startMonitor) window._startMonitor();
});

/* ─────────────────────────────────────────────
   NAV TABS
───────────────────────────────────────────── */
function setupNavTabs() {
  $$(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      $$(".nav-tab").forEach(t => t.classList.remove("active"));
      $$(".panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      $(`panel-${tab.dataset.panel}`).classList.add("active");

      if (tab.dataset.panel === "models")  refreshModelsTable();
      if (tab.dataset.panel === "history") refreshHistory();
    });
  });
}

/* ─────────────────────────────────────────────
   TASK / VARIANT
───────────────────────────────────────────── */
async function loadTaskConfig() {
  try {
    const res = await fetch("/api/tasks");
    state.tasksConfig = await res.json();
  } catch (e) {
    console.error("Failed to load task config", e);
  }
}

function setupTaskButtons() {
  $$(".task-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".task-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.task = btn.dataset.task;
      // Set default variant to first key
      const variants = Object.keys(state.tasksConfig[state.task] || {});
      state.variant = variants[0] || "";
      renderVariantPills();
    });
  });
}

function renderVariantPills() {
  const container = $("variantPills");
  container.innerHTML = "";
  const variants = state.tasksConfig[state.task] || {};

  Object.entries(variants).forEach(([key, cfg]) => {
    const pill = document.createElement("button");
    pill.className = "variant-pill" + (key === state.variant ? " active" : "");
    pill.textContent = variantLabel(state.task, key, cfg);
    pill.addEventListener("click", () => {
      $$(".variant-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.variant = key;
    });
    container.appendChild(pill);
  });
}

function variantLabel(task, key, cfg) {
  if (task.endsWith("_sr"))  return `×${cfg.scale}` + (key.includes("large") ? " Large" : "");
  if (task.endsWith("_dn"))  return `σ=${cfg.noise}`;
  if (task.includes("jpeg")) return `Q${cfg.jpeg}`;
  return key;
}

/* ─────────────────────────────────────────────
   FILE DROP / BROWSE
───────────────────────────────────────────── */
function setupDropZone() {
  const zone  = $("dropZone");
  const input = $("fileInput");

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    handleFiles(Array.from(e.dataTransfer.files));
  });

  // Only open the picker when clicking the zone background itself.
  // Clicks on the <label> or its children are excluded because the <label>
  // already triggers the hidden <input> natively — calling .click() here too
  // would open the file picker a second time after the user closes the first.
  zone.addEventListener("click", e => {
    const label = zone.querySelector("label");
    if (label && (e.target === label || label.contains(e.target))) return;
    if (e.target === input) return;
    input.click();
  });
}

function setupFileInput() {
  $("fileInput").addEventListener("change", e => {
    handleFiles(Array.from(e.target.files));
    e.target.value = "";
  });
}

async function handleFiles(files) {
  if (!files.length) return;

  // Check by file extension — MIME types are unreliable across OS/browser combos
  // (e.g. .tif has no MIME type on Windows, .bmp varies, .exr has none at all).
  const ALLOWED_EXT = new Set([
    "jpg","jpeg","jpe","jfif",
    "png",
    "bmp","dib",
    "tif","tiff",
    "webp",
    "pbm","pgm","ppm","pnm",
    "hdr","pic","exr",
    "jp2","j2k",
    "gif",
    "ico",
    "avif","heic","heif",
  ]);
  const getExt = name => name.split(".").pop().toLowerCase();
  const valid = files.filter(f => ALLOWED_EXT.has(getExt(f.name)));
  if (!valid.length) { toast("No valid image files selected.", "error"); return; }

  toast(`Uploading ${valid.length} file(s)…`);

  const form = new FormData();
  valid.forEach(f => form.append("files", f));

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    state.uploadedFiles.push(...data.files);
    renderFileList();
    toast(`${data.files.length} file(s) ready.`, "success");
  } catch (e) {
    toast("Upload failed: " + e.message, "error");
  }
}

function setupClearFiles() {
  $("btnClearFiles").addEventListener("click", () => {
    state.uploadedFiles = [];
    renderFileList();
    $("btnRun").disabled = true;
  });
}

function renderFileList() {
  const list = $("fileList");
  const items = $("fileItems");
  const count = state.uploadedFiles.length;

  if (count === 0) { list.style.display = "none"; $("btnRun").disabled = true; return; }

  list.style.display = "block";
  $("btnRun").disabled = false;
  $("fileCount").textContent = `${count} file${count > 1 ? "s" : ""}`;

  items.innerHTML = "";
  state.uploadedFiles.forEach((f, idx) => {
    const div = document.createElement("div");
    div.className = "file-item";
    div.innerHTML = `
      <img class="file-thumb" src="${f.path}" alt="" />
      <div class="file-meta">
        <div class="file-name">${f.original_name}</div>
        <div class="file-info">${f.info.width}×${f.info.height}px · ${f.info.size_kb} KB</div>
      </div>
      <button class="file-remove" data-idx="${idx}" title="Remove">✕</button>
    `;
    items.appendChild(div);
  });

  items.querySelectorAll(".file-remove").forEach(btn => {
    btn.addEventListener("click", () => {
      state.uploadedFiles.splice(parseInt(btn.dataset.idx), 1);
      renderFileList();
    });
  });
}

/* ─────────────────────────────────────────────
   TILE TOGGLE
───────────────────────────────────────────── */
function setupTileToggle() {
  const toggle = $("useTile");
  const opts   = $("tileOptions");
  const slider = $("tileSize");
  const val    = $("tileSizeVal");

  toggle.addEventListener("change", () => {
    opts.style.display = toggle.checked ? "flex" : "none";
  });
  // Snap displayed value to nearest multiple of 8 (minimum window_size across all models)
  slider.addEventListener("input", () => {
    const snapped = Math.max(8, Math.floor(parseInt(slider.value) / 8) * 8);
    val.textContent = snapped;
  });
}

/* ─────────────────────────────────────────────
   ENHANCE / RUN
───────────────────────────────────────────── */
function setupRunButton() {
  $("btnRun").addEventListener("click", startEnhancement);
}

async function startEnhancement() {
  if (!state.uploadedFiles.length) return;

  const payload = {
    file_ids:  state.uploadedFiles.map(f => f.id),
    file_meta: state.uploadedFiles.map(f => ({ file_id: f.id, original_name: f.original_name })),
    task:      state.task,
    variant:   state.variant,
    use_tile:  $("useTile").checked,
    tile_size: parseInt($("tileSize").value),
    device:    _selectedDevice,
  };

  $("btnRun").disabled = true;
  showProgress(true, "Submitting job…", 0);
  $("resultsSection").style.display = "none";
  $("resultsGrid").innerHTML = "";

  try {
    const res = await fetch("/api/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const { job_id, error } = await res.json();
    if (error) throw new Error(error);

    pollJob(job_id, payload.file_ids.length);
  } catch (e) {
    toast("Enhancement failed: " + e.message, "error");
    $("btnRun").disabled = false;
    showProgress(false);
  }
}

function pollJob(jobId, total) {
  const es = new EventSource(`/api/job/${jobId}/stream`);

  es.onmessage = e => {
    // SSE keepalive comments arrive as empty data — ignore them
    if (!e.data || e.data.trim() === "") return;

    let job;
    try { job = JSON.parse(e.data); } catch { return; }

    const pct = total > 0 ? (job.progress / total) * 100 : 0;

    const labels = {
      queued:        "Queued…",
      loading_model: "Loading model…",
      processing:    `Processing ${job.progress}/${total}…`,
      done:          "Done!",
      error:         "Error",
    };

    const slowWarn = (job.elapsed_sec > 30 && job.status === "processing")
      ? "  ⚠ Slow on CPU — see Activity Monitor below" : "";

    showProgress(true, (labels[job.status] || job.status) + slowWarn, pct);

    if (job.status === "done" || job.status === "error") {
      es.close();
      $("btnRun").disabled = false;
      showProgress(false);

      if (job.results && job.results.length) {
        renderResults(job.results);
        toast(`Enhancement complete! ${job.results.length} image(s) processed.`, "success");
      }
      if (job.errors && job.errors.length) {
        job.errors.forEach(err => toast(err, "error"));
      }
      // Refresh history in background so it's ready when user switches
      refreshHistory();
    }
  };

  es.onerror = () => {
    es.close();
    // Fall back to polling
    setTimeout(() => fetchJobStatus(jobId, total), 1000);
  };
}

async function fetchJobStatus(jobId, total) {
  try {
    const res = await fetch(`/api/job/${jobId}`);
    const job = await res.json();
    const pct = total > 0 ? (job.progress / total) * 100 : 0;
    const slowNote = (job.elapsed_sec > 30 && job.status === "processing")
      ? "  ⚠ Slow on CPU — see Activity Monitor below" : "";
    showProgress(true, `Processing ${job.progress}/${total}…${slowNote}`, pct);

    if (job.status !== "done" && job.status !== "error") {
      setTimeout(() => fetchJobStatus(jobId, total), 800);
    } else {
      $("btnRun").disabled = false;
      showProgress(false);
      if (job.results.length) renderResults(job.results);
    }
  } catch (e) {
    setTimeout(() => fetchJobStatus(jobId, total), 2000);
  }
}

function showProgress(show, label = "", pct = 0) {
  // Detail and elapsed are shown exclusively in the Activity Monitor —
  // no duplication here.
  const block = $("progressBlock");
  block.style.display = show ? "flex" : "none";
  if (show) {
    $("progressLabel").textContent = label;
    $("progressBar").style.width   = pct + "%";
  }
}

/* ─────────────────────────────────────────────
   BEFORE / AFTER SLIDER WIDGET
───────────────────────────────────────────── */

/**
 * createSlider(beforeSrc, afterSrc, initialPct = 50)
 * Returns a <div class="ba-slider"> element with full drag support.
 * Works with mouse, touch, and keyboard (← →).
 */
function createSlider(beforeSrc, afterSrc, initialPct = 50) {
  const wrap = document.createElement("div");
  wrap.className = "ba-slider";

  wrap.innerHTML = `
    <img class="ba-after"  src="${afterSrc}"  alt="After"  draggable="false"/>
    <img class="ba-before" src="${beforeSrc}" alt="Before" draggable="false"/>
    <div class="ba-divider">
      <div class="ba-handle">
        <svg viewBox="0 0 24 24" fill="none" stroke="#555" stroke-width="2.2"
             stroke-linecap="round" stroke-linejoin="round">
          <polyline points="15 18 9 12 15 6"/>
          <polyline points="9 6 15 12 9 18" transform="translate(6,0)"/>
        </svg>
      </div>
    </div>
    <span class="ba-label ba-label-before">BEFORE</span>
    <span class="ba-label ba-label-after">AFTER</span>
  `;

  const before   = wrap.querySelector(".ba-before");
  const divider  = wrap.querySelector(".ba-divider");
  let pct = initialPct;

  function setPos(x) {
    const rect = wrap.getBoundingClientRect();
    pct = Math.min(100, Math.max(0, ((x - rect.left) / rect.width) * 100));
    divider.style.left         = pct + "%";
    before.style.clipPath      = `inset(0 ${100 - pct}% 0 0)`;
  }

  // Initialise
  divider.style.left    = pct + "%";
  before.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;

  // Mouse
  let dragging = false;
  wrap.addEventListener("mousedown",  e => { dragging = true; setPos(e.clientX); e.preventDefault(); });
  window.addEventListener("mousemove", e => { if (dragging) setPos(e.clientX); });
  window.addEventListener("mouseup",   ()  => { dragging = false; });

  // Touch
  wrap.addEventListener("touchstart", e => { dragging = true; setPos(e.touches[0].clientX); }, { passive: true });
  wrap.addEventListener("touchmove",  e => { if (dragging) setPos(e.touches[0].clientX); },    { passive: true });
  wrap.addEventListener("touchend",   ()  => { dragging = false; });

  // Keyboard accessibility
  wrap.setAttribute("tabindex", "0");
  wrap.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft")  { pct = Math.max(0,   pct - 2); divider.style.left = pct + "%"; before.style.clipPath = `inset(0 ${100-pct}% 0 0)`; }
    if (e.key === "ArrowRight") { pct = Math.min(100, pct + 2); divider.style.left = pct + "%"; before.style.clipPath = `inset(0 ${100-pct}% 0 0)`; }
  });

  return wrap;
}

/* ─────────────────────────────────────────────
   RESULTS
───────────────────────────────────────────── */
function fmtProcTime(sec) {
  if (!sec && sec !== 0) return "";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(0).padStart(2, "0");
  return `${m}m ${s}s`;
}

function renderResults(results) {
  const section = $("resultsSection");
  const grid    = $("resultsGrid");
  section.style.display = "block";

  results.forEach(r => {
    const origFile = state.uploadedFiles.find(f => f.id === r.file_id);
    const name = r.orig_name || (origFile ? origFile.original_name : r.file_id);

    const card = document.createElement("div");
    card.className = "result-card";

    const metrics = r.metrics || {};
    const metricHtml = Object.entries(metrics)
      .filter(([,v]) => v && isFinite(v))
      .map(([k, v]) => `<span class="metric-chip">${k.toUpperCase()}: ${typeof v === "number" ? v.toFixed(2) : v}</span>`)
      .join("");

    const outInfo = r.output_info || {};

    const procLabel = r.proc_sec != null
      ? `<span class="proc-time-chip">⏱ ${fmtProcTime(r.proc_sec)}</span>` : "";

    // Static output image on card — slider opens in modal on click
    card.innerHTML = `
      <img class="result-img" src="${r.output_url}" alt="${name}" />
      <div class="result-body">
        <div class="result-title">${name}</div>
        <div class="result-info-row">
          ${outInfo.width ? `<span class="file-info">${outInfo.width}×${outInfo.height}px · ${outInfo.size_kb} KB</span>` : ""}
          ${procLabel}
        </div>
        <div class="result-metrics">${metricHtml}</div>
      </div>
      <div class="result-actions">
        <a class="btn-dl" href="${r.output_url}" download>⬇ Download</a>
        <a class="btn-dl" href="${r.comparison_url}" download>⬇ Compare</a>
      </div>
    `;

    // Clicking the card (not the download buttons) opens the slider modal
    card.addEventListener("click", e => {
      if (e.target.closest(".btn-dl")) return;
      openModal(r, name, metrics);
    });

    grid.appendChild(card);
  });

  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ─────────────────────────────────────────────
   MODAL
───────────────────────────────────────────── */
function setupModal() {
  $("modalClose").addEventListener("click", closeModal);
  $("modalBackdrop").addEventListener("click", closeModal);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });
}

function openModal(result, name, metrics) {
  const content = $("modalContent");
  const mHtml = Object.entries(metrics)
    .filter(([,v]) => v && isFinite(v))
    .map(([k,v]) => `<span class="modal-metric">${k.toUpperCase()}: ${typeof v === "number" ? v.toFixed(2) : v}</span>`)
    .join("");

  // Prefer original upload for "before"; fall back to comparison PNG
  const beforeSrc = result.original_url || result.comparison_url;
  const afterSrc  = result.output_url;

  // Build modal shell first, inject slider after
  const procChip = result.proc_sec != null
    ? `<span class="proc-time-chip" style="margin-left:8px;vertical-align:middle">⏱ ${fmtProcTime(result.proc_sec)}</span>` : "";

  content.innerHTML = `
    <h3 style="margin-bottom:10px;font-size:1rem;padding-right:28px">${name}${procChip}</h3>
    <div class="modal-slider" id="modalSliderWrap">
      <div style="padding:40px;text-align:center;color:#888;font-size:.8rem">Loading images…</div>
    </div>
    <p style="color:var(--text-3);font-size:.75rem;margin-top:8px;text-align:center">
      ◀ &nbsp;Drag divider to compare&nbsp; · &nbsp;← → keyboard also works
    </p>
    ${mHtml ? `<div class="modal-meta" style="margin-top:10px">${mHtml}</div>` : ""}
    <div style="display:flex;gap:8px;margin-top:14px">
      <a class="btn-dl" href="${result.output_url}" download>⬇ Download Enhanced</a>
      <a class="btn-dl" href="${result.comparison_url}" download>⬇ Comparison</a>
    </div>
  `;

  // Show modal immediately so user sees something
  $("modal").style.display = "flex";

  // Pre-load both images before inserting slider so dimensions are known
  const imgA = new Image();
  const imgB = new Image();
  let loaded = 0;

  function onLoad() {
    loaded++;
    if (loaded < 2) return;
    const wrap = $("modalSliderWrap");
    if (!wrap) return;
    wrap.innerHTML = "";
    const sliderEl = createSlider(beforeSrc, afterSrc, 50);
    wrap.appendChild(sliderEl);
    setTimeout(() => sliderEl.focus(), 80);
  }

  imgA.onload  = onLoad;
  imgB.onload  = onLoad;
  // If one image fails (e.g. original_url deleted), still show the slider
  imgA.onerror = onLoad;
  imgB.onerror = onLoad;
  imgA.src = beforeSrc;
  imgB.src = afterSrc;
}

function closeModal() { $("modal").style.display = "none"; }

/* ─────────────────────────────────────────────
   MODELS PANEL
───────────────────────────────────────────── */
function setupModelsPanel() {
  $("btnClearCache").addEventListener("click", async () => {
    await fetch("/api/models/clear_cache", { method: "POST" });
    toast("Memory cache cleared.", "success");
  });
}

async function refreshModelsTable() {
  const container = $("modelTable");
  container.innerHTML = `<div class="table-loading"><div class="spinner"></div></div>`;

  try {
    const res = await fetch("/api/models");
    const models = await res.json();

    const header = `
      <div class="model-row header">
        <span>Task</span>
        <span>Variant</span>
        <span>Status</span>
        <span>Size</span>
        <span class="col-fn">Filename</span>
      </div>`;

    const rows = models.map(m => `
      <div class="model-row">
        <span><strong>${m.task}</strong></span>
        <span>${m.variant}</span>
        <span><span class="badge-dl ${m.downloaded ? "yes" : "no"}">${m.downloaded ? "✓ Cached" : "Not downloaded"}</span></span>
        <span>${m.downloaded ? m.size_mb + " MB" : "—"}</span>
        <span class="col-fn" style="font-size:.75rem;color:var(--text-3)">${m.filename}</span>
      </div>
    `).join("");

    container.innerHTML = header + rows;
  } catch (e) {
    container.innerHTML = `<div class="table-loading" style="color:var(--danger)">Failed to load models.</div>`;
  }
}

/* ─────────────────────────────────────────────
   DEVICE SELECTOR
───────────────────────────────────────────── */
let _selectedDevice = "auto";  // set after detectDevice() resolves

async function detectDevice() {
  try {
    const res  = await fetch("/api/devices");
    const data = await res.json();

    const group = $("deviceBtnGroup");
    if (!group) return;

    const buttons = group.querySelectorAll(".device-btn");

    buttons.forEach(btn => {
      const dev = data.devices.find(d => d.id === btn.dataset.device);
      if (!dev) return;

      if (dev.available) {
        btn.disabled = false;
        btn.title = dev.info;
      } else {
        btn.disabled = true;
        btn.title = dev.info;
      }

      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _selectedDevice = btn.dataset.device;
        toast(`Compute device set to ${btn.textContent}`, "success");
      });
    });

    // Auto-select recommended device
    _selectedDevice = data.recommended;
    const recBtn = group.querySelector(`[data-device="${data.recommended}"]`);
    if (recBtn && !recBtn.disabled) {
      recBtn.classList.add("active");
    }

  } catch (e) {
    console.warn("Device detection failed:", e);
    // Fall back: enable CPU only
    const cpuBtn = document.querySelector(".device-btn[data-device='cpu']");
    if (cpuBtn) { cpuBtn.disabled = false; cpuBtn.classList.add("active"); }
    _selectedDevice = "cpu";
  }
}


/* ─────────────────────────────────────────────
   HISTORY PANEL
───────────────────────────────────────────── */
function setupHistoryPanel() {
  $("btnRefreshHistory").addEventListener("click", refreshHistory);
  $("btnClearHistory").addEventListener("click", async () => {
    if (!confirm("Clear all history entries? Output files on disk are kept.")) return;
    await fetch("/api/history/clear", { method: "DELETE" });
    toast("History cleared.", "success");
    refreshHistory();
  });
}

async function refreshHistory() {
  const list = $("historyList");
  list.innerHTML = '<div class="history-empty"><div class="spinner" style="margin:0 auto"></div></div>';
  try {
    const res  = await fetch("/api/history");
    const jobs = await res.json();
    renderHistory(jobs);
  } catch (e) {
    list.innerHTML = `<div class="history-empty" style="color:var(--danger)">Failed to load history.</div>`;
  }
}

function renderHistory(jobs) {
  const list = $("historyList");
  if (!jobs.length) {
    list.innerHTML = '<div class="history-empty">No history yet. Enhanced images will appear here.</div>';
    return;
  }
  list.innerHTML = "";
  jobs.forEach(job => list.appendChild(buildJobCard(job)));
}

function buildJobCard(job) {
  const results  = job.results  || [];
  const errors   = job.errors   || [];
  const hasFiles = results.length > 0;

  // Badge
  let badgeClass = "done", badgeText = `${results.length} result${results.length !== 1 ? "s" : ""}`;
  if (results.length === 0 && errors.length > 0) { badgeClass = "error"; badgeText = "Failed"; }
  else if (errors.length > 0) { badgeClass = "partial"; badgeText = `${results.length} ok, ${errors.length} err`; }

  // Header meta
  const date    = job.completed_at ? new Date(job.completed_at * 1000).toLocaleString() : "";
  const elapsed = job.elapsed_sec  ? ` · ${job.elapsed_sec}s` : "";
  const taskLabel = `${job.task}  /  ${job.variant}`;

  const card = document.createElement("div");
  card.className = "history-job";
  card.dataset.jobId = job.job_id;

  card.innerHTML = `
    <div class="history-job-header">
      <span class="history-job-chevron">▶</span>
      <span class="history-job-title">${taskLabel}</span>
      <span class="history-job-badge ${badgeClass}">${badgeText}</span>
      <span class="history-job-meta">${date}${elapsed}</span>
      <button class="btn-icon-danger btn-delete-job" title="Remove job from history" data-job-id="${job.job_id}">🗑</button>
    </div>
    <div class="history-job-body">
      ${results.map(r => buildResultRow(job.job_id, r)).join("")}
      ${errors.map(e => `
        <div class="history-error-row">
          <span>⚠</span><span>${e}</span>
        </div>`).join("")}
    </div>
  `;

  // Toggle open/close
  card.querySelector(".history-job-header").addEventListener("click", e => {
    if (e.target.closest(".btn-delete-job")) return;
    card.classList.toggle("open");
  });

  // Delete whole job
  card.querySelector(".btn-delete-job").addEventListener("click", async e => {
    e.stopPropagation();
    if (!confirm("Remove this job from history? Output files on disk are kept.")) return;
    await fetch(`/api/history/${job.job_id}`, { method: "DELETE" });
    card.remove();
    if (!$("historyList").querySelector(".history-job")) {
      $("historyList").innerHTML = '<div class="history-empty">No history yet.</div>';
    }
    toast("Job removed from history.", "success");
  });

  // Auto-open if only 1 job
  if (document.querySelectorAll(".history-job").length === 0 && results.length > 0) {
    card.classList.add("open");
  }

  return card;
}

function buildResultRow(jobId, r) {
  const name    = r.orig_name || r.file_id;
  const info    = r.output_info || {};
  const dimText  = info.width ? `${info.width}×${info.height}px · ${info.size_kb} KB` : "";
  const procText = r.proc_sec != null ? `⏱ ${fmtProcTime(r.proc_sec)}` : "";
  const metrics = r.metrics   || {};
  const chips   = Object.entries(metrics)
    .filter(([,v]) => v && isFinite(v))
    .map(([k,v]) => `<span class="history-chip">${k.toUpperCase()}: ${Number(v).toFixed(2)}</span>`)
    .join("");
  const outFile = r.output_url ? r.output_url.split("/").pop() : "";

  const rEnc   = JSON.stringify(r).replace(/"/g,"&quot;");
  const mEnc   = JSON.stringify(metrics).replace(/"/g,"&quot;");

  return `
    <div class="history-result-row" data-output-file="${outFile}"
         data-name="${name.replace(/"/g,"&quot;")}"
         data-result="${rEnc}"
         data-metrics="${mEnc}">
      <img class="history-thumb" src="${r.output_url}"
           alt="${name}" title="Click to open slider preview"
           onerror="this.style.background='var(--surface-2)'"/>
      <div class="history-result-info">
        <div class="history-result-name" title="${name}">${name}</div>
        <div class="history-result-dim">${dimText}${dimText && procText ? "  ·  " : ""}${procText}</div>
        <div class="history-result-chips">${chips}</div>
      </div>
      <div class="history-result-actions">
        <a class="btn-dl-sm" href="${r.output_url}"     download title="Download enhanced image">⬇ Enhanced</a>
        <a class="btn-dl-sm" href="${r.comparison_url}" download title="Download comparison">⬇ Compare</a>
      </div>
      <button class="btn-icon-danger btn-delete-result"
              title="Delete this result file from disk"
              data-job-id="${jobId}"
              data-filename="${outFile}">🗑</button>
    </div>`;
}

// History row click → open modal with slider
document.addEventListener("click", e => {
  if (e.target.closest(".btn-delete-result") || e.target.closest(".btn-dl-sm")) return;
  const row = e.target.closest(".history-result-row");
  if (!row) return;
  try {
    const r       = JSON.parse(row.dataset.result.replace(/&quot;/g, '"'));
    const metrics = JSON.parse(row.dataset.metrics.replace(/&quot;/g, '"'));
    const name    = row.dataset.name || "";
    openModal(r, name, metrics);
  } catch { /* ignore parse errors */ }
});

// Delegate result-row delete (works on dynamically created rows)
document.addEventListener("click", async e => {
  const btn = e.target.closest(".btn-delete-result");
  if (!btn) return;
  e.stopPropagation();
  const { jobId, filename } = btn.dataset;
  if (!confirm(`Delete "${filename}" from disk? This cannot be undone.`)) return;
  const res  = await fetch(`/api/history/${jobId}/results/${filename}`, { method: "DELETE" });
  const data = await res.json();
  if (data.ok) {
    const row = btn.closest(".history-result-row");
    const body = row.closest(".history-job-body");
    row.remove();
    // If the job body is now empty, remove the whole card
    if (!body.querySelector(".history-result-row, .history-error-row")) {
      body.closest(".history-job").remove();
    }
    toast(`Deleted ${data.deleted_files.length} file(s).`, "success");
  } else {
    toast("Delete failed.", "error");
  }
});


/* ─────────────────────────────────────────────
   ACTIVITY MONITOR
───────────────────────────────────────────── */
(function () {
  let _lastElapsed = 0;
  let _monitorLoggedOnce = false;
  let _monitorInterval = null;

  function fmtElapsed(sec) {
    if (!sec) return "";
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return m > 0 ? `${m}m ${s}s elapsed` : `${s}s elapsed`;
  }

  function setBar(barEl, pct) {
    barEl.style.width = pct + "%";
    barEl.classList.toggle("warn", pct >= 70 && pct < 90);
    barEl.classList.toggle("crit", pct >= 90);
  }

  function updateMonitor(data) {
    // Log first sample to console; visit /api/system/debug for full diagnostics
    if (!_monitorLoggedOnce) {
      console.log("[Activity Monitor] first sample:", JSON.stringify(data, null, 2));
      console.log("[Activity Monitor] raw diagnostics: /api/system/debug");
      _monitorLoggedOnce = true;
    }
    const dot = $("monitorDot");

    // ── CPU ──────────────────────────────────────────────────
    const cpu = Math.round(data.cpu_pct ?? 0);
    setBar($("cpuBar"), cpu);
    $("cpuVal").textContent = cpu + "%";

    // ── Per-core sparks ──────────────────────────────────────
    const coresEl = $("monitorCores");
    const cores   = data.cpu_per_core || [];
    if (cores.length) {
      // Rebuild only when core count changes
      if (coresEl.children.length !== cores.length) {
        coresEl.innerHTML = cores.map(() => `<div class="core-bar"></div>`).join("");
      }
      cores.forEach((pct, i) => {
        const bar = coresEl.children[i];
        bar.style.height = Math.max(3, Math.round(pct * 0.2)) + "px"; // max 20px
        bar.classList.toggle("warn", pct >= 70 && pct < 90);
        bar.classList.toggle("crit", pct >= 90);
        bar.title = `Core ${i + 1}: ${Math.round(pct)}%`;
      });
    }

    // ── RAM ──────────────────────────────────────────────────
    const ram = Math.round(data.ram_pct ?? 0);
    setBar($("ramBar"), ram);
    // Show "X.XX / YG  (ZZ%)" so RAM is always informative even when CPU=0
    if (data.ram_total_gb > 0) {
      $("ramVal").textContent = `${data.ram_used_gb}G`;
      $("ramBar").title = `${data.ram_used_gb} / ${data.ram_total_gb} GiB  (${ram}%)`;
    } else {
      $("ramVal").textContent = "–";
    }

    // ── GPU (optional) ────────────────────────────────────────
    if (data.gpu_pct != null) {
      $("gpuRow").style.display = "grid";
      setBar($("gpuBar"), Math.round(data.gpu_pct));
      $("gpuVal").textContent = data.gpu_mem_used != null
        ? `${data.gpu_mem_used}G`
        : Math.round(data.gpu_pct) + "%";
    }

    // ── Active job ticker ────────────────────────────────────
    const jobEl     = $("monitorJob");
    const labelEl   = $("monitorJobLabel");
    const elapsedEl = $("monitorJobElapsed");

    if (data.active_jobs > 0) {
      jobEl.style.display = "block";
      labelEl.textContent   = data.active_detail || "Processing…";
      elapsedEl.textContent = fmtElapsed(data.active_elapsed);

      // Dot state
      dot.className = cpu >= 80 ? "monitor-dot busy" : "monitor-dot working";
    } else {
      jobEl.style.display = "none";
      dot.className = "monitor-dot idle";
    }
  }

  async function pollSystem() {
    try {
      const res  = await fetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      updateMonitor(data);
    } catch (err) {
      console.warn("[Activity Monitor] poll error:", err);
    }
  }

  // 2 × 0.3 s server blocking = ~0.6 s; poll every 3 s to leave headroom
  function startMonitor() {
    if (_monitorInterval) return;
    // Small initial delay so the server's first cpu_percent() baseline is primed
    setTimeout(pollSystem, 500);
    _monitorInterval = setInterval(pollSystem, 3000);
  }

  // Expose so DOMContentLoaded can start it
  window._startMonitor = startMonitor;
})();

/* ─────────────────────────────────────────────
   TOAST
───────────────────────────────────────────── */
function createToastContainer() {
  const div = document.createElement("div");
  div.id = "toastContainer";
  document.body.appendChild(div);
}

function toast(msg, type = "") {
  const container = $("toastContainer");
  const el = document.createElement("div");
  el.className = "toast" + (type ? ` ${type}` : "");
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}