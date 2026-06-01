import importlib.util
import json
import logging
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path

from flask import (
    Flask, Response, jsonify, render_template_string,
    request, send_file, stream_with_context,
)
from werkzeug.middleware.proxy_fix import ProxyFix

from services.rigging.rigging_inference import rig_3d_model
from services.kimodo_motion.motion_inference import generate_motion
from services.kimodo_motion.retarget import blend_animation
from services.img_gen.img_inference import generate_character_image
from services.json_extraction.extraction_inference import extract_story_graphs

# services/2d_to_3d has a digit-prefixed name — load via importlib
_glb_spec = importlib.util.spec_from_file_location(
    "generate_glb",
    Path(__file__).parent / "services" / "2d_to_3d" / "generate_glb.py",
)
_glb_mod = importlib.util.module_from_spec(_glb_spec)
_glb_spec.loader.exec_module(_glb_mod)
generate_glb = _glb_mod.generate_glb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["OPTIONS"])
@app.route("/run", methods=["OPTIONS"])
@app.route("/cache/<path:p>", methods=["OPTIONS"])
def options_preflight(**_):
    return Response(status=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })

PIPELINE_OUTPUT_DIR = Path(__file__).parent / "pipeline_outputs"
PIPELINE_OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_IMAGES_DIR = PIPELINE_OUTPUT_DIR / "images"
CACHE_GLB_DIR = PIPELINE_OUTPUT_DIR / "glb"
CACHE_ANIMATED_DIR = PIPELINE_OUTPUT_DIR / "animated_glb"
for _d in [CACHE_IMAGES_DIR, CACHE_GLB_DIR, CACHE_ANIMATED_DIR]:
    _d.mkdir(exist_ok=True)

# session_id -> {"dir": str, "results": {char_id: animated_glb_path}}
SESSIONS: dict = {}

# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AR Storytelling Pipeline</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0d1117; color: #e2e8f0;
      margin: 0; padding: 32px 24px; min-height: 100vh;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { font-size: 1.75rem; margin: 0 0 4px; color: #f1f5f9; font-weight: 700; }
    .subtitle { color: #64748b; margin: 0 0 32px; font-size: .95rem; }

    .card {
      background: #161b22; border: 1px solid #21262d;
      border-radius: 12px; padding: 24px; margin-bottom: 20px;
    }
    .card-title {
      font-size: .75rem; font-weight: 600; letter-spacing: .08em;
      text-transform: uppercase; color: #64748b; margin: 0 0 18px;
    }

    /* Upload */
    .upload-layout { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
    .preview-box {
      flex: 0 0 180px; height: 180px;
      background: #0d1117; border: 1px dashed #30363d;
      border-radius: 8px; display: flex; align-items: center;
      justify-content: center; overflow: hidden; flex-shrink: 0;
    }
    .preview-box img { width: 100%; height: 100%; object-fit: contain; display: none; }
    .preview-hint { color: #484f58; font-size: .82rem; text-align: center; padding: 12px; }
    .upload-right { flex: 1; min-width: 220px; }
    .dropzone {
      border: 2px dashed #30363d; border-radius: 8px;
      padding: 26px 20px; text-align: center;
      cursor: pointer; transition: border-color .2s, background .2s;
      margin-bottom: 12px;
    }
    .dropzone:hover, .dropzone.dragover { border-color: #6366f1; background: #13172a; }
    .dropzone p { margin: 6px 0; color: #8b949e; font-size: .9rem; }
    .dropzone .sel-name { color: #818cf8; font-size: .85rem; font-weight: 500; }
    .btn-run {
      width: 100%; padding: 12px;
      background: #4f46e5; color: #fff;
      border: none; border-radius: 8px;
      font-size: .95rem; font-weight: 600; cursor: pointer;
      transition: background .15s;
    }
    .btn-run:hover:not(:disabled) { background: #4338ca; }
    .btn-run:disabled { opacity: .4; cursor: default; }

    /* Stage rows */
    .stage-row {
      display: flex; align-items: center; gap: 12px;
      padding: 11px 0; border-bottom: 1px solid #21262d; font-size: .88rem;
    }
    .stage-row:last-child { border-bottom: none; }
    .dot {
      flex-shrink: 0; width: 10px; height: 10px;
      border-radius: 50%; background: #30363d;
    }
    .dot.running { background: #6366f1; animation: pulse .9s ease-in-out infinite; }
    .dot.done    { background: #22c55e; }
    .dot.error   { background: #ef4444; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
    .stage-info { flex: 1; }
    .stage-label { color: #c9d1d9; font-weight: 500; }
    .stage-sub   { color: #484f58; font-size: .78rem; margin-top: 2px; }
    .stage-time  { font-size: .78rem; color: #484f58; white-space: nowrap; }

    /* Character cards */
    .char-cards { margin-top: 16px; display: flex; flex-direction: column; gap: 10px; }
    .char-card {
      background: #0d1117; border: 1px solid #21262d;
      border-radius: 8px; padding: 14px 16px;
    }
    .char-header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; }
    .char-name { font-weight: 600; font-size: .95rem; color: #f1f5f9; }
    .char-id-tag { font-size: .75rem; color: #484f58; font-family: monospace; }
    .char-stages { display: flex; gap: 8px; flex-wrap: wrap; }
    .cs {
      display: flex; align-items: center; gap: 6px;
      font-size: .8rem; color: #8b949e;
      background: #161b22; border: 1px solid #21262d;
      border-radius: 6px; padding: 5px 10px;
      transition: border-color .2s, color .2s;
    }
    .cs.active  { border-color: #6366f1; color: #a5b4fc; }
    .cs.done-s  { border-color: #22c55e; color: #86efac; }
    .cs.error-s { border-color: #ef4444; color: #fca5a5; }
    .cs-time    { font-size: .72rem; color: #484f58; }

    /* Error banner */
    .err-banner {
      background: #1a0a0a; border: 1px solid #7f1d1d;
      border-radius: 8px; padding: 12px 16px;
      color: #fca5a5; font-size: .88rem; margin-bottom: 16px;
      display: none;
    }

    /* Results */
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 20px;
    }
    .model-card { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }
    .model-card-hdr {
      padding: 12px 16px; border-bottom: 1px solid #21262d;
      display: flex; align-items: center; justify-content: space-between;
    }
    .model-card-hdr span { font-weight: 600; font-size: .92rem; }
    model-viewer { width: 100%; height: 360px; background: #0d1117; display: block; }
    .model-card-ftr { padding: 10px 16px; }
    a.btn-dl {
      display: inline-block; padding: 7px 16px;
      background: #16a34a; color: #fff;
      border-radius: 6px; text-decoration: none; font-size: .83rem; font-weight: 500;
    }
    a.btn-dl:hover { background: #15803d; }
  </style>
</head>
<body>
<div class="container">
  <h1>AR Storytelling Pipeline</h1>
  <p class="subtitle">Upload a Hindu scripture page image — characters are automatically extracted, modeled, rigged, and animated.</p>

  <div id="err-banner" class="err-banner"></div>

  <!-- Upload -->
  <div class="card">
    <div class="card-title">Input</div>
    <div class="upload-layout">
      <div class="preview-box">
        <img id="preview-img" alt="Preview">
        <div class="preview-hint" id="preview-hint">Image<br>preview</div>
      </div>
      <div class="upload-right">
        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
          <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" hidden>
          <p>Drag &amp; drop a scripture image or <strong style="color:#818cf8">browse</strong></p>
          <p id="sel-name" class="sel-name"></p>
        </div>
        <button class="btn-run" id="runBtn" onclick="runPipeline()" disabled>Run Full Pipeline</button>
      </div>
    </div>
  </div>

  <!-- Progress -->
  <div class="card" id="progress-card" style="display:none">
    <div class="card-title">Pipeline Progress</div>
    <div class="stage-row">
      <div class="dot" id="dot-extr"></div>
      <div class="stage-info">
        <div class="stage-label">JSON Extraction</div>
        <div class="stage-sub" id="sub-extr">Pending</div>
      </div>
      <div class="stage-time" id="time-extr"></div>
    </div>
    <div class="char-cards" id="char-cards"></div>
  </div>

  <!-- Results -->
  <div class="card" id="results-card" style="display:none">
    <div class="card-title">Results</div>
    <div class="results-grid" id="results-grid"></div>
  </div>
</div>

<script>
  const STAGE_LABELS = { img_gen: 'Image Gen', '3d': '3D Model', rig: 'Rigging', anim: 'Animation' };
  const timers = {};
  let selectedFile = null;

  // File selection
  document.getElementById('fileInput').addEventListener('change', function () {
    if (this.files[0]) setFile(this.files[0]);
  });
  const dz = document.getElementById('dropzone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragover');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  function setFile(file) {
    selectedFile = file;
    document.getElementById('sel-name').textContent = file.name;
    document.getElementById('runBtn').disabled = false;
    const reader = new FileReader();
    reader.onload = ev => {
      const img = document.getElementById('preview-img');
      img.src = ev.target.result;
      img.style.display = 'block';
      document.getElementById('preview-hint').style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  // Pipeline runner
  async function runPipeline() {
    if (!selectedFile) return;
    document.getElementById('runBtn').disabled = true;
    document.getElementById('err-banner').style.display = 'none';
    document.getElementById('progress-card').style.display = 'block';
    document.getElementById('results-card').style.display = 'none';
    document.getElementById('char-cards').innerHTML = '';
    document.getElementById('results-grid').innerHTML = '';
    setDot('extr', ''); setSub('extr', 'Starting...'); setTime('extr', '');

    const form = new FormData();
    form.append('image', selectedFile);

    try {
      const resp = await fetch('/run', { method: 'POST', body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        showError(err.error || 'Failed to start pipeline');
        document.getElementById('runBtn').disabled = false;
        return;
      }

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        const chunks = buf.split('\\n\\n');
        buf = chunks.pop();

        for (const chunk of chunks) {
          const line = chunk.trim();
          if (line.startsWith('data: ')) {
            try { handleEvent(JSON.parse(line.slice(6))); }
            catch (e) { console.warn('Bad SSE event:', line); }
          }
        }
      }
    } catch (e) {
      showError(e.message);
      document.getElementById('runBtn').disabled = false;
    }
  }

  // Event dispatch
  function handleEvent(ev) {
    console.log('[event]', ev);
    switch (ev.type) {
      case 'stage':  onStage(ev);      break;
      case 'char':   onChar(ev);       break;
      case 'result': onResult(ev);     break;
      case 'done':   onDone(ev);       break;
      case 'error':  onFatalError(ev); break;
    }
  }

  function onStage(ev) {
    if (ev.stage !== 'extraction') return;
    if (ev.status === 'start') {
      setDot('extr', 'running');
      setSub('extr', 'Running OCR and LLM extraction…');
      timers['extr'] = Date.now();
    } else if (ev.status === 'done') {
      setDot('extr', 'done');
      setTime('extr', elapsed('extr'));
      const chars = ev.characters || [];
      setSub('extr', 'Found ' + chars.length + ' character(s): ' + chars.map(c => c.name || c.id).join(', '));
      chars.forEach(createCharCard);
    } else if (ev.status === 'error') {
      setDot('extr', 'error');
      setSub('extr', 'Error: ' + (ev.message || 'unknown'));
    }
  }

  function onChar(ev) {
    const key = ev.char_id + '-' + ev.stage;
    if (ev.status === 'start') {
      setCS(ev.char_id, ev.stage, 'active');
      setCD(ev.char_id, ev.stage, 'running');
      timers[key] = Date.now();
    } else if (ev.status === 'done') {
      setCS(ev.char_id, ev.stage, 'done-s');
      setCD(ev.char_id, ev.stage, 'done');
      setCT(ev.char_id, ev.stage, elapsed(key));
    } else if (ev.status === 'error') {
      setCS(ev.char_id, ev.stage, 'error-s');
      setCD(ev.char_id, ev.stage, 'error');
    }
  }

  function onResult(ev) {
    document.getElementById('results-card').style.display = 'block';
    const card = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML =
      '<div class="model-card-hdr"><span>' + ev.char_id + '</span></div>' +
      '<model-viewer src="' + ev.url + '" autoplay animation-name="NPZ_Animation" ' +
      'camera-controls shadow-intensity="1" orientation="0 180deg 0" ' +
      'camera-orbit="0deg 75deg auto" alt="Animated ' + ev.char_id + '" ar>' +
      '</model-viewer>' +
      '<div class="model-card-ftr">' +
      '<a class="btn-dl" href="' + ev.url + '" download="' + ev.char_id + '_animated.glb">Download GLB</a>' +
      '</div>';
    document.getElementById('results-grid').appendChild(card);
  }

  function onDone() {
    document.getElementById('runBtn').disabled = false;
  }

  function onFatalError(ev) {
    showError(ev.message || 'Pipeline failed');
    document.getElementById('runBtn').disabled = false;
  }

  // DOM helpers
  function setDot(id, cls) {
    const el = document.getElementById('dot-' + id);
    if (el) el.className = 'dot' + (cls ? ' ' + cls : '');
  }
  function setSub(id, txt) {
    const el = document.getElementById('sub-' + id);
    if (el) el.textContent = txt;
  }
  function setTime(id, txt) {
    const el = document.getElementById('time-' + id);
    if (el) el.textContent = txt;
  }2026-05-29 16:36:49,500 [INFO] services.json_extraction.extraction_inference - [extraction]   "animations_generated": 4,
2026-05-29 16:36:49,500 [INFO] services.json_extraction.extraction_inference - [extraction]   "scenes_composed": 1
2026-05-29 16:36:49,500 [INFO] services.json_extraction.extraction_inference - [extraction] }
2026-05-29 16:36:49,501 [INFO] services.json_extraction.extraction_inference - Extraction complete | story=Vamana Charitam  chars=2  anims=4
2026-05-29 16:36:49,501 [INFO] __main__ - Extraction done: 2 character(s)
2026-05-29 16:36:49,502 [INFO] services.img_gen.img_inference - generate_character_image | prompt='Full body T-pose, front-facing, dark noble complexion, tall strong noble king, royal Asura crown - g'
2026-05-29 16:36:49,502 [INFO] services.img_gen.img_inference - Fetching image from Pollinations.ai
2026-05-29 16:36:50,605 [INFO] services.img_gen.img_inference - Image saved: /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/pipeline_outputs/3a3c5fff-34f5-49ee-a281-27374e7516a2/bali_chakravarti_img.png
2026-05-29 16:36:50,606 [INFO] services.img_gen.img_inference - generate_character_image | prompt="Full body T-pose, front-facing, fair elderly complexion, old thin sage, none - sage's matted hair, w"
2026-05-29 16:36:50,606 [INFO] services.img_gen.img_inference - Fetching image from Pollinations.ai
2026-05-29 16:36:50,608 [INFO] generate_glb - generate_glb | input=/home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/pipeline_outputs/3a3c5fff-34f5-49ee-a281-27374e7516a2/bali_chakravarti_img.png asset_name=bali_chakravarti
2026-05-29 16:36:50,608 [INFO] generate_glb - Job dir: /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/outputs/jobs/bali_chakravarti_1780052810
2026-05-29 16:36:50,609 [INFO] generate_glb - Running TripoSR: python /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/TripoSR_model/run.py /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/pipeline_outputs/3a3c5fff-34f5-49ee-a281-27374e7516a2/bali_chakravarti_img.png --output-dir /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/outputs/jobs/bali_chakravarti_1780052810
2026-05-29 16:36:50,609 [INFO] generate_glb - cwd: /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/TripoSR_model
2026-05-29 16:36:51,182 [INFO] generate_glb - Subprocess python: /home/ms_upendra/miniconda3/envs/triposr/bin/python
2026-05-29 16:36:51,540 [INFO] services.img_gen.img_inference - Image saved: /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/pipeline_outputs/3a3c5fff-34f5-49ee-a281-27374e7516a2/shukracharya_img.png
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:43,558 - INFO - Initializing model ...
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] /home/ms_upendra/miniconda3/envs/triposr/lib/python3.12/site-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr]   _torch_pytree._register_pytree_node(
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] /home/ms_upendra/miniconda3/envs/triposr/lib/python3.12/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr]   _torch_pytree._register_pytree_node(
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/TripoSR_model/tsr/system.py:69: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr]   ckpt = torch.load(weight_path, map_location="cpu")
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:51,168 - INFO - Initializing model finished in 7610.28ms.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:51,169 - INFO - Processing images ...
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:52,160 - INFO - Processing images finished in 991.52ms.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:52,160 - INFO - Running image 1/1 ...
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:52,160 - INFO - Running model ...
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:52,619 - INFO - Running model finished in 458.21ms.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:52,619 - INFO - Extracting mesh ...
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:54,003 - INFO - Extracting mesh finished in 1384.77ms.
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:54,004 - INFO - Exporting mesh ...
2026-05-29 16:37:55,474 [INFO] generate_glb - [subprocess stderr] 2026-05-29 16:37:54,144 - INFO - Exporting mesh finished in 140.33ms.
2026-05-29 16:37:55,476 [ERROR] __main__ - 3D gen failed for bali_chakravarti
Traceback (most recent call last):
  File "/home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/main.py", line 564, in pipeline
    glb_path = generate_glb(img_path_out, asset_name=cid)
  File "/home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/generate_glb.py", line 130, in generate_glb
    convert_obj_to_glb(objs[0], final_glb)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/generate_glb.py", line 47, in convert_obj_to_glb
    import trimesh
ModuleNotFoundError: No module named 'trimesh'
2026-05-29 16:37:55,478 [INFO] generate_glb - generate_glb | input=/home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/pipeline_outputs/3a3c5fff-34f5-49ee-a281-27374e7516a2/shukracharya_img.png asset_name=shukracharya
2026-05-29 16:37:55,479 [INFO] generate_glb - Job dir: /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/outputs/jobs/shukracharya_1780052875
2026-05-29 16:37:55,479 [INFO] generate_glb - Running TripoSR: python /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/TripoSR_model/run.py /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/pipeline_outputs/3a3c5fff-34f5-49ee-a281-27374e7516a2/shukracharya_img.png --output-dir /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/outputs/jobs/shukracharya_1780052875
2026-05-29 16:37:55,479 [INFO] generate_glb - cwd: /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/2d_to_3d/TripoSR_model
2026-05-29 16:37:56,051 [INFO] generate_glb - Subprocess python: /home/ms_upendra/miniconda3/envs/triposr/bin/python
  function elapsed(key) {
    return timers[key] ? ((Date.now() - timers[key]) / 1000).toFixed(1) + 's' : '';
  }
  function showError(msg) {
    const el = document.getElementById('err-banner');
    el.textContent = 'Error: ' + msg;
    el.style.display = 'block';
  }

  function createCharCard(char) {
    const card = document.createElement('div');
    card.className = 'char-card';
    card.id = 'cc-' + char.id;
    const stages = ['img_gen', '3d', 'rig', 'anim'];
    let stagesHtml = '';
    stages.forEach(function(s) {
      stagesHtml +=
        '<div class="cs" id="cs-' + char.id + '-' + s + '">' +
        '<div class="dot" id="cd-' + char.id + '-' + s + '"></div>' +
        '<span>' + STAGE_LABELS[s] + '</span>' +
        '<span class="cs-time" id="ct-' + char.id + '-' + s + '"></span>' +
        '</div>';
    });
    card.innerHTML =
      '<div class="char-header">' +
      '<span class="char-name">' + (char.name || char.id) + '</span>' +
      '<span class="char-id-tag">' + char.id + '</span>' +
      '</div>' +
      '<div class="char-stages">' + stagesHtml + '</div>';
    document.getElementById('char-cards').appendChild(card);
  }

  function setCS(cid, s, cls) {
    const el = document.getElementById('cs-' + cid + '-' + s);
    if (el) el.className = 'cs ' + cls;
  }
  function setCD(cid, s, cls) {
    const el = document.getElementById('cd-' + cid + '-' + s);
    if (el) el.className = 'dot' + (cls ? ' ' + cls : '');
  }
  function setCT(cid, s, txt) {
    const el = document.getElementById('ct-' + cid + '-' + s);
    if (el) el.textContent = txt;
  }
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    logger.info("GET / — serving pipeline UI")
    return render_template_string(_HTML)


@app.post("/run")
def run():
    """
    Accept a scripture image and stream the full pipeline as SSE events.

    Form field:
        image (file): input .jpg/.jpeg/.png scripture page image

    SSE event shapes:
        {"type":"stage",  "stage":"extraction", "status":"start|done|error", "characters":[...]}
        {"type":"char",   "char_id":"...", "stage":"img_gen|3d|rig|anim", "status":"start|done|error"}
        {"type":"result", "char_id":"...", "url":"/download/<session>/<file>"}
        {"type":"done",   "session_id":"..."}
        {"type":"error",  "message":"..."}

    Returns:
        text/event-stream — SSE stream
        400 JSON — missing image field
    """
    logger.info("POST /run from %s", request.remote_addr)

    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file field."}), 400

    image_file = request.files["image"]

    session_id = str(uuid.uuid4())
    session_dir = PIPELINE_OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    SESSIONS[session_id] = {"dir": str(session_dir), "results": {}}

    ext = Path(image_file.filename).suffix.lower() or ".jpg"
    image_path = session_dir / f"input{ext}"
    image_file.save(str(image_path))
    logger.info("Saved input: %s", image_path)

    event_q: queue.Queue = queue.Queue()

    def emit(event: dict):
        event_q.put(event)

    def pipeline():
        try:
            # ── Stage 1: JSON Extraction ──────────────────────────────────
            emit({"type": "stage", "stage": "extraction", "status": "start"})
            logger.info("Starting JSON extraction")
            try:
                result = extract_story_graphs(str(image_path))
            except Exception as e:
                logger.exception("JSON extraction failed")
                emit({"type": "stage", "stage": "extraction", "status": "error", "message": str(e)})
                emit({"type": "error", "message": f"JSON extraction failed: {e}"})
                return

            graph1 = result["graph1"]
            graph2 = result.get("graph2", {})
            characters = graph1.get("characters", [])

            # char_id → first matching animation from graph2
            anim_lookup: dict = {}
            for anim in graph2.get("animations", []):
                cid = anim.get("character_id", "")
                if cid and cid not in anim_lookup:
                    anim_lookup[cid] = anim

            emit({
                "type": "stage", "stage": "extraction", "status": "done",
                "characters": [
                    {"id": c["id"], "name": c.get("name", c["id"])}
                    for c in characters
                ],
            })
            logger.info("Extraction done: %d character(s)", len(characters))

            if not characters:
                emit({"type": "error", "message": "No characters found in extraction output."})
                return

            # ── Queue-based pipelining ────────────────────────────────────
            # img_gen thread generates images and pushes to img_q.
            # Main loop consumes img_q and runs 3d → rig → anim sequentially.
            # This overlaps img_gen time with the 3D pipeline for each character.
            img_q: queue.Queue = queue.Queue()

            def img_gen_worker():
                for char in characters:
                    cid = char["id"]
                    cached_img = CACHE_IMAGES_DIR / f"{cid}.png"

                    if cached_img.exists():
                        logger.info("Cache hit image for %s: %s", cid, cached_img)
                        emit({"type": "char", "char_id": cid, "stage": "img_gen", "status": "done"})
                        img_q.put((char, str(cached_img), None))
                        continue

                    prompt = char.get("prompt", "")
                    emit({"type": "char", "char_id": cid, "stage": "img_gen", "status": "start"})
                    try:
                        out_img = str(session_dir / f"{cid}_img.png")
                        img_path_out = generate_character_image(prompt, output_path=out_img)
                        shutil.copy(img_path_out, cached_img)
                        logger.info("Cached image: %s", cached_img)
                        emit({"type": "char", "char_id": cid, "stage": "img_gen", "status": "done"})
                        img_q.put((char, img_path_out, None))
                    except Exception as e:
                        logger.exception("img_gen failed for %s", cid)
                        emit({"type": "char", "char_id": cid, "stage": "img_gen",
                              "status": "error", "message": str(e)})
                        img_q.put((char, None, str(e)))
                img_q.put(None)  # sentinel

            threading.Thread(target=img_gen_worker, daemon=True).start()

            while True:
                item = img_q.get()
                if item is None:
                    break

                char, img_path_out, img_err = item
                cid = char["id"]

                if img_err:
                    logger.warning("Skipping %s — img_gen failed", cid)
                    continue

                # Use for/break pattern so a stage failure skips to next character
                for _ in [None]:
                    cached_glb = CACHE_GLB_DIR / f"{cid}.glb"
                    cached_anim = CACHE_ANIMATED_DIR / f"{cid}.glb"

                    # Full cache hit — skip all stages and serve directly
                    if cached_anim.exists():
                        logger.info("Full cache hit for %s — serving animated GLB", cid)
                        emit({"type": "char", "char_id": cid, "stage": "3d", "status": "done"})
                        emit({"type": "char", "char_id": cid, "stage": "rig", "status": "done"})
                        emit({"type": "char", "char_id": cid, "stage": "anim", "status": "done"})
                        SESSIONS[session_id]["results"][cid] = str(cached_anim)
                        emit({"type": "result", "char_id": cid,
                              "url": f"/cache/animated_glb/{cid}.glb"})
                        break

                    # 3D generation (TripoSR) — check GLB cache
                    glb_path = None
                    if cached_glb.exists():
                        logger.info("Cache hit GLB for %s: %s", cid, cached_glb)
                        glb_path = str(cached_glb)
                        emit({"type": "char", "char_id": cid, "stage": "3d", "status": "done"})
                    else:
                        emit({"type": "char", "char_id": cid, "stage": "3d", "status": "start"})
                        try:
                            glb_path = generate_glb(img_path_out, asset_name=cid)
                            shutil.copy(glb_path, cached_glb)
                            logger.info("Cached GLB: %s", cached_glb)
                            emit({"type": "char", "char_id": cid, "stage": "3d", "status": "done"})
                        except Exception as e:
                            logger.exception("3D gen failed for %s", cid)
                            emit({"type": "char", "char_id": cid, "stage": "3d",
                                  "status": "error", "message": str(e)})
                            break

                    # Rigging (RigAnything)
                    emit({"type": "char", "char_id": cid, "stage": "rig", "status": "start"})
                    rigged_glb = None
                    try:
                        rigged_glb = rig_3d_model(glb_path)
                        emit({"type": "char", "char_id": cid, "stage": "rig", "status": "done"})
                    except Exception as e:
                        logger.exception("Rigging failed for %s", cid)
                        emit({"type": "char", "char_id": cid, "stage": "rig",
                              "status": "error", "message": str(e)})
                        break

                    # Animation (Kimodo motion + retarget)
                    emit({"type": "char", "char_id": cid, "stage": "anim", "status": "start"})
                    try:
                        anim_data = anim_lookup.get(cid)
                        action   = anim_data["action"] if anim_data else "a person stands in a neutral T-pose"
                        duration = float(anim_data.get("duration_seconds", 5.0)) if anim_data else 5.0

                        npz_path = generate_motion(action, duration=duration)

                        animated_glb = str(CACHE_ANIMATED_DIR / f"{cid}.glb")
                        blend_animation(rigged_glb, npz_path, animated_glb, fps=30)

                        logger.info("Cached animated GLB: %s", animated_glb)
                        SESSIONS[session_id]["results"][cid] = animated_glb
                        emit({"type": "char", "char_id": cid, "stage": "anim", "status": "done"})
                        emit({"type": "result", "char_id": cid,
                              "url": f"/cache/animated_glb/{cid}.glb"})
                    except Exception as e:
                        logger.exception("Animation failed for %s", cid)
                        emit({"type": "char", "char_id": cid, "stage": "anim",
                              "status": "error", "message": str(e)})

            emit({"type": "done", "session_id": session_id})
            logger.info("Pipeline complete | session=%s", session_id)

        except Exception as e:
            logger.exception("Unexpected pipeline error")
            emit({"type": "error", "message": f"Unexpected error: {e}"})

    threading.Thread(target=pipeline, daemon=True).start()

    def generate():
        while True:
            event = event_q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/download/<session_id>/<filename>")
def download_file(session_id: str, filename: str):
    """
    Serve a generated animated GLB by session ID and filename.

    Returns:
        200 application/octet-stream — GLB binary
        400 JSON — path traversal attempt
        404 JSON — session or file not found
    """
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    filepath = Path(session["dir"]) / filename

    try:
        filepath.resolve().relative_to(Path(session["dir"]).resolve())
    except ValueError:
        logger.error("Directory traversal attempt: session=%s file=%s", session_id, filename)
        return jsonify({"error": "Invalid path"}), 400

    if not filepath.exists() or not filepath.is_file():
        return jsonify({"error": "File not found"}), 404

    logger.info("Serving %s | session=%s", filename, session_id)
    return send_file(
        str(filepath),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
    )


@app.get("/cache/<folder>/<filename>")
def serve_cache_file(folder: str, filename: str):
    """
    Serve a file from the pipeline cache directories.

    Path parameters:
        folder:   One of "images", "glb", "animated_glb"
        filename: Filename within the folder

    Returns:
        200 application/octet-stream — file contents
        400 JSON — invalid folder or path traversal
        404 JSON — file not found
    """
    if folder not in {"images", "glb", "animated_glb"}:
        return jsonify({"error": "Invalid folder"}), 400

    cache_dir = PIPELINE_OUTPUT_DIR / folder
    filepath = cache_dir / filename

    try:
        filepath.resolve().relative_to(cache_dir.resolve())
    except ValueError:
        logger.error("Directory traversal attempt: folder=%s file=%s", folder, filename)
        return jsonify({"error": "Invalid path"}), 400

    if not filepath.exists() or not filepath.is_file():
        return jsonify({"error": "File not found"}), 404

    logger.info("Serving cache %s/%s", folder, filename)
    return send_file(
        str(filepath),
        mimetype="model/gltf-binary",
        as_attachment=False,
        download_name=filename,
    )


@app.post("/animate")
def animate():
    """
    Synchronous pipeline endpoint for WebAR clients.

    Accepts a scripture image, runs the full pipeline (extraction → image gen →
    3D → rigging → animation), and returns a JSON array of animated GLB URLs.

    Form field:
        image (file): input .jpg/.jpeg/.png image

    Returns:
        200 application/json — [{"url": "https://host/cache/animated_glb/{id}.glb", "char_id": "..."}, ...]
        400 JSON — missing image field
        422 JSON — no characters found
        500 JSON — pipeline error
    """
    logger.info("POST /animate from %s", request.remote_addr)

    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file field."}), 400

    image_file = request.files["image"]
    session_id = str(uuid.uuid4())
    session_dir = PIPELINE_OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image_file.filename).suffix.lower() or ".jpg"
    image_path = session_dir / f"input{ext}"
    image_file.save(str(image_path))
    logger.info("Saved input for /animate: %s", image_path)

    # Stage 1: extraction
    try:
        result = extract_story_graphs(str(image_path))
    except Exception as e:
        logger.exception("Extraction failed in /animate")
        return jsonify({"error": f"Extraction failed: {e}"}), 500

    graph1 = result["graph1"]
    graph2 = result.get("graph2", {})
    characters = graph1.get("characters", [])

    if not characters:
        return jsonify({"error": "No characters found in the image."}), 422

    anim_lookup: dict = {}
    for anim in graph2.get("animations", []):
        cid = anim.get("character_id", "")
        if cid and cid not in anim_lookup:
            anim_lookup[cid] = anim

    base_url = request.host_url.rstrip("/")
    glb_results = []

    for char in characters:
        cid = char["id"]
        logger.info("Processing character: %s", cid)

        # Full animation cache hit — skip all stages
        cached_anim = CACHE_ANIMATED_DIR / f"{cid}.glb"
        if cached_anim.exists():
            logger.info("Cache hit animated GLB for %s", cid)
            glb_results.append({"url": f"{base_url}/cache/animated_glb/{cid}.glb", "char_id": cid})
            continue

        # Image generation
        cached_img = CACHE_IMAGES_DIR / f"{cid}.png"
        if cached_img.exists():
            img_path = str(cached_img)
            logger.info("Cache hit image for %s", cid)
        else:
            prompt = char.get("prompt", "")
            try:
                out_img = str(session_dir / f"{cid}_img.png")
                img_path = generate_character_image(prompt, output_path=out_img)
                shutil.copy(img_path, cached_img)
            except Exception as e:
                logger.exception("Image gen failed for %s", cid)
                continue

        # 3D generation
        cached_glb = CACHE_GLB_DIR / f"{cid}.glb"
        if cached_glb.exists():
            glb_path = str(cached_glb)
            logger.info("Cache hit GLB for %s", cid)
        else:
            try:
                glb_path = generate_glb(img_path, asset_name=cid)
                shutil.copy(glb_path, cached_glb)
            except Exception as e:
                logger.exception("3D gen failed for %s", cid)
                continue

        # Rigging
        try:
            rigged_glb = rig_3d_model(glb_path)
        except Exception as e:
            logger.exception("Rigging failed for %s", cid)
            continue

        # Animation
        try:
            anim_data = anim_lookup.get(cid)
            action = anim_data["action"] if anim_data else "a person stands in a neutral T-pose"
            duration = float(anim_data.get("duration_seconds", 5.0)) if anim_data else 5.0
            npz_path = generate_motion(action, duration=duration)
            animated_glb = str(CACHE_ANIMATED_DIR / f"{cid}.glb")
            blend_animation(rigged_glb, npz_path, animated_glb, fps=30)
            glb_results.append({"url": f"{base_url}/cache/animated_glb/{cid}.glb", "char_id": cid})
            logger.info("Animated GLB ready for %s", cid)
        except Exception as e:
            logger.exception("Animation failed for %s", cid)
            continue

    shutil.rmtree(session_dir, ignore_errors=True)

    if not glb_results:
        return jsonify({"error": "Pipeline completed but produced no animated GLBs."}), 500

    logger.info("POST /animate complete — %d GLB(s) | session=%s", len(glb_results), session_id)
    return jsonify(glb_results)


if __name__ == "__main__":
    logger.info("Starting AR Storytelling Pipeline")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
