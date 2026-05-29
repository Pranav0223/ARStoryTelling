'''
we have different services in the services/
read claude.md for better understanding.

first step:
remove the exisiting html code and use this pipeline, 
allow me to upload an image from the frontend and send that to /home/ms_upendra/projects/Research/AR_StoryTelling_Pipeline/services/json_extraction/pipeline.py
get the results
  1. graph1
  2. graph2
  3. graph3


send the img prompt to the img_gen/img_inference.py check that and find the function, this returns an image 
send that image to the 2d_to_2d/generate_glb.py (find the fucntion) this returns a .glb file 
then use that .glb to rig which is in the rigging folder
and then animate which is in the kimodo_motion -> take prompt from the 
and show the output in the web page

'''

import io
import os
import logging
import tempfile
import shutil
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string

from services.rigging.rigging_inference import rig_3d_model, RIGGING_MODEL_DIR
from services.kimodo_motion.motion_inference import generate_motion
from services.kimodo_motion.retarget import blend_animation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── HTML test viewer ──────────────────────────────────────────────────────────
_HTML_VIEWER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AR Pipeline — Rigging Test</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: system-ui, sans-serif;
      max-width: 900px; margin: 50px auto; padding: 0 24px; color: #1a1a1a;
    }
    h1   { font-size: 1.7rem; margin-bottom: 6px; }
    p    { color: #555; margin: 0 0 20px; }
    .card {
      background: #f8f9fa; border: 1px solid #dee2e6;
      border-radius: 10px; padding: 20px; margin-bottom: 20px;
    }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    input[type=file] { flex: 1; min-width: 200px; }
    button {
      padding: 10px 26px; background: #2563eb; color: #fff;
      border: none; border-radius: 6px; cursor: pointer; font-size: 1rem;
      white-space: nowrap;
    }
    button:disabled { background: #93c5fd; cursor: default; }
    #status {
      min-height: 24px; margin-top: 14px;
      font-size: .9rem; color: #374151; font-family: monospace;
    }
    #status.error { color: #dc2626; }
    #status.ok    { color: #16a34a; }
    .spinner {
      display: inline-block; width: 14px; height: 14px;
      border: 2px solid #93c5fd; border-top-color: #2563eb;
      border-radius: 50%; animation: spin .7s linear infinite;
      vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    model-viewer {
      width: 100%; height: 540px;
      background: #e5e7eb; border-radius: 10px; display: none;
    }
    .dl-row { margin-top: 12px; display: none; }
    a.btn-dl {
      display: inline-block; padding: 8px 18px;
      background: #16a34a; color: #fff;
      border-radius: 6px; text-decoration: none; font-size: .9rem;
    }
    .hint { font-size: .82rem; color: #6b7280; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>AR Rigging Pipeline — Test Viewer</h1>
  <p>Upload a <code>.glb</code> mesh, run rigging, and inspect the result directly in this page. Inference takes a few minutes — the log streams to the terminal.
  &nbsp;&nbsp;<a href="/pipeline" style="color:#2563eb;font-size:.9rem">→ Full Rig &amp; Animate Pipeline</a></p>

  <div class="card">
    <div class="row">
      <input type="file" id="glbFile" accept=".glb">
      <button id="rigBtn" onclick="uploadAndRig()">Rig Model</button>
    </div>
    <div id="status"></div>
    <p class="hint">Optional params: mesh_simplify (1/0) and simplify_count can be appended as URL query params.</p>
  </div>

  <model-viewer
    id="viewer"
    auto-rotate
    camera-controls
    shadow-intensity="1"
    alt="Rigged 3D model"
    ar
  ></model-viewer>

  <div class="dl-row" id="dlRow">
    <a id="dlLink" class="btn-dl" download="rigged_model.glb">⬇ Download Rigged GLB</a>
    <span class="hint" style="margin-left:12px">Import into Blender to inspect bones and weights.</span>
  </div>

  <script>
    const setStatus = (msg, cls = '') => {
      const el = document.getElementById('status');
      el.className = cls;
      el.innerHTML = msg;
    };

    async function uploadAndRig() {
      const fileInput = document.getElementById('glbFile');
      const btn       = document.getElementById('rigBtn');
      const viewer    = document.getElementById('viewer');
      const dlRow     = document.getElementById('dlRow');
      const dlLink    = document.getElementById('dlLink');

      const file = fileInput.files[0];
      if (!file) { alert('Please select a .glb file first.'); return; }

      btn.disabled = true;
      viewer.style.display = 'none';
      dlRow.style.display  = 'none';

      setStatus('<span class="spinner"></span>Uploading model…');

      const form = new FormData();
      form.append('model', file);

      // pass optional query params from URL if present
      const search = new URLSearchParams(window.location.search);
      const params = new URLSearchParams();
      if (search.has('mesh_simplify'))  params.set('mesh_simplify',  search.get('mesh_simplify'));
      if (search.has('simplify_count')) params.set('simplify_count', search.get('simplify_count'));
      const qs = params.toString() ? '?' + params.toString() : '';

      setStatus('<span class="spinner"></span>Running rigging inference… (this takes several minutes — watch the terminal for live logs)');

      try {
        const resp = await fetch('/rig' + qs, { method: 'POST', body: form });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: resp.statusText }));
          setStatus('Error: ' + err.error, 'error');
          return;
        }

        const blob = await resp.blob();
        const url  = URL.createObjectURL(blob);

        viewer.src           = url;
        viewer.style.display = 'block';
        dlLink.href          = url;
        dlRow.style.display  = 'block';

        setStatus('✓ Rigging complete — rotate the model below to inspect it.', 'ok');
      } catch (e) {
        setStatus('Error: ' + e.message, 'error');
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    """HTML test viewer."""
    logger.info("GET / — serving test UI")
    return render_template_string(_HTML_VIEWER)


@app.post("/rig")
def rig():
    """
    Accept a .glb file, run the rigging pipeline, and return the rigged .glb.

    Form field:
        model (file): the input .glb mesh

    Query params (optional):
        mesh_simplify  (int, default 1): 1 = simplify mesh before inference, 0 = skip
        simplify_count (int, default 8192): target face count for simplification

    Returns:
        200 application/octet-stream — rigged GLB binary
        400 JSON — malformed request
        500 JSON — inference error
    """
    logger.info("POST /rig — new rigging request from %s", request.remote_addr)

    if "model" not in request.files:
        logger.warning("Missing 'model' file field in request")
        return jsonify({"error": "No model file provided. Use form field name 'model'."}), 400

    file = request.files["model"]
    if not file.filename.lower().endswith(".glb"):
        logger.warning("Unsupported file type: %s", file.filename)
        return jsonify({"error": "Only .glb files are supported."}), 400

    try:
        mesh_simplify  = int(request.args.get("mesh_simplify", 1))
        simplify_count = int(request.args.get("simplify_count", 8192))
    except ValueError as e:
        return jsonify({"error": f"Invalid query param: {e}"}), 400

    # Save upload to a temp dir using the original filename so the asset stem is clean.
    tmpdir = tempfile.mkdtemp(prefix="rig_upload_")
    try:
        upload_path = Path(tmpdir) / file.filename
        file.save(str(upload_path))
        logger.info("Saved upload to %s", upload_path)

        output_path = rig_3d_model(
            str(upload_path),
            mesh_simplify=mesh_simplify,
            simplify_count=simplify_count,
        )

        logger.info("Sending rigged model: %s", output_path)
        return send_file(
            output_path,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="rigged_model.glb",
        )

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        logger.error("Rigging runtime error: %s", e)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error during rigging")
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.debug("Cleaned up temp dir: %s", tmpdir)


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    """
    Serve a previously rigged GLB directly by its path under rigging_model/outputs/.

    Example: GET /outputs/spyro_the_dragon/spyro_the_dragon_simplified_rig.glb

    Useful for viewing a cached result in the browser without re-uploading.
    """
    full_path = RIGGING_MODEL_DIR / "outputs" / filename
    logger.info("GET /outputs/%s", filename)

    if not full_path.exists() or not full_path.is_file():
        logger.warning("Output file not found: %s", full_path)
        return jsonify({"error": f"File not found: {filename}"}), 404

    # Prevent directory traversal
    try:
        full_path.resolve().relative_to((RIGGING_MODEL_DIR / "outputs").resolve())
    except ValueError:
        logger.error("Directory traversal attempt: %s", filename)
        return jsonify({"error": "Invalid path"}), 400

    return send_file(str(full_path), mimetype="application/octet-stream")


_HTML_PIPELINE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AR Pipeline — Rig &amp; Animate</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: system-ui, sans-serif;
      max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1a1a1a;
    }
    h1   { font-size: 1.7rem; margin-bottom: 4px; }
    .sub { color: #555; margin: 0 0 24px; font-size: .95rem; }
    .back { font-size: .85rem; color: #2563eb; text-decoration: none; }
    .back:hover { text-decoration: underline; }

    .layout { display: flex; gap: 24px; align-items: flex-start; }
    .left  { flex: 0 0 360px; }
    .right { flex: 1; }

    .card {
      background: #f8f9fa; border: 1px solid #dee2e6;
      border-radius: 10px; padding: 20px; margin-bottom: 16px;
    }
    .card h2 { font-size: 1rem; margin: 0 0 14px; color: #374151; }

    .step {
      display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px;
    }
    .step-badge {
      flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
      background: #2563eb; color: #fff;
      display: flex; align-items: center; justify-content: center;
      font-size: .8rem; font-weight: 700;
    }
    .step-body { flex: 1; }
    .step-body label {
      display: block; font-size: .82rem; font-weight: 600;
      color: #374151; margin-bottom: 5px;
    }
    input[type=file], input[type=number], textarea {
      width: 100%; padding: 6px 8px; font-size: .88rem;
      border: 1px solid #d1d5db; border-radius: 5px; background: #fff;
      font-family: inherit;
    }
    textarea { resize: vertical; min-height: 72px; }
    .hint { font-size: .78rem; color: #6b7280; margin-top: 4px; }

    .divider { border: none; border-top: 1px dashed #d1d5db; margin: 4px 0 16px; }

    .options-row {
      display: flex; gap: 14px; margin-bottom: 16px;
    }
    .opt-field { flex: 1; }
    .opt-field label {
      display: block; font-size: .82rem; font-weight: 600;
      color: #374151; margin-bottom: 4px;
    }

    button#runBtn {
      width: 100%; padding: 11px; background: #2563eb; color: #fff;
      border: none; border-radius: 7px; cursor: pointer;
      font-size: 1rem; font-weight: 600;
    }
    button#runBtn:disabled { background: #93c5fd; cursor: default; }

    .status-panel { margin-top: 4px; }
    .stage {
      display: flex; align-items: center; gap: 9px;
      padding: 8px 0; border-bottom: 1px solid #e5e7eb;
      font-size: .85rem; color: #374151;
    }
    .stage:last-child { border-bottom: none; }
    .dot {
      flex-shrink: 0; width: 10px; height: 10px; border-radius: 50%;
      background: #d1d5db;
    }
    .dot.running { background: #2563eb; animation: pulse .9s infinite; }
    .dot.done    { background: #16a34a; }
    .dot.error   { background: #dc2626; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
    .stage-label { flex: 1; }
    .stage-sub { font-size: .75rem; color: #6b7280; display: block; }
    .stage-time { font-size: .75rem; color: #6b7280; white-space: nowrap; }

    model-viewer {
      width: 100%; height: 580px;
      background: #1e293b; border-radius: 12px; display: none;
    }
    .placeholder {
      width: 100%; height: 580px; border-radius: 12px;
      background: #1e293b; display: flex; flex-direction: column;
      align-items: center; justify-content: center; color: #94a3b8;
      font-size: .95rem;
    }
    .placeholder svg { margin-bottom: 16px; opacity: .45; }
    .dl-row { margin-top: 12px; display: none; }
    a.btn-dl {
      display: inline-block; padding: 9px 20px;
      background: #16a34a; color: #fff;
      border-radius: 6px; text-decoration: none; font-size: .9rem; font-weight: 600;
    }
    .anim-label { font-size: .8rem; color: #64748b; margin-top: 8px; }
  </style>
</head>
<body>
  <a class="back" href="/">← Rigging-only viewer</a>
  <h1 style="margin-top:12px">AR Storytelling Pipeline — Rig &amp; Animate</h1>
  <p class="sub">
    Upload a raw <code>.glb</code> mesh and describe the motion in plain text.
    The pipeline rigs the mesh with RigAnything, generates motion data via Kimodo,
    then maps the animation onto the bones automatically.
  </p>

  <div class="layout">
    <!-- ── LEFT ── -->
    <div class="left">
      <div class="card">
        <h2>Pipeline Inputs</h2>

        <div class="step">
          <div class="step-badge">1</div>
          <div class="step-body">
            <label>3D Mesh (.glb)</label>
            <input type="file" id="glbFile" accept=".glb">
            <div class="hint">Raw mesh — bones auto-generated by RigAnything.</div>
          </div>
        </div>

        <div class="step">
          <div class="step-badge">2</div>
          <div class="step-body">
            <label>Motion Prompt</label>
            <textarea id="promptText" placeholder="e.g. a person walks forward and waves their hand"></textarea>
            <div class="hint">Kimodo generates the .npz motion data from this description.</div>
          </div>
        </div>

        <hr class="divider">

        <div class="options-row">
          <div class="opt-field">
            <label for="duration">Duration (s)</label>
            <input type="number" id="duration" value="5" min="1" max="60" step="0.5">
          </div>
          <div class="opt-field">
            <label for="fps">FPS</label>
            <input type="number" id="fps" value="30" min="1" max="120">
          </div>
        </div>

        <button id="runBtn" onclick="runPipeline()">Run Full Pipeline</button>
      </div>

      <div class="card">
        <h2>Pipeline Status</h2>
        <div class="status-panel">
          <div class="stage">
            <div class="dot" id="d-rig"></div>
            <span class="stage-label">
              Rigging
              <span class="stage-sub">RigAnything — generates bones &amp; skin weights</span>
            </span>
            <span class="stage-time" id="t-rig"></span>
          </div>
          <div class="stage">
            <div class="dot" id="d-mot"></div>
            <span class="stage-label">
              Motion generation
              <span class="stage-sub">Kimodo — text prompt → output.npz</span>
            </span>
            <span class="stage-time" id="t-mot"></span>
          </div>
          <div class="stage">
            <div class="dot" id="d-ret"></div>
            <span class="stage-label">
              Bone mapping
              <span class="stage-sub">retarget.py — npz rotations → GLB animation track</span>
            </span>
            <span class="stage-time" id="t-ret"></span>
          </div>
        </div>
        <div id="errMsg" style="display:none;margin-top:12px;font-size:.82rem;color:#dc2626;word-break:break-word;"></div>
      </div>
    </div>

    <!-- ── RIGHT: canvas ── -->
    <div class="right">
      <div class="placeholder" id="placeholder">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg>
        Animated model will appear here
      </div>

      <model-viewer
        id="viewer"
        autoplay
        animation-name="NPZ_Animation"
        camera-controls
        shadow-intensity="1"
        alt="Animated 3D model"
        ar
      ></model-viewer>

      <div class="dl-row" id="dlRow">
        <a id="dlLink" class="btn-dl" download="animated_model.glb">⬇ Download Animated GLB</a>
        <div class="anim-label">Animation: NPZ_Animation &nbsp;|&nbsp; Import into Blender to inspect bones and weights.</div>
      </div>
    </div>
  </div>

  <script>
    function setDot(id, state) {
      document.getElementById('d-' + id).className = 'dot' + (state ? ' ' + state : '');
    }
    function setTime(id, ms) {
      document.getElementById('t-' + id).textContent = ms != null ? (ms/1000).toFixed(1)+'s' : '';
    }
    function showError(msg) {
      const el = document.getElementById('errMsg');
      el.textContent = 'Error: ' + msg;
      el.style.display = 'block';
    }

    async function runPipeline() {
      const glbFile  = document.getElementById('glbFile').files[0];
      const prompt   = document.getElementById('promptText').value.trim();
      const fps      = document.getElementById('fps').value;
      const duration = document.getElementById('duration').value;

      if (!glbFile) { alert('Please select a .glb file.'); return; }
      if (!prompt)  { alert('Please enter a motion prompt.'); return; }

      const btn = document.getElementById('runBtn');
      btn.disabled = true;

      ['rig','mot','ret'].forEach(id => { setDot(id,''); setTime(id,null); });
      document.getElementById('errMsg').style.display = 'none';
      document.getElementById('viewer').style.display = 'none';
      document.getElementById('placeholder').style.display = 'flex';
      document.getElementById('dlRow').style.display = 'none';

      const form = new FormData();
      form.append('model', glbFile);
      form.append('prompt', prompt);

      setDot('rig', 'running');

      const t0 = Date.now();
      try {
        const qs = new URLSearchParams({ fps, duration }).toString();
        const resp = await fetch('/animate?' + qs, { method: 'POST', body: form });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: resp.statusText }));
          const msg = err.error || resp.statusText;
          const lo  = msg.toLowerCase();
          if (lo.includes('rig')) {
            setDot('rig', 'error');
          } else if (lo.includes('motion') || lo.includes('kimodo') || lo.includes('npz')) {
            setDot('rig', 'done'); setDot('mot', 'error');
          } else {
            setDot('rig', 'done'); setDot('mot', 'done'); setDot('ret', 'error');
          }
          showError(msg);
          return;
        }

        // Approximate stage timings: rig ~70%, motion ~20%, retarget ~10%
        const total = Date.now() - t0;
        setDot('rig', 'done'); setTime('rig', total * 0.70);
        setDot('mot', 'done'); setTime('mot', total * 0.20);
        setDot('ret', 'done'); setTime('ret', total * 0.10);

        const blob = await resp.blob();
        const url  = URL.createObjectURL(blob);

        const viewer = document.getElementById('viewer');
        viewer.src = url;
        viewer.style.display = 'block';
        document.getElementById('placeholder').style.display = 'none';
        document.getElementById('dlLink').href = url;
        document.getElementById('dlRow').style.display = 'block';

      } catch (e) {
        setDot('rig', 'error');
        showError(e.message);
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


# ── Pipeline routes ───────────────────────────────────────────────────────────

@app.get("/pipeline")
def pipeline_viewer():
    """HTML viewer for the full rig + animate pipeline."""
    logger.info("GET /pipeline — serving pipeline UI")
    return render_template_string(_HTML_PIPELINE)


@app.post("/animate")
def animate():
    """
    Full 3-stage pipeline: rig mesh → generate motion from prompt → retarget bones.

    Form fields:
        model  (file): raw input .glb mesh
        prompt (str):  natural-language motion description for Kimodo

    Query params (optional):
        fps            (int,   default 30):   animation frame rate
        duration       (float, default 5.0):  motion duration in seconds
        mesh_simplify  (int,   default 1):    1 = simplify mesh before rigging, 0 = skip
        simplify_count (int,   default 8192): target face count for simplification

    Returns:
        200 application/octet-stream — animated GLB binary
        400 JSON — malformed request
        500 JSON — pipeline error
    """
    logger.info("POST /animate — new pipeline request from %s", request.remote_addr)

    if "model" not in request.files:
        return jsonify({"error": "Missing 'model' file field (.glb)."}), 400

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Missing 'prompt' form field — describe the motion."}), 400

    glb_file = request.files["model"]
    if not glb_file.filename.lower().endswith(".glb"):
        return jsonify({"error": "model must be a .glb file."}), 400

    try:
        fps            = int(request.args.get("fps", 30))
        duration       = float(request.args.get("duration", 5.0))
        mesh_simplify  = int(request.args.get("mesh_simplify", 1))
        simplify_count = int(request.args.get("simplify_count", 8192))
    except ValueError as e:
        return jsonify({"error": f"Invalid query param: {e}"}), 400

    tmpdir = tempfile.mkdtemp(prefix="animate_upload_")
    try:
        glb_upload   = str(Path(tmpdir) / glb_file.filename)
        animated_glb = str(Path(tmpdir) / "animated.glb")

        glb_file.save(glb_upload)
        logger.info("Saved upload — glb=%s  prompt=%r  duration=%.1fs", glb_upload, prompt, duration)

        # ── Stage 1: Rig ──────────────────────────────────────────────────────
        logger.info("Stage 1/3 — Rigging...")
        rigged_glb = rig_3d_model(
            glb_upload,
            mesh_simplify=mesh_simplify,
            simplify_count=simplify_count,
        )
        logger.info("Stage 1/3 — Rigging complete: %s", rigged_glb)

        # ── Stage 2: Generate motion via Kimodo ───────────────────────────────
        logger.info("Stage 2/3 — Generating motion: %r", prompt)
        npz_path = generate_motion(prompt, duration=duration)
        logger.info("Stage 2/3 — Motion generated: %s", npz_path)

        # ── Stage 3: Retarget (bone mapping) ─────────────────────────────────
        logger.info("Stage 3/3 — Mapping motion onto bones...")
        blend_animation(
            glb_path=rigged_glb,
            npz_path=npz_path,
            out_path=animated_glb,
            fps=fps,
        )
        logger.info("Stage 3/3 — Bone mapping complete: %s", animated_glb)

        # Read into memory so the file survives tmpdir cleanup
        animated_bytes = Path(animated_glb).read_bytes()
        logger.info("Pipeline complete — sending %d bytes", len(animated_bytes))

        return send_file(
            io.BytesIO(animated_bytes),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="animated_model.glb",
        )

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        logger.error("Pipeline error: %s", e)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error in /animate")
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.debug("Cleaned up temp dir: %s", tmpdir)


if __name__ == "__main__":
    logger.info("Starting AR StoryTelling Pipeline — Rigging API")
    app.run(host="0.0.0.0", port=5000, debug=True)
