// ar.js — WebXR AR scene: hit-test reticle → tap to place animated characters + voice overs
import * as THREE from 'three';
import { GLTFLoader }  from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

// ── Scene data from sessionStorage ────────────────────────────────────────────

const sceneData = JSON.parse(sessionStorage.getItem('ar_scene') || 'null');
if (!sceneData?.characters?.length) {
  window.location.href = './capture.html';
}

// ── DOM refs ──────────────────────────────────────────────────────────────────

const startScreenEl = document.getElementById('start-screen');
const startBtn      = document.getElementById('start-btn');
const ssStoryEl     = document.getElementById('ss-story');
const ssCharsEl     = document.getElementById('ss-chars');
const overlayEl     = document.getElementById('overlay');
const hintEl        = document.getElementById('hint');
const loadCharsEl   = document.getElementById('loading-chars');
const storyBadgeEl  = document.getElementById('story-badge');
const errorEl       = document.getElementById('error-msg');

// ── Populate start screen ─────────────────────────────────────────────────────

ssStoryEl.textContent = sceneData.story || '';
ssCharsEl.textContent = sceneData.characters.map(c => c.name).join(' · ');

// ── Three.js renderer ─────────────────────────────────────────────────────────

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setClearColor(0x000000, 0);   // fully transparent — lets AR camera show through
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 20);

// ── Lighting ──────────────────────────────────────────────────────────────────

scene.add(new THREE.AmbientLight(0xffffff, 6.0));
const sun = new THREE.DirectionalLight(0xffffff, 4.0);
sun.position.set(1, 3, 2);
scene.add(sun);
// Fill light from below to avoid dark undersides
const fill = new THREE.DirectionalLight(0xffffff, 2.0);
fill.position.set(-1, -1, -2);
scene.add(fill);

// ── Reticle: rectangle with corner accents + center ring ──────────────────────

const reticleGroup = new THREE.Group();
reticleGroup.matrixAutoUpdate = false;
reticleGroup.visible = false;
scene.add(reticleGroup);

const RW = 0.22, RD = 0.16;         // book-page size in metres (~A5)
const CYAN  = new THREE.Color(0x22d3ee);
const WHITE = new THREE.Color(0xffffff);

// Main rectangle outline
{
  const pts = [
    new THREE.Vector3(-RW / 2, 0, -RD / 2),
    new THREE.Vector3( RW / 2, 0, -RD / 2),
    new THREE.Vector3( RW / 2, 0,  RD / 2),
    new THREE.Vector3(-RW / 2, 0,  RD / 2),
    new THREE.Vector3(-RW / 2, 0, -RD / 2),
  ];
  reticleGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: CYAN })
  ));
}

// Corner accent lines (L-shapes at each corner)
const CORNER = 0.03;
[
  { cx: -RW / 2, cz: -RD / 2, dx:  1, dz:  1 },
  { cx:  RW / 2, cz: -RD / 2, dx: -1, dz:  1 },
  { cx:  RW / 2, cz:  RD / 2, dx: -1, dz: -1 },
  { cx: -RW / 2, cz:  RD / 2, dx:  1, dz: -1 },
].forEach(({ cx, cz, dx, dz }) => {
  const mat = new THREE.LineBasicMaterial({ color: WHITE });
  reticleGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(cx,                 0, cz),
      new THREE.Vector3(cx + dx * CORNER,   0, cz),
    ]), mat
  ));
  reticleGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(cx, 0, cz),
      new THREE.Vector3(cx, 0, cz + dz * CORNER),
    ]), mat
  ));
});

// Centre ring
{
  const geom = new THREE.RingGeometry(0.04, 0.065, 32);
  geom.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
  reticleGroup.add(new THREE.Mesh(
    geom,
    new THREE.MeshBasicMaterial({ color: CYAN, side: THREE.DoubleSide })
  ));
}

// ── State ─────────────────────────────────────────────────────────────────────

let hitTestSource = null;
let xrSession     = null;
let lastHitMatrix = null;
let placed        = false;
const mixers      = [];
let lastTimestamp = 0;

// ── Render loop ───────────────────────────────────────────────────────────────

renderer.setAnimationLoop((timestamp, frame) => {
  const delta = lastTimestamp ? (timestamp - lastTimestamp) / 1000 : 0;
  lastTimestamp = timestamp;

  if (!frame) return;

  if (hitTestSource && !placed) {
    const refSpace = renderer.xr.getReferenceSpace();
    const results  = frame.getHitTestResults(hitTestSource);

    if (results.length > 0) {
      const pose = results[0].getPose(refSpace);
      if (pose) {
        reticleGroup.visible = true;
        reticleGroup.matrix.fromArray(pose.transform.matrix);
        lastHitMatrix = pose.transform.matrix.slice();
        hintEl.textContent = 'Tap to place the scene';
      }
    } else {
      reticleGroup.visible = false;
      hintEl.textContent   = 'Move phone over a flat surface';
    }
  }

  for (const m of mixers) m.update(delta);
  renderer.render(scene, camera);
});

// ── Start AR session ──────────────────────────────────────────────────────────

async function startAR() {
  if (!navigator.xr) {
    return showError('WebXR not available. Open in Chrome on Android.');
  }

  const supported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  if (!supported) {
    return showError('AR not supported. Needs Android Chrome + ARCore.');
  }

  startBtn.disabled    = true;
  startBtn.textContent = 'Starting…';

  try {
    const session = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['hit-test'],
      optionalFeatures: ['dom-overlay'],
      domOverlay: { root: overlayEl },
    });

    xrSession = session;

    // Three.js defaults to 'local-floor' which many devices don't support.
    // 'local' is broadly supported across all ARCore devices.
    renderer.xr.setReferenceSpaceType('local');

    // sessionstart fires after Three.js's makeXRCompatible() completes and
    // the internal reference space is ready — safe to request hit-test here.
    renderer.xr.addEventListener('sessionstart', async () => {
      try {
        const viewerSpace = await session.requestReferenceSpace('viewer');
        hitTestSource = await session.requestHitTestSource({ space: viewerSpace });
        hintEl.textContent   = 'Move phone over a flat surface';
        hintEl.style.display = 'block';
      } catch (err) {
        showError('Surface detection failed: ' + err.message);
      }
    }, { once: true });

    // Do NOT await — camera feed starts as soon as makeXRCompatible() resolves.
    renderer.xr.setSession(session);

    startScreenEl.style.display = 'none';

    if (sceneData.story) {
      storyBadgeEl.textContent   = sceneData.story;
      storyBadgeEl.style.display = 'block';
    }

    hintEl.textContent   = 'Initialising…';
    hintEl.style.display = 'block';

    session.addEventListener('select', onTap);
    session.addEventListener('end', () => {
      hitTestSource = null;
      xrSession     = null;
      storyBadgeEl.style.display = 'none';
      hintEl.style.display       = 'none';
    });

  } catch (err) {
    showError('Could not start AR: ' + err.message);
    startBtn.disabled    = false;
    startBtn.textContent = 'Start AR';
  }
}

// ── Tap to place ──────────────────────────────────────────────────────────────

function onTap() {
  if (placed || !lastHitMatrix) return;
  placed = true;

  reticleGroup.visible = false;
  hintEl.style.display = 'none';
  loadCharsEl.style.display = 'flex';

  // Extract only position from hit matrix — we do NOT apply the surface
  // quaternion so characters always stand upright in world Y-up space.
  const matrix = new THREE.Matrix4().fromArray(lastHitMatrix);
  const pos    = new THREE.Vector3();
  const quat   = new THREE.Quaternion();
  const scl    = new THREE.Vector3();
  matrix.decompose(pos, quat, scl);

  placeScene(pos);
}

// ── Place characters in the world ─────────────────────────────────────────────

const CHAR_HEIGHT_M  = 0.05;   // character height (metres) — ~5cm figurine on a page
const CHAR_SPACING_M = 0.09;   // centre-to-centre spacing (metres) — wider gap

function placeScene(anchorPos) {
  // Anchor sits at the detected surface point; no quaternion so models stay upright
  const anchor = new THREE.Group();
  anchor.position.copy(anchorPos);
  scene.add(anchor);

  // Camera angle — used when only one character is present
  const camPos = new THREE.Vector3();
  camera.getWorldPosition(camPos);
  const camFaceY = Math.atan2(
    camPos.x - anchorPos.x,
    camPos.z - anchorPos.z
  ) + Math.PI;

  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
  const loader = new GLTFLoader();
  loader.setDRACOLoader(dracoLoader);

  let loaded = 0;
  const total = sceneData.characters.length;

  sceneData.characters.forEach((char, idx) => {
    loader.load(
      char.url,
      gltf => {
        const model = gltf.scene;

        // 1. Bounding box before adding to any parent:
        //    world matrix = local matrix (no parent), so bounds are in model-local space.
        const rawBox = new THREE.Box3().setFromObject(model);
        const rawH   = rawBox.max.y - rawBox.min.y;

        // 2. Scale to target height
        const s = rawH > 0.0001 ? CHAR_HEIGHT_M / rawH : 1;
        model.scale.setScalar(s);

        // 3. Lift so feet sit exactly on the detected surface.
        //    -(rawBox.min.y * s) pushes the scaled mesh bottom to y=0 in anchor space.
        //    Add a tiny floor buffer (2 mm) so walk-animation toe-dips don't clip through.
        model.position.y = -(rawBox.min.y * s) + 0.002;

        // 4. Spread characters in a row along anchor X axis
        model.position.x = (idx - (total - 1) / 2) * CHAR_SPACING_M;
        model.position.z = 0;

        // 5. Orientation:
        //    • 1 character  → face camera
        //    • multiple     → face the scene centre (i.e. face each other)
        //    Model front is -Z (glTF default). rotation.y = -PI/2 turns -Z to face +X;
        //    rotation.y = +PI/2 turns -Z to face -X.
        if (total === 1) {
          model.rotation.y = camFaceY;
        } else {
          // Characters to the left of centre face right; those to the right face left
          model.rotation.y = model.position.x < 0 ? -Math.PI / 2 : Math.PI / 2;
        }

        // 6. Add to anchor after all local transforms are set
        anchor.add(model);

        // Animations disabled — show characters in upright bind pose.
        // Re-enable once standing position is confirmed.

        loaded++;
        if (loaded === total) onAllLoaded();
      },
      undefined,
      err => {
        console.error('GLB load failed for', char.char_id, err);
        loaded++;
        if (loaded === total) onAllLoaded();
      }
    );
  });
}

function onAllLoaded() {
  loadCharsEl.style.display = 'none';
  startVoiceovers(sceneData.timeline);
}

// ── Voice overs ───────────────────────────────────────────────────────────────
// Uses Web Speech API (built into Chrome Android).
// Utterances are chained via onend to guarantee sequential playback,
// with a gap derived from the timeline's start/end times.
// Loops indefinitely so the narration replays.

function startVoiceovers(timeline) {
  if (!timeline?.length || !window.speechSynthesis) return;

  const entries = [...timeline].sort((a, b) => a.start_time - b.start_time);
  let idx = 0;

  function speakNext() {
    const entry = entries[idx % entries.length];
    idx++;

    const utter = new SpeechSynthesisUtterance(entry.voiceover);
    utter.lang  = 'en-IN';
    utter.rate  = 0.85;
    utter.pitch = 1.0;

    utter.onend = () => {
      // Gap to next entry (or 2 s buffer before looping)
      const next   = entries[idx % entries.length];
      const gapSec = (next && idx < entries.length)
        ? Math.max(0, next.start_time - entry.end_time)
        : 2;
      setTimeout(speakNext, Math.max(400, gapSec * 1000));
    };

    utter.onerror = () => setTimeout(speakNext, 1000);

    speechSynthesis.speak(utter);
  }

  speechSynthesis.cancel(); // clear any stale queue
  speakNext();
}

// ── Status / error helpers ────────────────────────────────────────────────────

function showError(msg) {
  errorEl.textContent   = msg;
  errorEl.style.display = 'block';
}


// ── Init ──────────────────────────────────────────────────────────────────────

startBtn.addEventListener('click', startAR);
