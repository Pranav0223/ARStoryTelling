// capture.js — camera capture + /animate API call

(function () {
  'use strict';

  const video       = document.getElementById('video');
  const canvas      = document.getElementById('canvas');
  const previewImg  = document.getElementById('preview-img');
  const loadPanel   = document.getElementById('loading-panel');
  const elapsedEl   = document.getElementById('elapsed-time');
  const errorEl     = document.getElementById('error-banner');
  const titleEl     = document.getElementById('overlay-title');

  let capturedBlob = null;
  let timerHandle  = null;

  // ── Camera ──────────────────────────────────────────────────────────────────

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width:  { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      video.srcObject = stream;
    } catch (err) {
      showError('Camera access denied. Please allow camera access and reload.');
    }
  }

  // ── State management ─────────────────────────────────────────────────────────

  function setState(state) {
    // Show/hide overlay bottom bars
    document.querySelectorAll('[data-state]').forEach(el => {
      el.style.display = el.dataset.state === state ? 'flex' : 'none';
    });

    // Show camera feed or frozen preview
    video.style.display      = state === 'camera'  ? 'block' : 'none';
    previewImg.style.display = state === 'preview' ? 'block' : 'none';

    const labels = {
      camera:  'Point camera at scripture page',
      preview: 'Looks good?',
    };
    titleEl.textContent = labels[state] || '';
  }

  // ── Capture ──────────────────────────────────────────────────────────────────

  function capturePhoto() {
    const w = video.videoWidth  || 1280;
    const h = video.videoHeight || 720;
    canvas.width  = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);

    canvas.toBlob(blob => {
      capturedBlob = blob;
      previewImg.src = URL.createObjectURL(blob);
      setState('preview');
      errorEl.style.display = 'none';
    }, 'image/jpeg', 0.92);
  }

  // ── Submit to /animate ────────────────────────────────────────────────────────

  async function submitImage() {
    if (!capturedBlob) return;

    // Show loading panel, hide other UI
    loadPanel.style.display = 'flex';
    errorEl.style.display   = 'none';

    const startMs = Date.now();
    timerHandle = setInterval(() => {
      elapsedEl.textContent = Math.floor((Date.now() - startMs) / 1000) + 's';
    }, 1000);

    const form = new FormData();
    form.append('image', capturedBlob, 'capture.jpg');

    try {
      const res = await fetch('/animate', { method: 'POST', body: form });
      clearInterval(timerHandle);

      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(body.error || `HTTP ${res.status}`);
      }

      const data = await res.json();

      if (!data.characters || data.characters.length === 0) {
        throw new Error('No characters found in the image.');
      }

      sessionStorage.setItem('ar_scene', JSON.stringify(data));

      // Release the camera stream so ARCore can acquire it on ar.html
      const stream = video.srcObject;
      if (stream) stream.getTracks().forEach(t => t.stop());
      video.srcObject = null;

      window.location.href = './ar.html';

    } catch (err) {
      clearInterval(timerHandle);
      loadPanel.style.display = 'none';
      setState('preview');
      showError(err.message);
    }
  }

  // ── Error helper ──────────────────────────────────────────────────────────────

  function showError(msg) {
    errorEl.textContent  = 'Error: ' + msg;
    errorEl.style.display = 'block';
  }

  // ── Events ────────────────────────────────────────────────────────────────────

  document.getElementById('capture-btn').addEventListener('click', capturePhoto);
  document.getElementById('retake-btn').addEventListener('click', () => {
    errorEl.style.display = 'none';
    setState('camera');
  });
  document.getElementById('send-btn').addEventListener('click', submitImage);

  // ── Init ──────────────────────────────────────────────────────────────────────

  startCamera().then(() => setState('camera'));
})();
