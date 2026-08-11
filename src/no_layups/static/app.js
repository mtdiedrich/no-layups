import { createViewer } from "./viewer.js";

const STEP_LABELS = {
  validating: "Checking video…",
  camera_check: "Checking camera stability…",
  extracting_pose: "Analyzing motion…",
  filtering: "Smoothing tracking…",
  segmenting: "Finding swing moments…",
  computing: "Computing metrics…",
};

const POLL_INTERVAL_MS = 1000;

function show(screenId) {
  for (const el of document.querySelectorAll("section")) {
    el.classList.toggle("active", el.id === screenId);
  }
}

function prettyKey(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(value, unit) {
  if (value === null || value === undefined) return "—";
  return unit ? `${value.toFixed(1)} ${unit}` : value.toFixed(1);
}

function formatDelta(value, unit) {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return unit ? `${sign}${value.toFixed(1)} ${unit}` : `${sign}${value.toFixed(1)}`;
}

function main() {
  const uploadForm = document.getElementById("upload-form");
  const dropZone = document.getElementById("drop-zone");
  const videoInput = document.getElementById("video-input");
  const selectedFile = document.getElementById("selected-file");
  const uploadSubmit = document.getElementById("upload-submit");
  const uploadError = document.getElementById("upload-error");
  const processingStep = document.getElementById("processing-step");
  const processingCancel = document.getElementById("processing-cancel");

  let pollHandle = null;

  function stopPolling() {
    if (pollHandle !== null) {
      clearTimeout(pollHandle);
      pollHandle = null;
    }
  }

  function resetToUpload(message) {
    stopPolling();
    uploadForm.reset();
    selectedFile.textContent = "";
    uploadSubmit.disabled = true;
    processingCancel.hidden = true;
    if (message) {
      uploadError.textContent = message;
      uploadError.hidden = false;
    } else {
      uploadError.hidden = true;
    }
    show("upload-screen");
  }

  dropZone.addEventListener("click", () => videoInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      videoInput.files = e.dataTransfer.files;
      videoInput.dispatchEvent(new Event("change"));
    }
  });
  videoInput.addEventListener("change", () => {
    const file = videoInput.files[0];
    selectedFile.textContent = file ? file.name : "";
    uploadSubmit.disabled = !file;
  });

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = videoInput.files[0];
    if (!file) return;
    const handedness = uploadForm.elements["handedness"].value;

    const body = new FormData();
    body.append("video", file);
    body.append("handedness", handedness);

    uploadSubmit.disabled = true;
    try {
      const res = await fetch("/api/swings", { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `upload failed (${res.status})`);
      }
      const { job_id } = await res.json();
      uploadError.hidden = true;
      show("processing-screen");
      pollJob(job_id);
    } catch (err) {
      resetToUpload(err.message);
    }
  });

  function pollJob(jobId) {
    processingStep.textContent = "Uploading…";
    processingCancel.hidden = true;

    async function tick() {
      let job;
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error(`job status request failed (${res.status})`);
        job = await res.json();
      } catch (err) {
        processingStep.textContent = err.message;
        processingCancel.hidden = false;
        return;
      }

      if (job.status === "queued" || job.status === "processing") {
        processingStep.textContent = STEP_LABELS[job.step] || "Processing…";
        pollHandle = setTimeout(tick, POLL_INTERVAL_MS);
        return;
      }

      if (job.status === "error") {
        processingStep.textContent = job.error?.message || "Processing failed.";
        processingCancel.hidden = false;
        return;
      }

      // status === "done"
      try {
        const swing = await fetch(`/api/swings/${job.swing_id}`).then((r) => r.json());
        showAnalysis(job.swing_id, swing);
      } catch (err) {
        processingStep.textContent = "Could not load the analyzed swing.";
        processingCancel.hidden = false;
      }
    }

    tick();
  }

  processingCancel.addEventListener("click", () => resetToUpload());

  function showAnalysis(swingId, swingData) {
    show("analysis-screen");

    const canvas = document.getElementById("viewer-canvas");
    const refCanvas = document.getElementById("ref-viewer-canvas");
    const viewer = createViewer(canvas, swingData);
    let refViewer = null;
    let referenceSwingData = null;
    let comparisonData = null;

    const playButton = document.getElementById("play-pause");
    const speedSelect = document.getElementById("speed-select");
    const frameSlider = document.getElementById("frame-slider");
    const frameLabel = document.getElementById("frame-label");
    const mirrorToggle = document.getElementById("mirror-toggle");
    const newSwingButton = document.getElementById("new-swing");
    const tickRow = document.getElementById("tick-row");
    const metricsTbody = document.getElementById("metrics-tbody");
    const referenceBadge = document.getElementById("reference-badge");

    const compareToggle = document.getElementById("compare-toggle");
    const compareControlsEl = document.getElementById("compare-controls");
    const comparePlayButton = document.getElementById("compare-play-pause");
    const phaseSlider = document.getElementById("phase-slider");
    const phaseLabel = document.getElementById("phase-label");

    const adjustButton = document.getElementById("adjust-keyframes");
    const editorControls = document.getElementById("keyframe-editor-controls");
    const pendingSpan = document.getElementById("pending-keyframes");
    const applyButton = document.getElementById("apply-keyframes");
    const cancelButton = document.getElementById("cancel-keyframes");
    const keyframeError = document.getElementById("keyframe-error");

    frameSlider.max = String(viewer.frameCount - 1);
    frameSlider.value = "0";
    mirrorToggle.checked = false;
    compareToggle.checked = false;
    compareToggle.disabled = true;
    referenceBadge.hidden = true;

    let isPlaying = false;
    let mainCurrentFrame = 0;
    function setPlaying(next) {
      isPlaying = next;
      playButton.textContent = isPlaying ? "Pause" : "Play";
    }
    setPlaying(false);

    playButton.onclick = () => {
      if (isPlaying) {
        viewer.pause();
        setPlaying(false);
      } else {
        viewer.play();
        setPlaying(true);
      }
    };

    speedSelect.onchange = () => viewer.setSpeed(parseFloat(speedSelect.value));

    frameSlider.oninput = () => {
      viewer.setFrame(parseInt(frameSlider.value, 10));
      setPlaying(false);
    };

    mirrorToggle.onchange = () => {
      viewer.setMirror(mirrorToggle.checked);
      if (refViewer) refViewer.setMirror(mirrorToggle.checked);
    };

    for (const [id, preset] of [
      ["cam-face-on", "faceOn"],
      ["cam-down-the-line", "downTheLine"],
      ["cam-top", "top"],
    ]) {
      document.getElementById(id).onclick = () => {
        viewer.setCameraPreset(preset);
        if (refViewer) refViewer.setCameraPreset(preset);
      };
    }

    viewer.onFrameChange((i) => {
      mainCurrentFrame = i;
      frameSlider.value = String(i);
      frameLabel.textContent = `Frame ${i} / ${viewer.frameCount - 1}`;
      if (i === viewer.frameCount - 1) setPlaying(false);
    });

    // A fresh upload would call createViewer() again on the same canvas,
    // and Three.js only ever binds one WebGL context per canvas — reload
    // instead of trying to tear down and re-create the viewer in place.
    newSwingButton.onclick = () => window.location.reload();

    // --- Section 11.3 timeline ticks -----------------------------------
    function renderTicks(keyframes, pending) {
      tickRow.innerHTML = "";
      const marks = keyframes || pending;
      if (!marks) return;
      for (const kind of ["address", "top", "impact"]) {
        const frameIdx = marks[kind];
        if (frameIdx === null || frameIdx === undefined) continue;
        const dot = document.createElement("div");
        dot.className = `tick ${keyframes ? kind : "unset"}`;
        dot.style.left = `${(frameIdx / Math.max(1, viewer.frameCount - 1)) * 100}%`;
        dot.title = `${kind}: frame ${frameIdx}`;
        tickRow.appendChild(dot);
      }
    }
    renderTicks(swingData.keyframes, null);

    // --- Section 11.5 metrics panel --------------------------------------
    function renderMetricsTable() {
      let rows;
      if (comparisonData) {
        rows = comparisonData.metrics
          .map(
            (m) => `
          <tr>
            <td>${m.label}</td>
            <td>${formatValue(m.user, m.unit)}</td>
            <td>${formatValue(m.reference, m.unit)}</td>
            <td>${formatDelta(m.delta, m.unit)}</td>
            <td>${m.rating ? `<span class="rating-dot ${m.rating}"></span>` : "—"}</td>
          </tr>`
          )
          .join("");
      } else if (swingData.metrics) {
        rows = Object.entries(swingData.metrics)
          .map(
            ([key, m]) => `
          <tr>
            <td>${prettyKey(key)}</td>
            <td>${formatValue(m?.value, m?.unit)}</td>
            <td>—</td><td>—</td><td>—</td>
          </tr>`
          )
          .join("");
      } else {
        rows = '<tr><td colspan="5">Set key moments below to see metrics.</td></tr>';
      }
      metricsTbody.innerHTML = rows;
    }
    renderMetricsTable();

    function maybeEnableCompareToggle() {
      compareToggle.disabled = !(comparisonData && referenceSwingData);
    }

    fetch("/api/reference")
      .then((r) => (r.ok ? r.json() : null))
      .then((ref) => {
        referenceSwingData = ref;
        referenceBadge.hidden = !ref?.meta?.warnings?.includes("synthetic_reference");
        maybeEnableCompareToggle();
      });

    async function refreshComparison() {
      if (!swingData.keyframes) {
        comparisonData = null;
        maybeEnableCompareToggle();
        renderMetricsTable();
        return;
      }
      const res = await fetch(`/api/swings/${swingId}/comparison`);
      comparisonData = res.ok ? await res.json() : null;
      maybeEnableCompareToggle();
      renderMetricsTable();
    }
    refreshComparison();

    // --- Section 11.4 side-by-side comparison ----------------------------
    let comparePlaying = false;
    let comparePhase = 0;
    let comparePlayStart = 0;
    let comparePlayStartPhase = 0;
    let compareRaf = null;

    function applyPhase(p) {
      comparePhase = Math.max(0, Math.min(100, p));
      const idx = Math.round(comparePhase);
      viewer.setFrame(comparisonData.phase_to_frame.user[idx]);
      if (refViewer) refViewer.setFrame(comparisonData.phase_to_frame.reference[idx]);
      phaseSlider.value = String(idx);
      phaseLabel.textContent = `Phase ${idx}%`;
    }

    function stopComparePlayback() {
      if (compareRaf !== null) {
        cancelAnimationFrame(compareRaf);
        compareRaf = null;
      }
      comparePlaying = false;
      comparePlayButton.textContent = "Play";
    }

    function playCompareLoop() {
      compareRaf = requestAnimationFrame(playCompareLoop);
      const elapsedS = (performance.now() - comparePlayStart) / 1000;
      const speed = parseFloat(speedSelect.value);
      const ratePerSecond = 100 / (swingData.meta.frame_count / swingData.meta.fps);
      const next = comparePlayStartPhase + elapsedS * speed * ratePerSecond;
      if (next >= 100) {
        applyPhase(100);
        stopComparePlayback();
      } else {
        applyPhase(next);
      }
    }

    comparePlayButton.onclick = () => {
      if (comparePlaying) {
        stopComparePlayback();
        return;
      }
      comparePlaying = true;
      comparePlayStart = performance.now();
      comparePlayStartPhase = comparePhase >= 100 ? 0 : comparePhase;
      comparePlayButton.textContent = "Pause";
      playCompareLoop();
    };

    phaseSlider.oninput = () => {
      stopComparePlayback();
      applyPhase(parseFloat(phaseSlider.value));
    };

    function setCompareMode(enabled) {
      playButton.disabled = enabled;
      frameSlider.disabled = enabled;
      compareControlsEl.hidden = !enabled;
      refCanvas.hidden = !enabled;
      if (enabled) {
        viewer.pause();
        setPlaying(false);
        if (!refViewer) refViewer = createViewer(refCanvas, referenceSwingData);
        applyPhase(0);
      } else {
        stopComparePlayback();
      }
    }

    compareToggle.onchange = () => setCompareMode(compareToggle.checked);

    // --- Section 11.5 keyframe editor -------------------------------------
    let editMode = false;
    let pending = { address: null, top: null, impact: null };

    function updatePendingDisplay() {
      pendingSpan.textContent = `address=${pending.address ?? "?"}  top=${pending.top ?? "?"}  impact=${pending.impact ?? "?"}`;
      renderTicks(swingData.keyframes, pending);
      const complete = pending.address !== null && pending.top !== null && pending.impact !== null;
      const ordered = complete && pending.address < pending.top && pending.top < pending.impact;
      applyButton.disabled = !ordered;
      keyframeError.textContent = complete && !ordered ? "Keyframes must satisfy address < top < impact." : "";
    }

    function enterEditMode() {
      editMode = true;
      adjustButton.textContent = "Cancel adjusting";
      pending = {
        address: swingData.keyframes?.address ?? null,
        top: swingData.keyframes?.top ?? null,
        impact: swingData.keyframes?.impact ?? null,
      };
      editorControls.hidden = false;
      updatePendingDisplay();
    }

    function exitEditMode() {
      editMode = false;
      adjustButton.textContent = "Adjust key moments";
      editorControls.hidden = true;
      keyframeError.textContent = "";
    }

    adjustButton.onclick = () => (editMode ? exitEditMode() : enterEditMode());
    cancelButton.onclick = () => exitEditMode();

    document.getElementById("set-address").onclick = () => {
      pending.address = mainCurrentFrame;
      updatePendingDisplay();
    };
    document.getElementById("set-top").onclick = () => {
      pending.top = mainCurrentFrame;
      updatePendingDisplay();
    };
    document.getElementById("set-impact").onclick = () => {
      pending.impact = mainCurrentFrame;
      updatePendingDisplay();
    };

    applyButton.onclick = async () => {
      applyButton.disabled = true;
      try {
        const res = await fetch(`/api/swings/${swingId}/keyframes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(pending),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `could not update keyframes (${res.status})`);
        }
        const updated = await res.json();
        swingData.keyframes = updated.keyframes;
        swingData.metrics = updated.metrics;
        swingData.trajectories = updated.trajectories;
        swingData.meta.keyframe_source = updated.meta.keyframe_source;
        comparisonData = updated.comparison;
        maybeEnableCompareToggle();
        renderMetricsTable();
        renderTicks(swingData.keyframes, null);
        exitEditMode();
      } catch (err) {
        keyframeError.textContent = err.message;
        applyButton.disabled = false;
      }
    };

    if (!swingData.keyframes) {
      enterEditMode();
    }
  }
}

main();
