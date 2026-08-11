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
        showAnalysis(swing);
      } catch (err) {
        processingStep.textContent = "Could not load the analyzed swing.";
        processingCancel.hidden = false;
      }
    }

    tick();
  }

  processingCancel.addEventListener("click", () => resetToUpload());

  function showAnalysis(swingData) {
    show("analysis-screen");

    const canvas = document.getElementById("viewer-canvas");
    const viewer = createViewer(canvas, swingData);

    const playButton = document.getElementById("play-pause");
    const speedSelect = document.getElementById("speed-select");
    const frameSlider = document.getElementById("frame-slider");
    const frameLabel = document.getElementById("frame-label");
    const mirrorToggle = document.getElementById("mirror-toggle");
    const newSwingButton = document.getElementById("new-swing");

    frameSlider.max = String(viewer.frameCount - 1);
    frameSlider.value = "0";
    mirrorToggle.checked = false;

    let isPlaying = false;
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

    mirrorToggle.onchange = () => viewer.setMirror(mirrorToggle.checked);

    for (const [id, preset] of [
      ["cam-face-on", "faceOn"],
      ["cam-down-the-line", "downTheLine"],
      ["cam-top", "top"],
    ]) {
      document.getElementById(id).onclick = () => viewer.setCameraPreset(preset);
    }

    viewer.onFrameChange((i) => {
      frameSlider.value = String(i);
      frameLabel.textContent = `Frame ${i} / ${viewer.frameCount - 1}`;
      if (i === viewer.frameCount - 1) setPlaying(false);
    });

    // A fresh upload would call createViewer() again on the same canvas,
    // and Three.js only ever binds one WebGL context per canvas — reload
    // instead of trying to tear down and re-create the viewer in place.
    newSwingButton.onclick = () => window.location.reload();
  }
}

main();
