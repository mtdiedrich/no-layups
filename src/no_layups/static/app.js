import { createViewer } from "./viewer.js";

async function main() {
  const canvas = document.getElementById("viewer-canvas");
  const swingData = await fetch("/sample.swing.json").then((r) => r.json());
  const viewer = createViewer(canvas, swingData);

  const playButton = document.getElementById("play-pause");
  const speedSelect = document.getElementById("speed-select");
  const frameSlider = document.getElementById("frame-slider");
  const frameLabel = document.getElementById("frame-label");
  const mirrorToggle = document.getElementById("mirror-toggle");

  frameSlider.max = String(viewer.frameCount - 1);

  let isPlaying = false;

  function setPlaying(next) {
    isPlaying = next;
    playButton.textContent = isPlaying ? "Pause" : "Play";
  }

  playButton.addEventListener("click", () => {
    if (isPlaying) {
      viewer.pause();
      setPlaying(false);
    } else {
      viewer.play();
      setPlaying(true);
    }
  });

  speedSelect.addEventListener("change", () => {
    viewer.setSpeed(parseFloat(speedSelect.value));
  });

  frameSlider.addEventListener("input", () => {
    viewer.setFrame(parseInt(frameSlider.value, 10));
    setPlaying(false);
  });

  mirrorToggle.addEventListener("change", () => {
    viewer.setMirror(mirrorToggle.checked);
  });

  for (const [id, preset] of [
    ["cam-face-on", "faceOn"],
    ["cam-down-the-line", "downTheLine"],
    ["cam-top", "top"],
  ]) {
    document.getElementById(id).addEventListener("click", () => viewer.setCameraPreset(preset));
  }

  viewer.onFrameChange((i) => {
    frameSlider.value = String(i);
    frameLabel.textContent = `Frame ${i} / ${viewer.frameCount - 1}`;
    if (i === viewer.frameCount - 1) setPlaying(false);
  });
}

main();
