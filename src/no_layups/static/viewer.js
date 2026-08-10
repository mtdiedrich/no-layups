import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const UP = new THREE.Vector3(0, 1, 0);

// Section 11.2 — the exact 12 bones.
const BONES = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

// Section 11.2 — camera presets, relative to the address hip midpoint.
const CAMERA_PRESETS = {
  faceOn: [0, 0.3, 2.5],
  downTheLine: [2.5, 0.3, 0],
  top: [0.01, 3, 0.01],
};

function placeBone(mesh, a, b) {
  const dir = b.clone().sub(a);
  const len = dir.length();
  mesh.position.copy(a).addScaledVector(dir, 0.5);
  mesh.scale.set(1, len, 1);
  mesh.quaternion.setFromUnitVectors(UP, dir.normalize());
}

function addressHipMidpoint(swingData) {
  const addressIdx = swingData.keyframes ? swingData.keyframes.address : 0;
  const frame = swingData.frames[addressIdx] ?? swingData.frames[0];
  const lh = frame?.pos?.left_hip;
  const rh = frame?.pos?.right_hip;
  if (!lh || !rh) return new THREE.Vector3(0, 0, 0);
  return new THREE.Vector3((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2);
}

function minAnkleY(frames) {
  let min = Infinity;
  for (const frame of frames) {
    for (const joint of ["left_ankle", "right_ankle"]) {
      const p = frame.pos[joint];
      if (p) min = Math.min(min, p[1]);
    }
  }
  return Number.isFinite(min) ? min : 0;
}

export function createViewer(canvas, swingData) {
  const { joints, frames } = swingData;
  const fps = swingData.meta.fps;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161c);
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 0.6));

  const grid = new THREE.GridHelper(20, 20, 0x444a58, 0x2a2e38);
  grid.position.y = minAnkleY(frames) - 0.01;
  scene.add(grid);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const jointGeometry = new THREE.SphereGeometry(0.02, 12, 8);
  const noseGeometry = new THREE.SphereGeometry(0.06, 12, 8);
  const jointMeshes = {};
  for (const joint of joints) {
    const geometry = joint === "nose" ? noseGeometry : jointGeometry;
    const material = new THREE.MeshStandardMaterial({ color: 0xffd166, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    jointMeshes[joint] = mesh;
  }

  const boneGeometry = new THREE.CylinderGeometry(0.012, 0.012, 1, 8);
  const boneMeshes = BONES.map(([a, b]) => {
    const material = new THREE.MeshStandardMaterial({ color: 0x4dd0e1, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(boneGeometry, material);
    scene.add(mesh);
    return { a, b, mesh };
  });

  const hipMid = addressHipMidpoint(swingData);

  function setCameraPreset(name) {
    const [dx, dy, dz] = CAMERA_PRESETS[name];
    camera.position.set(hipMid.x + dx, hipMid.y + dy, hipMid.z + dz);
    controls.target.copy(hipMid);
    controls.update();
  }
  setCameraPreset("faceOn");

  let currentFrame = 0;
  const frameListeners = [];

  function applyFrame(i) {
    currentFrame = Math.max(0, Math.min(frames.length - 1, i));
    const pos = frames[currentFrame].pos;
    for (const joint of joints) {
      const p = pos[joint];
      const mesh = jointMeshes[joint];
      mesh.visible = Boolean(p);
      if (p) mesh.position.set(p[0], p[1], p[2]);
    }
    for (const { a, b, mesh } of boneMeshes) {
      const pa = pos[a];
      const pb = pos[b];
      mesh.visible = Boolean(pa && pb);
      if (pa && pb) {
        placeBone(mesh, new THREE.Vector3(...pa), new THREE.Vector3(...pb));
      }
    }
    for (const cb of frameListeners) cb(currentFrame);
  }
  applyFrame(0);

  let playing = false;
  let speed = 1.0;
  let playStartTime = 0;
  let playStartFrame = 0;

  function play() {
    if (playing) return;
    playing = true;
    playStartTime = performance.now();
    playStartFrame = currentFrame;
  }

  function pause() {
    playing = false;
  }

  function setSpeed(x) {
    speed = x;
  }

  function setFrame(i) {
    pause();
    applyFrame(i);
  }

  function onFrameChange(cb) {
    frameListeners.push(cb);
  }

  function setMirror(enabled) {
    scene.scale.z = enabled ? -1 : 1;
  }

  function resize() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }

  function tick() {
    requestAnimationFrame(tick);
    if (playing) {
      const elapsedS = (performance.now() - playStartTime) / 1000;
      const next = playStartFrame + Math.floor(elapsedS * speed * fps);
      if (next >= frames.length - 1) {
        applyFrame(frames.length - 1);
        playing = false;
      } else {
        applyFrame(next);
      }
    }
    controls.update();
    renderer.render(scene, camera);
  }

  window.addEventListener("resize", resize);
  resize();
  tick();

  return {
    setFrame,
    play,
    pause,
    setSpeed,
    onFrameChange,
    setCameraPreset,
    setMirror,
    frameCount: frames.length,
  };
}
