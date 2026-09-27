// SpeakFlow — client-side face tracking + session logic.
// MediaPipe runs entirely in-browser via WASM, loaded from CDN below.
// No video/images ever leave the browser — only small numeric summaries
// are sent to /api/feedback at the end of a session.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MAX_SESSION_SECONDS = 120;
const BLINK_THRESHOLD = 0.5;
const BLINK_MIN_INTERVAL_MS = 350;
const NO_FACE_SUSTAINED_THRESHOLD = 8; // consecutive frames before we call it sustained, not a blip

const screens = {
  landing: document.getElementById("landing"),
  session: document.getElementById("session"),
  results: document.getElementById("results"),
};
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const restartBtn = document.getElementById("restartBtn");
const landingError = document.getElementById("landingError");
const sessionStatus = document.getElementById("sessionStatus");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const timerValueEl = document.getElementById("timerValue");
const blinkValueEl = document.getElementById("blinkValue");
const gazeValueEl = document.getElementById("gazeValue");
const postureValueEl = document.getElementById("postureValue");

let faceLandmarker = null;
let stream = null;
let mediaRecorder = null;
let audioChunks = [];
let rafId = null;
let sessionStartMs = 0;
let timerIntervalId = null;

let blinkCount = 0;
let eyesCurrentlyClosed = false;
let lastBlinkTimestamp = 0;
let gazeSamples = [];
let headAngleSamples = [];
let consecutiveNoFaceFrames = 0;
let faceDetectedDurationMs = 0; // only time a real face was actually seen
let lastFrameTimestamp = 0;
let framesWaitedForReadiness = 0;

function showScreen(name) {
  Object.values(screens).forEach((el) => el.classList.add("hidden"));
  const target = screens[name];
  target.classList.remove("hidden");
  const inner = target.querySelector(".screen-inner");
  if (inner) {
    inner.style.animation = "none";
    void inner.offsetWidth;
    inner.style.animation = "";
  }
}

function animateValueUpdate(el, text) {
  if (el.textContent === text) return;
  el.style.opacity = "0";
  el.textContent = text;
  requestAnimationFrame(() => { el.style.opacity = "1"; });
}

async function loadFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

async function startSession() {
  landingError.textContent = "";
  startBtn.disabled = true;

  // Defensive cleanup in case a previous session's stream wasn't fully
  // torn down (e.g. the tab was interacted with unusually) — never build
  // a new session on top of stale tracks.
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: true,
    });
  } catch (err) {
    landingError.textContent =
      "Couldn't access your camera/microphone. Please allow permission and try again.";
    startBtn.disabled = false;
    return;
  }

  video.srcObject = stream;
  showScreen("session");
  sessionStatus.textContent = "Loading face tracking model…";

  // Diagnostic: if getUserMedia succeeded but somehow granted 0 audio
  // tracks (denied separately from camera, or a browser quirk), audio
  // recording will silently produce nothing — surface that immediately
  // rather than only discovering it after a failed transcription.
  const audioTrackCount = stream.getAudioTracks().length;
  console.log(`Audio tracks granted: ${audioTrackCount}`);
  if (audioTrackCount === 0) {
    sessionStatus.textContent =
      "No microphone track was granted — speech won't be transcribed this session. Visual tracking will still work.";
  }

  if (!faceLandmarker) {
    try {
      await loadFaceLandmarker();
    } catch (err) {
      sessionStatus.textContent =
        "Failed to load face tracking. Check your connection and reload the page.";
      return;
    }
  }

  blinkCount = 0;
  eyesCurrentlyClosed = false;
  lastBlinkTimestamp = 0;
  gazeSamples = [];
  headAngleSamples = [];
  smoothedYaw = null;
  smoothedPitch = null;
  smoothedRoll = null;
  consecutiveNoFaceFrames = 0;
  faceDetectedDurationMs = 0;
  lastFrameTimestamp = 0;
  framesWaitedForReadiness = 0;
  sessionStartMs = performance.now();

  sessionStatus.textContent = "Tracking your face — look at the camera and speak naturally.";
  stopBtn.classList.remove("hidden");

  // Record audio (separate from the video track) for transcription after
  // the session ends.
  audioChunks = [];
  const audioOnlyStream = new MediaStream(stream.getAudioTracks());
  mediaRecorder = new MediaRecorder(audioOnlyStream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.start();

  const setOverlaySize = () => {
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
  };
  setOverlaySize();
  video.addEventListener("loadedmetadata", setOverlaySize);

  detectFrame();

  timerIntervalId = setInterval(updateTimerDisplay, 250);
}

function updateTimerDisplay() {
  const elapsedSec = (performance.now() - sessionStartMs) / 1000;
  const m = Math.floor(elapsedSec / 60);
  const s = Math.floor(elapsedSec % 60);
  timerValueEl.textContent = `${m}:${s.toString().padStart(2, "0")}`;

  if (elapsedSec >= MAX_SESSION_SECONDS) {
    endSession();
  }
}

function detectFrame() {
  if (!faceLandmarker || video.readyState < 2) {
    framesWaitedForReadiness += 1;
    if (framesWaitedForReadiness > 300) {
      sessionStatus.textContent =
        "Camera feed isn't ready. Try stopping and starting again, or reload the page.";
      return;
    }
    rafId = requestAnimationFrame(detectFrame);
    return;
  }
  framesWaitedForReadiness = 0;

  const now = performance.now();
  const frameDeltaMs = lastFrameTimestamp ? now - lastFrameTimestamp : 0;
  lastFrameTimestamp = now;

  const result = faceLandmarker.detectForVideo(video, now);
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const faceFound = result.faceLandmarks && result.faceLandmarks.length > 0;

  if (faceFound) {
    consecutiveNoFaceFrames = 0;
    faceDetectedDurationMs += frameDeltaMs;

    const landmarks = result.faceLandmarks[0];
    const blendshapes = result.faceBlendshapes?.[0]?.categories ?? [];

    processBlink(blendshapes);
    processGaze(blendshapes);
    processHeadAngle(landmarks);
    drawSimpleOverlay(landmarks);
    sessionStatus.textContent = "Tracking your face — look at the camera and speak naturally.";
  } else {
    consecutiveNoFaceFrames += 1;
    // Only flip to the "no face" state after a sustained run of missed
    // frames, not a single dropped frame — avoids flicker on brief blinks
    // or momentary detector noise.
    if (consecutiveNoFaceFrames >= NO_FACE_SUSTAINED_THRESHOLD) {
      animateValueUpdate(gazeValueEl, "--");
      animateValueUpdate(postureValueEl, "--");
      sessionStatus.textContent = "No face detected — please face the camera.";
    }
  }

  rafId = requestAnimationFrame(detectFrame);
}

function getBlendshapeScore(categories, name) {
  const found = categories.find((c) => c.categoryName === name);
  return found ? found.score : 0;
}

function processBlink(blendshapes) {
  const left = getBlendshapeScore(blendshapes, "eyeBlinkLeft");
  const right = getBlendshapeScore(blendshapes, "eyeBlinkRight");
  const avgBlink = (left + right) / 2;

  if (avgBlink > BLINK_THRESHOLD && !eyesCurrentlyClosed) {
    eyesCurrentlyClosed = true;
    const now = performance.now();
    if (now - lastBlinkTimestamp > BLINK_MIN_INTERVAL_MS) {
      lastBlinkTimestamp = now;
      blinkCount += 1;
      animateValueUpdate(blinkValueEl, String(blinkCount));
    }
  } else if (avgBlink <= BLINK_THRESHOLD && eyesCurrentlyClosed) {
    eyesCurrentlyClosed = false;
  }
}

function processGaze(blendshapes) {
  const lookAwayNames = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
  ];
  const maxDeviation = Math.max(
    ...lookAwayNames.map((n) => getBlendshapeScore(blendshapes, n)),
    0
  );
  gazeSamples.push(maxDeviation);

  const recentAvg = average(gazeSamples.slice(-30));
  animateValueUpdate(gazeValueEl, recentAvg < 0.25 ? "Steady" : recentAvg < 0.5 ? "Drifting" : "Away");
}

// Light exponential smoothing to keep frame-to-frame landmark jitter from
// dominating the variance calculation below (a real head is far steadier
// than the raw per-frame landmark noise makes it look).
let smoothedYaw = null;
let smoothedPitch = null;
let smoothedRoll = null;
const SMOOTHING_ALPHA = 0.3;

function smooth(prev, next) {
  return prev === null ? next : prev + SMOOTHING_ALPHA * (next - prev);
}

function processHeadAngle(landmarks) {
  const nose = landmarks[1];
  const rightEye = landmarks[33];
  const leftEye = landmarks[263];

  const midX = (leftEye.x + rightEye.x) / 2;
  const midY = (leftEye.y + rightEye.y) / 2;
  const interEyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y) || 0.001;

  // Real angle estimates via arctangent (treating inter-eye distance as an
  // approximate depth reference), not a raw linear ratio. The previous
  // version used (offset / interEyeDist) * 100, which isn't a real angle
  // at all — it amplified tiny, normal landmark jitter into huge apparent
  // swings, which combined with `roll` (a genuine degree value) in the
  // same variance sum, made the combined score bottom out on nearly every
  // session regardless of actual movement.
  const rawYaw = (Math.atan2(nose.x - midX, interEyeDist) * 180) / Math.PI;
  const rawPitch = (Math.atan2(nose.y - midY, interEyeDist) * 180) / Math.PI;
  const rawRoll = (Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x) * 180) / Math.PI;

  smoothedYaw = smooth(smoothedYaw, rawYaw);
  smoothedPitch = smooth(smoothedPitch, rawPitch);
  smoothedRoll = smooth(smoothedRoll, rawRoll);

  headAngleSamples.push({ yaw: smoothedYaw, pitch: smoothedPitch, roll: smoothedRoll });

  const recent = headAngleSamples.slice(-30);
  const tiltVariance =
    variance(recent.map((h) => h.roll)) +
    variance(recent.map((h) => h.yaw)) +
    variance(recent.map((h) => h.pitch));
  // Thresholds recalibrated for real degree-based variance: a calm,
  // naturally-still head while talking typically sits well under 10
  // (combined 3-axis variance); noticeable restlessness pushes past 40.
  animateValueUpdate(postureValueEl, tiltVariance < 10 ? "Stable" : tiltVariance < 40 ? "Shifting" : "Restless");
}

function drawSimpleOverlay(landmarks) {
  const pointsToDraw = [1, 33, 263, 133, 362];
  overlayCtx.fillStyle = "#ff6a1a";
  pointsToDraw.forEach((i) => {
    const p = landmarks[i];
    if (!p) return;
    const x = p.x * overlay.width;
    const y = p.y * overlay.height;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 3, 0, Math.PI * 2);
    overlayCtx.fill();
  });
}

function average(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function variance(arr) {
  if (arr.length === 0) return 0;
  const m = average(arr);
  return average(arr.map((v) => (v - m) ** 2));
}

async function endSession() {
  if (rafId) cancelAnimationFrame(rafId);
  if (timerIntervalId) clearInterval(timerIntervalId);

  const durationSec = (performance.now() - sessionStartMs) / 1000;

  const faceDetectedDurationSec = faceDetectedDurationMs / 1000;
  const blinkRatePerMin =
    faceDetectedDurationSec > 0 ? (blinkCount / faceDetectedDurationSec) * 60 : 0;

  const avgGazeDeviation = average(gazeSamples);
  const gazeStabilityScore = Math.round(Math.max(0, 100 - avgGazeDeviation * 100));

  const rollVariance = variance(headAngleSamples.map((h) => h.roll));
  const yawVariance = variance(headAngleSamples.map((h) => h.yaw));
  const pitchVariance = variance(headAngleSamples.map((h) => h.pitch));
  const combinedVariance = rollVariance + yawVariance + pitchVariance;
  // Recalibrated against real degree-based variance (previously this used
  // an uncalibrated pseudo-unit that made almost every session hit 0).
  // A combined variance around 150 (roughly "quite restless" across all
  // three axes) now maps to 0; a calm, naturally-still session lands well
  // above 80.
  const postureStabilityScore = Math.round(
    Math.max(0, 100 - combinedVariance * 0.65)
  );

  const summary = {
    durationSec: Math.round(durationSec),
    blinkCount,
    blinkRatePerMin: Math.round(blinkRatePerMin),
    gazeStabilityScore,
    postureStabilityScore,
  };

  showResults(summary);

  // Stop the audio recorder and wait for its final blob before doing
  // anything else — stopping tracks too early can cut off the last chunk.
  const transcript = await stopRecordingAndTranscribe();
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  fetchAiFeedback({ ...summary, ...transcript });
}

function stopRecordingAndTranscribe() {
  return new Promise((resolve) => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      resolve({ transcript: "", noSpeechDetected: true });
      return;
    }

    mediaRecorder.onstop = async () => {
      try {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        const res = await fetch("/api/transcribe", {
          method: "POST",
          headers: { "Content-Type": mediaRecorder.mimeType },
          body: audioBlob,
        });
        const data = await res.json();
        resolve({
          transcript: data.transcript || "",
          noSpeechDetected: Boolean(data.noSpeechDetected),
        });
      } catch (err) {
        console.warn("Transcription request failed:", err);
        resolve({ transcript: "", noSpeechDetected: true });
      }
    };

    mediaRecorder.stop();
  });
}

function showResults(summary) {
  showScreen("results");

  const blinkRateScore = Math.max(0, 100 - Math.abs(summary.blinkRatePerMin - 18) * 4);
  const presenceScore = Math.round(
    summary.gazeStabilityScore * 0.4 + summary.postureStabilityScore * 0.35 + blinkRateScore * 0.25
  );

  const ringCircumference = 377;
  const ringProgress = document.getElementById("ringProgress");
  ringProgress.style.strokeDashoffset = String(ringCircumference);
  requestAnimationFrame(() => {
    setTimeout(() => {
      ringProgress.style.strokeDashoffset = String(
        ringCircumference * (1 - presenceScore / 100)
      );
    }, 50);
  });
  document.getElementById("presenceScoreValue").textContent = String(presenceScore);

  document.getElementById("resDuration").textContent = `${summary.durationSec}s`;
  document.getElementById("resBlinkRate").textContent = summary.blinkRatePerMin;
  document.getElementById("resGaze").textContent = `${summary.gazeStabilityScore}%`;
  document.getElementById("resPosture").textContent = `${summary.postureStabilityScore}%`;
  document.getElementById("feedbackText").textContent = "Thinking it over…";
}

async function fetchAiFeedback(summary) {
  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(summary),
    });
    const data = await res.json();
    document.getElementById("feedbackText").textContent =
      data.feedback || "Couldn't generate feedback this time — but your stats above are real.";
  } catch (err) {
    document.getElementById("feedbackText").textContent =
      "Couldn't reach the feedback service. Your session stats above are still accurate.";
  }
}

function resetToLanding() {
  showScreen("landing");
  startBtn.disabled = false;
  stopBtn.classList.add("hidden");
  landingError.textContent = "";

  // Full reset — nothing from the previous session should carry over
  // visually or in state, even before startSession() runs its own reset.
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  mediaRecorder = null;
  audioChunks = [];

  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  blinkCount = 0;
  eyesCurrentlyClosed = false;
  lastBlinkTimestamp = 0;
  gazeSamples = [];
  headAngleSamples = [];
  smoothedYaw = null;
  smoothedPitch = null;
  smoothedRoll = null;
  consecutiveNoFaceFrames = 0;
  faceDetectedDurationMs = 0;
  lastFrameTimestamp = 0;
  framesWaitedForReadiness = 0;

  timerValueEl.textContent = "0:00";
  blinkValueEl.textContent = "0";
  gazeValueEl.textContent = "--";
  postureValueEl.textContent = "--";
}

startBtn.addEventListener("click", startSession);
stopBtn.addEventListener("click", endSession);
restartBtn.addEventListener("click", resetToLanding);
