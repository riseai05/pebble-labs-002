// Pebble Labs — Experiment 002: focus/attention tracking during reading or work.
// MediaPipe runs entirely in-browser via WASM, loaded from CDN below.
// No video ever leaves the browser. Visible metrics (blink/gaze/posture) plus
// self-reports, a device-type tag, and interaction/distraction event timestamps
// are sent to /api/log-session at the end of a session. No video or audio is
// ever transmitted, and no keystroke content or mouse position is ever logged —
// only the fact and timing of blur/visibility/idle/trigger events.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const READ_SESSION_SECONDS = 300; // 5 minutes (the passage takes ~3.5)
const WORK_SESSION_SECONDS = 600; // 10 minutes
function maxSessionSec() {
  return selectedTaskType === "read" ? READ_SESSION_SECONDS : WORK_SESSION_SECONDS;
}
const BLINK_THRESHOLD = 0.5;
const BLINK_MIN_INTERVAL_MS = 350;
const BLINK_REOPEN_THRESHOLD = 0.35; // eyes must reopen past this before the next blink counts (stops flicker around 0.5, e.g. when looking down at a phone)
const NO_FACE_SUSTAINED_THRESHOLD = 8; // consecutive frames before we call it sustained, not a blip
const SELF_REPORT_EVERY_SEC = 150; // every 2.5 min
const IDLE_THRESHOLD_MS = 15000; // only tracked in "work" mode

const screens = {
  landing: document.getElementById("landing"),
  session: document.getElementById("session"),
  results: document.getElementById("results"),
};
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const restartBtn = document.getElementById("restartBtn");
const landingError = document.getElementById("landingError");
const consentCheck = document.getElementById("consentCheck");
const exitSurvey = document.getElementById("exitSurvey");
const postSurveyThanks = document.getElementById("postSurveyThanks");
const submitSurveyBtn = document.getElementById("submitSurveyBtn");
const surpriseInput = document.getElementById("surpriseInput");
const sessionStatus = document.getElementById("sessionStatus");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const timerValueEl = document.getElementById("timerValue");
const blinkValueEl = document.getElementById("blinkValue");
const gazeValueEl = document.getElementById("gazeValue");
const postureValueEl = document.getElementById("postureValue");

const selfReportOverlay = document.getElementById("selfReportOverlay");
const selfReportWhy = document.getElementById("selfReportWhy");
const distractionBanner = document.getElementById("distractionBanner");
const distractionCheckOverlay = document.getElementById("distractionCheckOverlay");
const sessionReadingPanel = document.getElementById("sessionReadingPanel");

let faceLandmarker = null;
let stream = null;
let rafId = null;
let sessionStartMs = 0;
let timerIntervalId = null;
let selfReportIntervalId = null;
let idleTimeoutId = null;
let distractionTriggerTimeoutIds = [];
let distractionFollowupTimeoutIds = [];
let bannerHideTimeoutId = null;

let blinkCount = 0;
let eyesCurrentlyClosed = false;
let lastBlinkTimestamp = 0;
let gazeSamples = [];
let headAngleSamples = [];
let timelineBuckets = []; // 10-second chunks of raw sensor data, saved as a per-session timeline
let consecutiveNoFaceFrames = 0;
let faceDetectedDurationMs = 0; // only time a real face was actually seen
let lastFrameTimestamp = 0;
let framesWaitedForReadiness = 0;
let selfReports = []; // { atSec, response, why }
let distractionEvents = []; // { type, atSec }
let sessionNickname = ""; // optional, saved to Supabase only — never shown back to the user
let selectedTaskType = "read"; // "read" or "work"
let lastSessionSummary = null; // held until exit survey is submitted
let isIdle = false;
let pendingSelfReportAtSec = null;
let pendingSelfReportValue = null;

// --- Device detection (user agent + screen/touch only, nothing invasive) ---
function detectDeviceType() {
  const ua = navigator.userAgent || "";
  const isTouch = (navigator.maxTouchPoints || 0) > 0 || "ontouchstart" in window;
  const isMobileUA = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
  const isSmallScreen = Math.min(window.innerWidth, window.innerHeight) < 500;
  return isMobileUA || (isTouch && isSmallScreen) ? "mobile" : "desktop";
}
const deviceType = detectDeviceType();

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

let modelLoadPromise = null;
let modelLoadMs = null;
let modelDelegate = null;
let framesProcessed = 0; // frames actually run through the face model this session
let warmUpInfo = null;

async function loadFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const build = (delegate) =>
    FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate,
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  try {
    faceLandmarker = await build("GPU");
    modelDelegate = "GPU";
  } catch (gpuErr) {
    // some phones can't start the GPU path — slower CPU is better than failing outright
    faceLandmarker = await build("CPU");
    modelDelegate = "CPU";
  }
}

// Start downloading/initializing the model as soon as the page opens, so it is
// already ready by the time the person has read the consent screen and tapped Start.
function ensureModel() {
  if (!modelLoadPromise) {
    const t0 = performance.now();
    modelLoadPromise = loadFaceLandmarker()
      .then(() => { modelLoadMs = Math.round(performance.now() - t0); })
      .catch((err) => { modelLoadPromise = null; throw err; });
  }
  return modelLoadPromise;
}
ensureModel().catch(() => { /* retried when Start is tapped */ });

// The first seconds of tracking are slow (graphics pipeline compiles on the first
// frames), which used to land inside the measured session. Run the model on the
// live camera BEFORE the session clock starts, until it is flowing, then begin.
async function warmUpTracking(maxMs = 8000, wantFaceFrames = 15) {
  const t0 = performance.now();
  let frames = 0;
  let faceFrames = 0;
  while (performance.now() - t0 < maxMs && faceFrames < wantFaceFrames) {
    await new Promise((r) => requestAnimationFrame(r));
    if (video.readyState < 2) continue;
    try {
      const res = faceLandmarker.detectForVideo(video, performance.now());
      frames += 1;
      if (res.faceLandmarks && res.faceLandmarks.length > 0) faceFrames += 1;
    } catch (e) {
      break;
    }
  }
  return { warmUpMs: Math.round(performance.now() - t0), warmUpFrames: frames, warmUpFaceFrames: faceFrames };
}

function handleStartClick() {
  landingError.textContent = "";
  primeAudio();

  if (!consentCheck.checked) {
    landingError.textContent = "Please confirm you're 18+ and okay with the webcam use before starting.";
    return;
  }

  const nicknameEl = document.getElementById("nicknameInput");
  sessionNickname = nicknameEl ? nicknameEl.value.trim().slice(0, 30) : "";

  const taskInput = document.querySelector('input[name="taskType"]:checked');
  selectedTaskType = taskInput ? taskInput.value : "read";

  startSession();
}

async function startSession() {
  landingError.textContent = "";
  startBtn.disabled = true;

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 480 }, height: { ideal: 360 }, frameRate: { ideal: 30 }, facingMode: "user" },
    });
  } catch (err) {
    landingError.textContent =
      "Couldn't access your camera. Please allow permission and try again.";
    startBtn.disabled = false;
    return;
  }

  video.srcObject = stream;
  showScreen("session");
  sessionReadingPanel.classList.toggle("hidden", selectedTaskType !== "read");
  sessionStatus.textContent = "Loading face tracking model…";

  try {
    await ensureModel();
  } catch (err) {
    sessionStatus.textContent =
      "Failed to load face tracking. Check your connection and reload the page.";
    startBtn.disabled = false;
    return;
  }

  sessionStatus.textContent = "Getting ready — look at the screen for a moment…";
  warmUpInfo = await warmUpTracking();

  blinkCount = 0;
  eyesCurrentlyClosed = false;
  lastBlinkTimestamp = 0;
  gazeSamples = [];
  headAngleSamples = [];
  timelineBuckets = [];
  smoothedYaw = null;
  smoothedPitch = null;
  smoothedRoll = null;
  consecutiveNoFaceFrames = 0;
  faceDetectedDurationMs = 0;
  lastFrameTimestamp = 0;
  framesWaitedForReadiness = 0;
  selfReports = [];
  distractionEvents = [];
  isIdle = false;
  framesProcessed = 0;
  sessionStartMs = performance.now();

  sessionStatus.textContent = "Tracking — go ahead and read or work as you normally would.";
  stopBtn.classList.remove("hidden");

  const setOverlaySize = () => {
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
  };
  setOverlaySize();
  video.addEventListener("loadedmetadata", setOverlaySize);

  detectFrame();

  timerIntervalId = setInterval(updateTimerDisplay, 250);
  selfReportIntervalId = setInterval(promptSelfReport, SELF_REPORT_EVERY_SEC * 1000);

  attachDistractionListeners();
  scheduleDistractionTriggers();
}

// --- Invisible distraction tracking (never shown in UI) ---
function logDistractionEvent(type) {
  const atSec = Math.round((performance.now() - sessionStartMs) / 1000);
  distractionEvents.push({ type, atSec });
}

function handleWindowBlur() { logDistractionEvent("window_blur"); }
function handleWindowFocus() { logDistractionEvent("window_focus"); }
function handleVisibilityChange() {
  logDistractionEvent(document.visibilityState === "hidden" ? "tab_hidden" : "tab_visible");
}

function resetIdleTimer() {
  if (selectedTaskType !== "work") return; // a focused reader can legitimately be still — skip idle tracking in read mode
  if (isIdle) {
    logDistractionEvent("idle_end");
    isIdle = false;
  }
  clearTimeout(idleTimeoutId);
  idleTimeoutId = setTimeout(() => {
    isIdle = true;
    logDistractionEvent("idle_start");
  }, IDLE_THRESHOLD_MS);
}

function attachDistractionListeners() {
  window.addEventListener("blur", handleWindowBlur);
  window.addEventListener("focus", handleWindowFocus);
  document.addEventListener("visibilitychange", handleVisibilityChange);
  if (selectedTaskType === "work") {
    window.addEventListener("mousemove", resetIdleTimer);
    window.addEventListener("keydown", resetIdleTimer);
    resetIdleTimer();
  }
}

function detachDistractionListeners() {
  window.removeEventListener("blur", handleWindowBlur);
  window.removeEventListener("focus", handleWindowFocus);
  document.removeEventListener("visibilitychange", handleVisibilityChange);
  window.removeEventListener("mousemove", resetIdleTimer);
  window.removeEventListener("keydown", resetIdleTimer);
  clearTimeout(idleTimeoutId);
}

// --- Controlled distraction trigger: two fixed moments per session (1:09 and
// 3:15) in both read and work mode, so results are comparable across modes.
// Disclosed in general terms on the consent screen but not timed or
// described exactly, so it stays a valid test while nothing is hidden. ---
const DISTRACTION_TRIGGERS = [
  { atSec: 69,  sender: "Ali", text: "Bro are you free tonight?" },       // 1:09
  { atSec: 195, sender: "Mom", text: "Call me when you get a sec" },      // 3:15
];

const DISTRACTION_FOLLOWUP_SEC = 30;
let lastDistractionUiMs = -Infinity; // last time a banner / yes-no question appeared

function scheduleDistractionTriggers() {
  DISTRACTION_TRIGGERS.forEach((trigger) => {
    if (trigger.atSec < maxSessionSec()) {
      const id = setTimeout(() => fireDistractionTrigger(trigger), trigger.atSec * 1000);
      distractionTriggerTimeoutIds.push(id);
    }
  });
}

function fireDistractionTrigger(trigger) {
  logDistractionEvent("triggered_distraction_shown");
  lastDistractionUiMs = performance.now();
  playChime();
  document.getElementById("bannerSender").textContent = trigger.sender;
  document.getElementById("bannerText").textContent = trigger.text;
  distractionBanner.classList.remove("hidden");
  bannerHideTimeoutId = setTimeout(() => distractionBanner.classList.add("hidden"), 4000);

  const followupId = setTimeout(showDistractionCheck, DISTRACTION_FOLLOWUP_SEC * 1000); // fixed delay so every session is measured the same way
  distractionFollowupTimeoutIds.push(followupId);
}

// Soft two-note notification chime, synthesized in the browser (no audio file,
// and deliberately not a copy of any platform's real alert tone). The audio
// context is unlocked on the Start tap so it is allowed to play later.
let audioCtx = null;
function primeAudio() {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    if (!audioCtx) audioCtx = new AC();
    if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  } catch (e) { /* audio is optional */ }
}

async function playChime() {
  try {
    if (!audioCtx) { logDistractionEvent("banner_sound_unavailable"); return; }
    if (audioCtx.state !== "running") {
      await Promise.race([audioCtx.resume().catch(() => {}), new Promise((r) => setTimeout(r, 300))]);
    }
    if (audioCtx.state !== "running") { logDistractionEvent("banner_sound_blocked"); return; }
    const now = audioCtx.currentTime;
    [[1046.5, 0], [1318.5, 0.14]].forEach(([freq, offset]) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, now + offset);
      gain.gain.exponentialRampToValueAtTime(0.28, now + offset + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.7);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(now + offset);
      osc.stop(now + offset + 0.75);
    });
    logDistractionEvent("banner_sound_played");
  } catch (e) {
    logDistractionEvent("banner_sound_unavailable");
  }
}

function showDistractionCheck() {
  lastDistractionUiMs = performance.now();
  distractionCheckOverlay.classList.remove("hidden");
}

function handleDistractionCheckResponse(response) {
  markTap();
  logDistractionEvent(`triggered_distraction_response_${response}`);
  distractionCheckOverlay.classList.add("hidden");
}

function clearDistractionTimers() {
  distractionTriggerTimeoutIds.forEach(clearTimeout);
  distractionFollowupTimeoutIds.forEach(clearTimeout);
  distractionTriggerTimeoutIds = [];
  distractionFollowupTimeoutIds = [];
  clearTimeout(bannerHideTimeoutId);
  distractionBanner.classList.add("hidden");
  distractionCheckOverlay.classList.add("hidden");
}

// --- Self-report: single tap, with an optional "why" for drifting/lost answers ---
function promptSelfReport() {
  const elapsedSec = (performance.now() - sessionStartMs) / 1000;
  if (elapsedSec > maxSessionSec() - 20) return; // not worth asking right at the end
  const busy = !distractionCheckOverlay.classList.contains("hidden") ||
    performance.now() - lastDistractionUiMs < 30000;
  if (busy) { // don't stack on top of the banner / yes-no question; try again shortly
    setTimeout(promptSelfReport, 15000);
    return;
  }
  pendingSelfReportAtSec = Math.round((performance.now() - sessionStartMs) / 1000);
  selfReportWhy.classList.add("hidden");
  selfReportOverlay.classList.remove("hidden");
}

function handleSelfReportChoice(value) {
  markTap();
  pendingSelfReportValue = value;
  if (value === "1") {
    finalizeSelfReport(null);
  } else {
    selfReportWhy.classList.remove("hidden");
  }
}

function finalizeSelfReport(why) {
  markTap();
  selfReports.push({ atSec: pendingSelfReportAtSec, response: pendingSelfReportValue, why: why || null });
  selfReportOverlay.classList.add("hidden");
  selfReportWhy.classList.add("hidden");
}

function updateTimerDisplay() {
  const elapsedSec = (performance.now() - sessionStartMs) / 1000;
  const m = Math.floor(elapsedSec / 60);
  const s = Math.floor(elapsedSec % 60);
  timerValueEl.textContent = `${m}:${s.toString().padStart(2, "0")}`;

  if (elapsedSec >= maxSessionSec()) {
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
  framesProcessed += 1;
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
    sessionStatus.textContent = "Tracking — go ahead and read or work as you normally would.";
  } else {
    consecutiveNoFaceFrames += 1;
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
      currentBucket().blinks += 1;
      animateValueUpdate(blinkValueEl, String(blinkCount));
    }
  } else if (avgBlink <= BLINK_REOPEN_THRESHOLD && eyesCurrentlyClosed) {
    eyesCurrentlyClosed = false;
  }
}

const TIMELINE_BUCKET_SEC = 10;
function currentBucket() {
  const idx = Math.max(0, Math.floor((performance.now() - sessionStartMs) / 1000 / TIMELINE_BUCKET_SEC));
  if (!timelineBuckets[idx]) {
    timelineBuckets[idx] = { gaze: [], blinks: 0, yaw: [], pitch: [], roll: [] };
  }
  return timelineBuckets[idx];
}

// Mark the current 10s chunk as containing a screen tap (self-report / yes-no
// answer). Tapping a phone shakes it, which would otherwise look like fidgeting.
function markTap() {
  if (sessionStartMs) currentBucket().tap = true;
}

const TIMELINE_MIN_FRAMES = 10; // chunks with fewer face frames than this are too thin to trust (e.g. the last partial chunk)
const TIMELINE_BASELINE_SEC = 30; // first 30s = this person's own "normal" gaze, used to remove phone-angle offset

function buildTimeline() {
  const avg = (arr) => (arr.length ? average(arr) : null);
  const round1 = (n) => (n === null ? null : Math.round(n * 10) / 10);

  // Personal baseline: mean raw gaze deviation over the first 30s of face frames.
  // Reading down at a phone looks like "looking away" to the camera, so absolute
  // gaze scores mostly reflect the angle; change from baseline is what matters.
  const baselineFrames = [];
  for (let i = 0; i < Math.ceil(TIMELINE_BASELINE_SEC / TIMELINE_BUCKET_SEC); i++) {
    if (timelineBuckets[i]) baselineFrames.push(...timelineBuckets[i].gaze);
  }
  const baseline = baselineFrames.length >= TIMELINE_MIN_FRAMES ? average(baselineFrames) : null;

  const out = [];
  for (let i = 0; i < timelineBuckets.length; i++) {
    const b = timelineBuckets[i];
    if (!b || b.gaze.length < TIMELINE_MIN_FRAMES) continue; // too few face frames to trust
    const meanDev = avg(b.gaze);
    const row = {
      t: i * TIMELINE_BUCKET_SEC, // chunk start, seconds from session start
      gaze: Math.round(Math.max(0, 100 - meanDev * 100)),
      // gazeRel: points of extra look-away vs. this person's own baseline (positive = looking away more)
      gazeRel: baseline === null ? null : Math.round((meanDev - baseline) * 1000) / 10,
      blinks: b.blinks,
      pitch: round1(avg(b.pitch)),
      fidget: round1(variance(b.roll) + variance(b.yaw) + variance(b.pitch)),
      n: b.gaze.length, // face frames in this chunk
    };
    if (b.tap) row.tap = true; // a screen tap happened here — treat fidget with caution
    out.push(row);
  }
  return out;
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
  currentBucket().gaze.push(maxDeviation);

  const recentAvg = average(gazeSamples.slice(-30));
  animateValueUpdate(gazeValueEl, recentAvg < 0.25 ? "Steady" : recentAvg < 0.5 ? "Drifting" : "Away");
}

// Light exponential smoothing to keep frame-to-frame landmark jitter from
// dominating the variance calculation below.
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

  const rawYaw = (Math.atan2(nose.x - midX, interEyeDist) * 180) / Math.PI;
  const rawPitch = (Math.atan2(nose.y - midY, interEyeDist) * 180) / Math.PI;
  const rawRoll = (Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x) * 180) / Math.PI;

  smoothedYaw = smooth(smoothedYaw, rawYaw);
  smoothedPitch = smooth(smoothedPitch, rawPitch);
  smoothedRoll = smooth(smoothedRoll, rawRoll);

  // rawPitch/smoothedPitch doubles as a forward-lean / head-drop proxy:
  // a sustained positive pitch means the head is tilting down relative to
  // the eye line, which is what forward lean and head-drop both produce.
  // This is not the same as a true body-pose forward-lean angle (that needs
  // shoulder/torso landmarks from a pose model, which isn't in this
  // pipeline) — treat it as a head-only approximation.
  headAngleSamples.push({ yaw: smoothedYaw, pitch: smoothedPitch, roll: smoothedRoll });
  {
    const b = currentBucket();
    b.yaw.push(smoothedYaw);
    b.pitch.push(smoothedPitch);
    b.roll.push(smoothedRoll);
  }

  const recent = headAngleSamples.slice(-30);
  const tiltVariance =
    variance(recent.map((h) => h.roll)) +
    variance(recent.map((h) => h.yaw)) +
    variance(recent.map((h) => h.pitch));
  animateValueUpdate(postureValueEl, tiltVariance < 10 ? "Stable" : tiltVariance < 40 ? "Shifting" : "Restless");
}

function drawSimpleOverlay(landmarks) {
  const pointsToDraw = [1, 33, 263, 133, 362];
  overlayCtx.fillStyle = "#c9a46b";
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
  if (selfReportIntervalId) clearInterval(selfReportIntervalId);
  detachDistractionListeners();
  clearDistractionTimers();
  selfReportOverlay.classList.add("hidden");

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
  const postureStabilityScore = Math.round(Math.max(0, 100 - combinedVariance * 0.65));

  // Structured posture detail (backend-only, never shown in the UI).
  // headDropAngle: average smoothed pitch — a proxy for forward lean/head drop.
  // fidgetScore: same combined variance used above, kept separately so it can
  // be analyzed on its own as a movement-frequency proxy.
  // shoulderAsymmetry is intentionally omitted — it needs body-pose landmarks
  // (shoulders/torso), which this face-only tracker doesn't capture.
  const headDropAngle = Math.round(average(headAngleSamples.map((h) => h.pitch)) * 10) / 10;
  const fidgetScore = Math.round(combinedVariance * 10) / 10;

  const summary = {
    durationSec: Math.round(durationSec),
    blinkCount,
    blinkRatePerMin: Math.round(blinkRatePerMin),
    gazeStabilityScore,
    postureStabilityScore,
    headDropAngle,
    fidgetScore,
    timeline: buildTimeline(),
    perf: {
      modelLoadMs,
      delegate: modelDelegate,
      ...(warmUpInfo || {}),
      avgFps: durationSec > 0 ? Math.round((framesProcessed / durationSec) * 10) / 10 : null,
    },
  };

  lastSessionSummary = summary;
  showResults(summary);

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  // Logging is deferred until the exit survey is submitted (see submitExitSurvey)
  // so the survey answers travel with the same session row instead of a second write.
}

function submitExitSurvey() {
  const accuracyInput = document.querySelector('input[name="accuracyRating"]:checked');
  const wouldUseInput = document.querySelector('input[name="wouldUse"]:checked');

  const exitSurveyAnswers = {
    accuracyRating: accuracyInput ? accuracyInput.value : null,
    surprise: surpriseInput.value.trim(),
    wouldUse: wouldUseInput ? wouldUseInput.value : null,
  };

  logSessionToSupabase(lastSessionSummary, exitSurveyAnswers);

  exitSurvey.classList.add("hidden");
  postSurveyThanks.classList.remove("hidden");
  restartBtn.classList.remove("hidden");
}

async function logSessionToSupabase(summary, exitSurveyAnswers) {
  try {
    await fetch("/api/log-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: crypto.randomUUID(),
        nickname: sessionNickname,
        timestamp: new Date().toISOString(),
        taskType: selectedTaskType,
        deviceType,
        ...summary,
        selfReports,
        distractionEvents,
        exitSurvey: exitSurveyAnswers || null,
      }),
    });
  } catch (err) {
    console.warn("Logging failed (session still shown to user):", err);
  }
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
  // headDropAngle and fidgetScore are logged but intentionally not shown here —
  // only the metrics the tester already saw during the session (blink/gaze/posture)
  // are surfaced at the end, per the "visible metrics only" rule.
}

function resetToLanding() {
  showScreen("landing");
  startBtn.disabled = false;
  stopBtn.classList.add("hidden");
  landingError.textContent = "";
  exitSurvey.classList.remove("hidden");
  postSurveyThanks.classList.add("hidden");
  restartBtn.classList.add("hidden");
  document.querySelectorAll('input[name="accuracyRating"]').forEach((el) => { el.checked = false; });
  document.querySelectorAll('input[name="wouldUse"]').forEach((el) => { el.checked = false; });
  surpriseInput.value = "";

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  blinkCount = 0;
  eyesCurrentlyClosed = false;
  lastBlinkTimestamp = 0;
  gazeSamples = [];
  headAngleSamples = [];
  timelineBuckets = [];
  smoothedYaw = null;
  smoothedPitch = null;
  smoothedRoll = null;
  consecutiveNoFaceFrames = 0;
  faceDetectedDurationMs = 0;
  lastFrameTimestamp = 0;
  framesWaitedForReadiness = 0;
  selfReports = [];
  distractionEvents = [];

  timerValueEl.textContent = "0:00";
  blinkValueEl.textContent = "0";
  gazeValueEl.textContent = "--";
  postureValueEl.textContent = "--";
}

startBtn.addEventListener("click", handleStartClick);
stopBtn.addEventListener("click", endSession);
restartBtn.addEventListener("click", resetToLanding);
submitSurveyBtn.addEventListener("click", submitExitSurvey);

document.querySelectorAll(".self-report-btn[data-value]").forEach((btn) => {
  btn.addEventListener("click", () => handleSelfReportChoice(btn.dataset.value));
});
document.querySelectorAll(".why-tag").forEach((btn) => {
  btn.addEventListener("click", () => finalizeSelfReport(btn.dataset.tag));
});
const skipWhyBtn = document.getElementById("selfReportSkipWhy");
if (skipWhyBtn) skipWhyBtn.addEventListener("click", () => finalizeSelfReport(null));

document.querySelectorAll("[data-distraction-response]").forEach((btn) => {
  btn.addEventListener("click", () => handleDistractionCheckResponse(btn.dataset.distractionResponse));
});
