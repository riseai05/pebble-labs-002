// Per-person relative blink detector.
//
// Why: the eyelid score a face model reports for "open eyes" is not constant. It
// rises when someone looks down (notes, phone) and gets noisier in dim light or at
// odd angles. A fixed threshold then gets stuck "closed" or flickers. This detector
// learns each person's own open-eye level continuously and counts a blink only as a
// quick dip above that level.
//
// Pure logic, no DOM — so it can be tested in Node with synthetic signals.

export const DEFAULTS = {
  riseDelta: 0.18,      // blink starts when score > baseline + riseDelta
  fallDelta: 0.09,      // blink ends when score < baseline + fallDelta
  minPeak: 0.28,        // ...and the score itself must reach at least this
  minIntervalMs: 250,   // two blinks closer than this are one
  maxBlinkMs: 700,      // a lid-drop longer than this is a long closure / look-down shift, not a blink
  baselineDown: 0.10,   // baseline follows a lower score fairly quickly (eyes opened wider)
  baselineUp: 0.015,    // ...and a higher score slowly (looking down raises the open-eye level)
  warmupFrames: 20,     // frames used to seed the baseline
};

export function makeBlinkDetector(opts = {}) {
  const c = { ...DEFAULTS, ...opts };
  let baseline = null;
  let warm = [];
  let inBlink = false;
  let blinkStartMs = 0;
  let peak = 0;
  let lastBlinkMs = -1e9;
  let longClosures = 0;
  let noise = 0.02; // running typical wobble of the open-eye score; high = dim light / bad angle

  // Returns { blink: boolean, longClosure: boolean, baseline }
  function update(score, nowMs) {
    const out = { blink: false, longClosure: false, baseline };

    if (baseline === null) {
      warm.push(score);
      if (warm.length >= c.warmupFrames) {
        const s = warm.slice().sort((a, b) => a - b);
        baseline = s[Math.floor(s.length * 0.3)]; // low-ish percentile = open eyes even if a blink was in the warm-up
        warm = [];
      }
      out.baseline = baseline;
      return out;
    }

    if (!inBlink) {
      const rise = Math.max(c.riseDelta, 5 * noise); // noisy signal => need a bigger jump to count
      if (score > baseline + rise && score >= c.minPeak) {
        inBlink = true;
        blinkStartMs = nowMs;
        peak = score;
      } else {
        noise += (Math.abs(score - baseline) - noise) * 0.03;
        const k = score < baseline ? c.baselineDown : c.baselineUp;
        baseline += (score - baseline) * k;
      }
    } else {
      if (score > peak) peak = score;
      const dur = nowMs - blinkStartMs;
      if (score < baseline + c.fallDelta) {
        inBlink = false;
        if (dur <= c.maxBlinkMs) {
          if (nowMs - lastBlinkMs >= c.minIntervalMs) {
            lastBlinkMs = nowMs;
            out.blink = true;
          }
        } else {
          longClosures += 1;
          out.longClosure = true;
        }
      } else if (dur > c.maxBlinkMs) {
        // Held "closed" too long: the eyelid level has shifted (looking down) or the eyes are shut.
        // Re-learn the baseline from here instead of staying stuck, and don't count a blink.
        inBlink = false;
        baseline = score;
        longClosures += 1;
        out.longClosure = true;
      }
    }
    out.baseline = baseline;
    return out;
  }

  return { update, get baseline() { return baseline; }, get longClosures() { return longClosures; }, get noise() { return noise; }, get inBlink() { return inBlink; } };
}

// ---- Eyelid-shape signal -----------------------------------------------------
// A second, independent way to see a blink: measure the gap between the upper and
// lower eyelid points on the face mesh, relative to the eye's width (the classic
// "eye aspect ratio"). Turned into a 0..1 "closure" score relative to this person's
// own open-eye level, so it can feed the same relative detector.
const EYE_RIGHT = { outer: 33, inner: 133, pairs: [[160, 144], [159, 145], [158, 153]] };
const EYE_LEFT = { outer: 263, inner: 362, pairs: [[385, 380], [386, 374], [387, 373]] };

function eyeRatio(lm, eye, w, h) {
  const d = (a, b) => Math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h);
  const width = d(eye.outer, eye.inner);
  if (!width) return null;
  const v = eye.pairs.reduce((s, [a, b]) => s + d(a, b), 0) / eye.pairs.length;
  return v / width;
}

export function eyeAspectRatio(lm, w, h) {
  if (!lm || lm.length < 400) return null;
  const r = eyeRatio(lm, EYE_RIGHT, w, h);
  const l = eyeRatio(lm, EYE_LEFT, w, h);
  if (r === null || l === null) return null;
  return (r + l) / 2;
}

export function makeEarClosure() {
  let open = null;
  return {
    // returns closure 0 (fully open for this person) .. 1 (closed)
    update(ear) {
      if (ear === null || ear === undefined) return null;
      if (open === null) open = ear;
      if (ear > open) open += (ear - open) * 0.2;       // wider than thought: learn quickly
      else open -= (open - ear) * 0.004;                 // narrower (looking down): drift slowly
      return Math.max(0, Math.min(1, 1 - ear / open));
    },
    get open() { return open; },
  };
}
