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
  let noise = 0.02; // running typical frame-to-frame wobble of the open-eye score; high = dim light / bad angle
  let prev = null;

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
        // Noise = frame-to-frame wobble, NOT distance from the baseline. (Measuring distance from the baseline
        // made the "noise" explode every time the open-eye level shifted, e.g. when the head went down, which
        // then demanded a huge jump to count a blink and froze the counter.)
        if (prev !== null) {
          const step = Math.abs(score - prev);
          if (step < 0.25) noise += (step - noise) * 0.03; // big jumps are blinks / glitches, not wobble
        }
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
  const rawUpdate = update;
  function trackedUpdate(score, nowMs) { const r = rawUpdate(score, nowMs); prev = score; return r; }

  return { update: trackedUpdate, get baseline() { return baseline; }, get longClosures() { return longClosures; }, get noise() { return noise; }, get inBlink() { return inBlink; } };
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

// ---- Scale-free dip detector (works when the eyes sit half-closed) ----------
// The relative detectors above need a fixed-size jump. With the head down and the eyes looking
// at a phone, the eyelids stay half-closed, the open-eye level collapses, and a blink only adds a
// small absolute step, so fixed jumps never fire. This one compares each value to the person's OWN
// recent open level (a high percentile of the last few seconds): a blink is a quick dip to a fraction
// of that level and back, whatever the level is. Used on the eyelid-shape ratio (EAR).
export const DIP_DEFAULTS = {
  windowMs: 2500,       // how far back "my open level" looks
  pct: 0.8,             // open level = this percentile of the window (robust to blinks taking a minority of frames)
  enterRatio: 0.62,     // a dip starts when the value falls below this fraction of the open level
  exitRatio: 0.8,       // ...and ends when it climbs back above this fraction
  maxMs: 1000,          // a dip longer than this is a long closure / posture change, not a blink
  refractoryMs: 200,    // two blinks closer than this are one
  minWindowFrames: 12,  // frames needed before judging
  minOpen: 0.03,        // below this open level the signal is too small to judge
};

export function makeRatioDipDetector(opts = {}) {
  const c = { ...DIP_DEFAULTS, ...opts };
  let buf = [];
  let inDip = false;
  let start = 0;
  let last = -1e9;
  let longClosures = 0;
  const level = () => {
    const s = buf.map((b) => b.v).sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(s.length * c.pct))];
  };
  return {
    // returns { blink, longClosure }
    update(v, nowMs) {
      const out = { blink: false, longClosure: false };
      if (v === null || v === undefined || !Number.isFinite(v)) return out;
      buf.push({ t: nowMs, v });
      while (buf.length && nowMs - buf[0].t > c.windowMs) buf.shift();
      if (buf.length < c.minWindowFrames) return out;
      const open = level();
      if (open < c.minOpen) return out;
      const r = v / open;
      if (!inDip) {
        if (r < c.enterRatio) { inDip = true; start = nowMs; }
      } else {
        const dur = nowMs - start;
        if (r >= c.exitRatio) {
          inDip = false;
          if (dur <= c.maxMs && nowMs - last >= c.refractoryMs) { last = nowMs; out.blink = true; }
        } else if (dur > c.maxMs) {
          // stuck low: the level has shifted (head down / eyes looking down). Re-learn from the latest frames.
          inDip = false;
          longClosures += 1;
          out.longClosure = true;
          buf = buf.slice(-4);
        }
      }
      return out;
    },
    get longClosures() { return longClosures; },
    get open() { return buf.length ? level() : null; },
  };
}

// Merge blink events from several detectors into one count: events closer than mergeMs are one blink.
export function makeBlinkFuser(mergeMs = 450) {
  let last = -1e9;
  return {
    // call when ANY detector reports a blink; returns true if it is a new blink
    add(nowMs) {
      if (nowMs - last < mergeMs) return false;
      last = nowMs;
      return true;
    },
  };
}
