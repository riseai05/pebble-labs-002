// Gaze tracking relative to the person's own normal eye position. Pure logic, no DOM, testable in Node.
//
// Why this replaces the old gaze score:
//  * The old score was the MAX of eight eye-direction blendshapes, in absolute terms. Reading a passage or
//    looking at a phone held below the camera is eye movement, so ordinary use scored "Away".
//  * MediaPipe's look-down blendshapes rise when the eyelids close (corr 0.84-0.87 with the blink score on
//    real sessions; 64-78% of "looking down" frames were closing lids). Every blink counted as looking away.
//
// What this does instead:
//  1. Turns the blendshapes into a signed gaze position (h = left/right, v = up/down).
//  2. Ignores frames where the eyelids are closing/closed or lowered (the eye can't be read then).
//  3. Smooths "where the eyes are" over about a second, so line-by-line reading sweeps don't count.
//  4. Judges distance from THIS person's own typical position (the median over the session so far), so camera
//     angle, phone height and how the person normally looks at the screen drop out.
//  5. A state only changes after it has lasted a moment, so a quick glance is not "Away".

const median = (arr) => {
  if (!arr.length) return null;
  const s = arr.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export const GAZE_DEFAULTS = {
  eyesOpenMaxBlink: 0.45, // eyelid score above this = closing / closed / lowered: gaze can't be read
  smoothMs: 1000,         // median window for "where the eyes are"
  calibMs: 15000,         // learn the person's normal position for this long before judging
  minCalibFrames: 40,     // ...and from at least this many readable frames
  driftDist: 0.22,        // distance from own normal position that counts as drifting
  awayDist: 0.42,         // ...and as away
  holdMs: 1500,           // a worse state must last this long before it is shown
  unreadableMs: 3000,     // eyes unreadable this long: show "--" instead of holding the last state
  refreshMs: 1000,        // how often the live reference position is re-estimated
};

// MediaPipe / ARKit blendshape names. "Left"/"Right" are the subject's eyes. Looking to the subject's right makes
// the left eye look "in" (toward the nose) and the right eye look "out"; looking left is the opposite.
export function gazeFromBlendshapes(get) {
  const h = (get("eyeLookInLeft") + get("eyeLookOutRight") - get("eyeLookOutLeft") - get("eyeLookInRight")) / 2;
  const v = (get("eyeLookUpLeft") + get("eyeLookUpRight") - get("eyeLookDownLeft") - get("eyeLookDownRight")) / 2;
  const blink = (get("eyeBlinkLeft") + get("eyeBlinkRight")) / 2;
  return { h, v, blink };
}

export function makeGazeTracker(opts = {}) {
  const c = { ...GAZE_DEFAULTS, ...opts };
  const win = [];          // recent readable {t,h,v} for smoothing
  const frames = [];       // every update: {t, readable, sh, sv}  (sh/sv = smoothed position; null if unreadable)
  let firstReadableT = null;
  let readableCount = 0;
  let ref = null;          // live reference {h, v}
  let lastRefT = -1e9;
  let sinceDrift = null;
  let sinceAway = null;
  let lastState = "calibrating";
  let unreadableSince = null;

  function refreshRef(t) {
    const hs = [], vs = [];
    for (const f of frames) if (f.sh !== null) { hs.push(f.sh); vs.push(f.sv); }
    if (hs.length >= c.minCalibFrames) ref = { h: median(hs), v: median(vs) };
    lastRefT = t;
  }

  // t = ms since the session started. Returns { state, d, readable }.
  // state: "calibrating" | "steady" | "drifting" | "away" | "unreadable"
  function update({ t, h, v, blink }) {
    const readable = Number.isFinite(h) && Number.isFinite(v) && Number.isFinite(blink) && blink <= c.eyesOpenMaxBlink;
    if (!readable) {
      frames.push({ t, readable: false, sh: null, sv: null });
      if (unreadableSince === null) unreadableSince = t;
      // pause the persistence timers: a blink in the middle of a glance must not cancel or complete it
      if (t - unreadableSince > c.unreadableMs) lastState = ref ? "unreadable" : "calibrating";
      return { state: lastState, d: null, readable: false };
    }
    unreadableSince = null;
    readableCount += 1;
    if (firstReadableT === null) firstReadableT = t;
    win.push({ t, h, v });
    while (win.length && t - win[0].t > c.smoothMs) win.shift();
    const sh = median(win.map((w) => w.h));
    const sv = median(win.map((w) => w.v));
    frames.push({ t, readable: true, sh, sv });

    if (t - lastRefT >= c.refreshMs) refreshRef(t);
    if (!ref || t - firstReadableT < c.calibMs) {
      lastState = "calibrating";
      return { state: lastState, d: null, readable: true };
    }
    const d = Math.hypot(sh - ref.h, sv - ref.v);
    if (d >= c.awayDist) { if (sinceAway === null) sinceAway = t; } else sinceAway = null;
    if (d >= c.driftDist) { if (sinceDrift === null) sinceDrift = t; } else sinceDrift = null;
    if (sinceAway !== null && t - sinceAway >= c.holdMs) lastState = "away";
    else if (sinceDrift !== null && t - sinceDrift >= c.holdMs) lastState = "drifting";
    else lastState = "steady";
    return { state: lastState, d, readable: true };
  }

  // Post-hoc pass with the final reference (the median over the whole session), so early frames are judged
  // against the person's true normal, not against a half-learned one. bucketMs groups results for the timeline.
  function finalize(bucketMs = 10000) {
    const hs = [], vs = [];
    for (const f of frames) if (f.sh !== null) { hs.push(f.sh); vs.push(f.sv); }
    const total = frames.length;
    const out = { method: "relative-v2", frames: total, readable: hs.length, readableShare: total ? hs.length / total : null,
      ref: null, score: null, driftShare: null, awayShare: null, meanDist: null, buckets: [] };
    if (hs.length < c.minCalibFrames) return out;
    const r = { h: median(hs), v: median(vs) };
    out.ref = { h: Math.round(r.h * 1000) / 1000, v: Math.round(r.v * 1000) / 1000 };
    let on = 0, drift = 0, away = 0, dSum = 0;
    for (const f of frames) {
      const bi = Math.max(0, Math.floor(f.t / bucketMs));
      const b = out.buckets[bi] || (out.buckets[bi] = { total: 0, readable: 0, off: 0, away: 0, dSum: 0 });
      b.total += 1;
      if (f.sh === null) continue;
      const d = Math.hypot(f.sh - r.h, f.sv - r.v);
      b.readable += 1; b.dSum += d; dSum += d;
      if (d < c.driftDist) on += 1; else { b.off += 1; drift += 1; }
      if (d >= c.awayDist) { b.away += 1; away += 1; }
    }
    out.score = Math.round((on / hs.length) * 100);
    out.driftShare = Math.round((drift / hs.length) * 100) / 100;
    out.awayShare = Math.round((away / hs.length) * 100) / 100;
    out.meanDist = Math.round((dSum / hs.length) * 1000) / 1000;
    return out;
  }

  return { update, finalize, get state() { return lastState; }, get ref() { return ref; }, get readableCount() { return readableCount; } };
}
