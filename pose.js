// Real head pose + tracking-quality helpers. Pure logic, no DOM, so it can be tested in Node.
//
// Why: the old "pitch" was atan2(nose.y - eyeMid.y, eyeDist) on 2D landmarks. That number
// mostly measures how the phone is held (it sat at ~20 degrees in nearly every session) and
// it cannot tell "head tilted down" from "face is foreshortened because of the camera angle".
// MediaPipe can also return a 4x4 face transformation matrix (a true 3D pose of the face
// relative to the camera). Pitch/yaw/roll from that matrix are the right input for
// "head down", and they let us tell head movement apart from eye movement.

const RAD2DEG = 180 / Math.PI;
const clamp01 = (x) => Math.max(0, Math.min(1, x));

// d: 16 numbers. MediaPipe's layout is not guaranteed by the JS wrapper, so detect it:
// the bottom row of a rigid transform is [0,0,0,1] and the translation is non-zero.
function rotationFromMatrix(d) {
  if (!d || d.length < 16) return null;
  for (let i = 0; i < 16; i++) if (!Number.isFinite(d[i])) return null;
  const near = (v, t) => Math.abs(v - t) < 1e-3;
  let colMajor;
  if (near(d[3], 0) && near(d[7], 0) && near(d[11], 0) && near(d[15], 1)) colMajor = true;
  else if (near(d[12], 0) && near(d[13], 0) && near(d[14], 0) && near(d[15], 1)) colMajor = false;
  else return null; // not a rigid transform we understand: say so instead of guessing
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R[r][c] = colMajor ? d[c * 4 + r] : d[r * 4 + c];
  // remove any uniform scale so the angles stay valid
  for (let c = 0; c < 3; c++) {
    const n = Math.hypot(R[0][c], R[1][c], R[2][c]);
    if (!n) return null;
    for (let r = 0; r < 3; r++) R[r][c] /= n;
  }
  return R;
}

// Returns { pitch, yaw, roll } in degrees, or null. Sign of pitch is arbitrary here;
// makePitchOrienter() fixes it so that positive = head tilted down.
export function poseFromMatrix(d) {
  const R = rotationFromMatrix(d);
  if (!R) return null;
  const pitch = Math.asin(Math.max(-1, Math.min(1, -R[1][2]))) * RAD2DEG;
  const yaw = Math.atan2(R[0][2], R[2][2]) * RAD2DEG;
  const roll = Math.atan2(R[1][0], R[1][1]) * RAD2DEG;
  return { pitch, yaw, roll };
}

// Learns, during the session, which direction of the matrix pitch means "head down", by
// correlating it with the old 2D nose-vs-eyes proxy (which grows when the head tilts down).
// No assumption about the matrix axis convention is baked in.
export function makePitchOrienter(minFrames = 60) {
  let n = 0, mA = 0, mB = 0, cov = 0;
  return {
    update(matrixPitch, proxyPitch) {
      n += 1;
      const dA = matrixPitch - mA;
      mA += dA / n;
      const dB = proxyPitch - mB;
      mB += dB / n;
      cov += dA * (proxyPitch - mB);
      return this.sign;
    },
    get sign() { return n >= minFrames && cov < 0 ? -1 : 1; },
    get frames() { return n; },
    get confident() { return n >= minFrames; },
  };
}

// Inter-eye distance as a fraction of frame width. Small = far away / tiny face = noisy eyes.
export function faceSizeFrac(lm, w, h) {
  if (!lm || !lm[33] || !lm[263] || !w) return null;
  return Math.hypot((lm[33].x - lm[263].x) * w, (lm[33].y - lm[263].y) * h) / w;
}

// 0..1 "how much should I trust the eye numbers this frame". Heuristic, documented so it can be tuned
// from logged data: turning the head sideways, tilting far from the person's own baseline, a small face,
// or eyelids that are already half-dropped from looking down all make blink/gaze readings unreliable.
export function frameQuality({ yaw, pitchRel, size, lookDown }) {
  const f = [];
  if (typeof yaw === "number") f.push(1 - clamp01((Math.abs(yaw) - 15) / 25));          // full trust within 15 deg, none at 40
  if (typeof pitchRel === "number") f.push(1 - clamp01((Math.abs(pitchRel) - 15) / 25)); // relative to own baseline
  if (typeof size === "number") f.push(clamp01((size - 0.08) / 0.07));                  // below 8% of width: far/small
  if (typeof lookDown === "number") f.push(1 - 0.6 * clamp01((lookDown - 0.4) / 0.4));   // lids drop when eyes look down
  if (!f.length) return null;
  return f.reduce((a, b) => a * b, 1);
}
