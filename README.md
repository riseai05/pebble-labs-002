# Pebble Labs — Experiment 002

Focus/attention tracking during reading or work. Uses your webcam +
MediaPipe Face Mesh (runs entirely client-side) to track blink rate,
gaze stability, and head posture during a 10-minute session, with
periodic self-report check-ins ("still focused?").

## What's in here
- `index.html` / `style.css` / `app.js` — the whole frontend. No build step.
  MediaPipe loads from a CDN at runtime.
- `api/log-session.js` — Vercel serverless function that logs each
  session's numeric summary + self-reports to Supabase.

No video ever leaves the browser. Only numeric summaries and
self-report answers are stored.

Part of Pebble Labs, PebbleX's research arm. Experiment 001 is
SpeakFlow (interview practice mode).
