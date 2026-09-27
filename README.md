# SpeakFlow (Web)

A browser-based presence/focus coach. Uses your webcam + MediaPipe Face Mesh
(runs entirely client-side) to track blink rate, gaze stability, and posture
during a practice session, then asks Gemini for written feedback.

## What's in here

- `index.html` / `style.css` / `app.js` — the whole frontend. No build step,
  no npm install needed. MediaPipe loads from a CDN at runtime.
- `api/feedback.js` — one Vercel serverless function. Takes the numeric
  session summary, calls Gemini, returns feedback text. This is the only
  server-side code, and it has zero dependencies (uses Node's built-in
  `fetch`).

Nothing here requires `npm install` to run or deploy.

## Cost check (per your constraints)

- Vercel free tier: covers this easily (static hosting + 1 serverless
  function, well under free-tier limits). No card required.
- Gemini API: free tier, using your existing key.
- MediaPipe Face Mesh: fully free, open-source, loaded from Google's public
  CDN.

If anything here ever needs a paid plan, that means usage has grown well
beyond a hackathon demo — worth flagging at that point, not now.

## Deploy (no terminal required)

1. Push this folder to a GitHub repo (new repo, e.g. `speakflow-web`).
2. Go to https://vercel.com, sign up with GitHub (free, no card).
3. Click **Add New → Project**, select your `speakflow-web` repo.
4. Leave all build settings as default — Vercel auto-detects a static
   site + `/api` function, no config needed.
5. Before deploying, add an environment variable:
   - **Name:** `GEMINI_API_KEY`
   - **Value:** your Gemini key
6. Click **Deploy**. You'll get a live URL like `speakflow-web.vercel.app`
   in about 30 seconds — no build queue wait like native app builds had.

## Local testing (optional)

Camera access requires a "secure context" (HTTPS or localhost) — opening
`index.html` directly via `file://` may block the camera in some browsers.
If you want to test before deploying:

```
npx serve .
```

(This one-time command doesn't install anything into the project — it just
runs a temporary local server.) Then open the printed `localhost` URL.

Otherwise, simplest path: just push to GitHub and test on the live Vercel
URL directly — avoids local server setup entirely.

## How the tracking works (approximate, by design)

- **Blink rate**: MediaPixe's built-in `eyeBlinkLeft`/`eyeBlinkRight`
  blendshape scores, counted as a blink when they cross a threshold.
- **Gaze stability**: how often the eyes' "look away" blendshapes
  (up/down/in/out) spike, averaged over the session.
- **Posture**: approximated from the nose tip and eye-corner landmark
  positions — no separate pose model, exactly as scoped.

These are proxies, not clinical-grade measurements — good enough for a
demo and directionally meaningful, not a medical or scientific tool.

## Known limitations (v1, by design)

- No accounts, no history — refreshing the page loses past sessions.
- No audio/speech analysis in this version.
- Works best in Chrome/Edge; camera permission UX varies slightly by browser.
