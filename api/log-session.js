// /api/log-session.js
// Receives a session summary from app.js and inserts it into the
// Supabase "sessions" table via Supabase's REST API.
// No video/audio ever reaches this function — only numeric summaries,
// interaction-event timestamps (blur/visibility/idle/trigger — never
// keystroke content or mouse position), and the tester's own answers.

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const SUPABASE_URL = process.env.SUPABASE_URL;
  const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY;

  if (!SUPABASE_URL || !SUPABASE_KEY) {
    console.error("Missing SUPABASE_URL or SUPABASE_ANON_KEY env vars");
    res.status(500).json({ error: "Server not configured" });
    return;
  }

  const {
    sessionId,
    timestamp,
    nickname,
    timeline,
    perf,
    sessionNotes,
    keyEvents,
    blinkTrace,
    taskType,
    deviceType,
    durationSec,
    blinkCount,
    blinkRatePerMin,
    gazeStabilityScore,
    postureStabilityScore,
    headDropAngle,
    fidgetScore,
    selfReports,
    distractionEvents,
    exitSurvey,
  } = req.body || {};

  if (!sessionId) {
    res.status(400).json({ error: "Missing sessionId" });
    return;
  }

  // The session row is saved the moment a session ends. The exit survey, if the
  // person fills it in, arrives later as a second request that only adds its
  // answers to that same row (so closing the tab on the survey never loses data).
  if (req.body && req.body.mode === "survey") {
    if (!/^[0-9a-fA-F-]{36}$/.test(String(sessionId))) {
      res.status(400).json({ error: "Bad sessionId" });
      return;
    }
    try {
      const r = await fetch(`${SUPABASE_URL}/rest/v1/sessions?session_id=eq.${sessionId}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          apikey: SUPABASE_KEY,
          Authorization: `Bearer ${SUPABASE_KEY}`,
          Prefer: "return=minimal",
        },
        body: JSON.stringify({ exit_survey: exitSurvey || null }),
      });
      if (!r.ok) {
        console.error("Supabase survey update failed:", r.status, await r.text());
        res.status(502).json({ error: "Failed to save survey" });
        return;
      }
      res.status(200).json({ ok: true });
    } catch (err) {
      console.error("Unexpected error saving survey:", err);
      res.status(500).json({ error: "Unexpected server error" });
    }
    return;
  }

  try {
    const row = {
      session_id: sessionId,
      created_at: timestamp,
      task_type: taskType,
      device_type: deviceType,
      duration_sec: durationSec,
      blink_count: blinkCount,
      blink_rate_per_min: blinkRatePerMin,
      gaze_stability_score: gazeStabilityScore,
      posture_stability_score: postureStabilityScore,
      head_drop_angle: headDropAngle,
      fidget_score: fidgetScore,
      self_reports: selfReports,
      distraction_events: distractionEvents,
      exit_survey: exitSurvey,
    };

    const insertRow = (body) =>
      fetch(`${SUPABASE_URL}/rest/v1/sessions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          apikey: SUPABASE_KEY,
          Authorization: `Bearer ${SUPABASE_KEY}`,
          Prefer: "return=minimal",
        },
        body: JSON.stringify(body),
      });

    // nickname is optional and saved to Supabase only. If the column hasn't been
    // added yet, retry without it so the session itself is never lost.
    const nick = typeof nickname === "string" ? nickname.trim().slice(0, 30) : "";
    // Optional columns (nickname, timeline). If a column hasn't been added in
    // Supabase yet, retry without the extras so the session itself is never lost.
    const extras = {};
    if (nick) extras.nickname = nick;
    if (Array.isArray(timeline) && timeline.length) extras.timeline = timeline;
    if (perf && typeof perf === "object") extras.perf = perf;
    if (typeof sessionNotes === "string" && sessionNotes) extras.session_notes = sessionNotes.slice(0, 4000);
    if (Array.isArray(keyEvents) && keyEvents.length) extras.key_events = keyEvents;
    if (blinkTrace && Array.isArray(blinkTrace.t) && blinkTrace.t.length && blinkTrace.t.length <= 12000) extras.blink_trace = blinkTrace;

    let response = await insertRow({ ...row, ...extras });
    if (!response.ok && Object.keys(extras).length) {
      // try each extra on its own so one missing column doesn't drop the other
      let saved = false;
      for (const key of Object.keys(extras)) {
        response = await insertRow({ ...row, [key]: extras[key] });
        if (response.ok) { saved = true; break; }
      }
      if (!saved) response = await insertRow(row);
    }

    if (!response.ok) {
      const errText = await response.text();
      console.error("Supabase insert failed:", response.status, errText);
      res.status(502).json({ error: "Failed to save session" });
      return;
    }

    res.status(200).json({ ok: true });
  } catch (err) {
    console.error("Unexpected error logging session:", err);
    res.status(500).json({ error: "Unexpected server error" });
  }
}
