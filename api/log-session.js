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

  try {
    const response = await fetch(`${SUPABASE_URL}/rest/v1/sessions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        apikey: SUPABASE_KEY,
        Authorization: `Bearer ${SUPABASE_KEY}`,
        Prefer: "return=minimal",
      },
      body: JSON.stringify({
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
      }),
    });

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
