// Vercel serverless function. Runs on Node's built-in runtime — no npm
// install needed, `fetch` is globally available in Node 18+.
// Keeps GEMINI_API_KEY server-side; the browser never sees it.

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "Server missing GEMINI_API_KEY" });
    return;
  }

  const { durationSec, blinkCount, blinkRatePerMin, gazeStabilityScore, postureStabilityScore, transcript, noSpeechDetected } =
    req.body || {};

  const isShortSession = durationSec < 25;
  const extremeMetrics = [gazeStabilityScore, postureStabilityScore].some((v) => v === 0 || v === 100);
  const hasTranscript = transcript && transcript.trim().length > 0 && !noSpeechDetected;

  const prompt = `You are giving feedback to someone practicing answering interview questions on camera. Here is their session data:

- Duration: ${durationSec} seconds
- Blinks: ${blinkCount} total (${blinkRatePerMin} per minute)
- Gaze stability score: ${gazeStabilityScore}/100
- Posture stability score: ${postureStabilityScore}/100
${hasTranscript ? `- What they said (transcript): """${transcript}"""` : "- No speech was captured during this session (silence throughout, or audio wasn't detected)."}

Rules you must follow:
- Reference the specific numbers above directly — never give generic advice that could apply to anyone.
${isShortSession ? "- This session was very short (under 25 seconds) — explicitly acknowledge the sample is too small to draw a strong conclusion, rather than inventing confident feedback from thin data.\n" : ""}${extremeMetrics ? "- One or more scores hit an extreme (0 or 100) — flag that this likely reflects a measurement edge case or very brief/unusual sample, not necessarily a real pattern.\n" : ""}${hasTranscript ? "- Blend BOTH verbal and nonverbal observations: comment on actual content from the transcript (filler words like \"um\"/\"like\", pacing, clarity, structure of the answer) AND the visual signals (gaze, posture, blinks) together — don't treat them as two separate reports." : "- No speech was captured, so do NOT invent or guess at anything about what they said, filler words, pacing, or content quality. Explicitly note that no speech was detected, and base your feedback only on the visual signals (gaze, posture, blinks)."}
- Frame feedback around what these signals typically mean in an interview context. For example: gaze instability late in an answer often maps to losing confidence or running out of prepared points. Rigid, completely unchanging posture the whole time isn't necessarily good either — it can read as tense rather than composed.
- Keep it to 3-4 complete sentences total. Make sure your response ends with a complete sentence — do not cut off mid-thought.
- Never use stock phrases like "make more eye contact" or "sit up straight" without tying them directly to the numbers above.
- Plain text only, no markdown formatting.`;

  try {
    const geminiRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          generationConfig: {
            maxOutputTokens: 600,
            // Newer Gemini models spend part of maxOutputTokens on hidden
            // "thinking" tokens before writing the visible answer. Without
            // capping that, a short visible response can get cut off even
            // with a generous token limit. Keeping thinking minimal leaves
            // the budget for the actual written feedback.
            thinkingConfig: { thinkingLevel: "low" },
          },
        }),
      }
    );
    const geminiJson = await geminiRes.json();

    // Log the full response (truncated) so real failures are diagnosable
    // from Vercel's function logs without needing browser devtools.
    console.log("Gemini raw response:", JSON.stringify(geminiJson).slice(0, 1500));

    const finishReason = geminiJson.candidates?.[0]?.finishReason;
    if (finishReason && finishReason !== "STOP") {
      console.warn(`Gemini finishReason was "${finishReason}", not STOP.`);
    }

    if (!geminiRes.ok) {
      console.error("Gemini API error (non-OK status):", JSON.stringify(geminiJson));
      res.status(200).json({
        feedback: "Coach feedback isn't available right now, but your session stats above are accurate — try again shortly.",
      });
      return;
    }

    const feedback =
      geminiJson.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ||
      "Your session data looked reasonable, but we couldn't generate written feedback this time — try again in a moment.";

    res.status(200).json({ feedback, _finishReason: finishReason });
  } catch (err) {
    console.error("Failed to reach Gemini:", String(err));
    res.status(500).json({ error: "Failed to reach Gemini", detail: String(err) });
  }
};
