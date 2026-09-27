// Vercel serverless function — transcribes recorded session audio via
// Deepgram. Receives raw audio bytes in the request body (whatever
// MediaRecorder's mimeType produced), returns { transcript, noSpeechDetected }.

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const apiKey = process.env.DEEPGRAM_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "Server missing DEEPGRAM_API_KEY" });
    return;
  }

  try {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const audioBuffer = Buffer.concat(chunks);

    // A near-empty buffer means essentially no audio was captured at all —
    // don't bother calling Deepgram, just report no speech directly.
    if (audioBuffer.length < 2000) {
      res.status(200).json({ transcript: "", noSpeechDetected: true });
      return;
    }

    const contentType = req.headers["content-type"] || "audio/webm";

    const dgRes = await fetch(
      "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
      {
        method: "POST",
        headers: {
          Authorization: `Token ${apiKey}`,
          "Content-Type": contentType,
        },
        body: audioBuffer,
      }
    );
    const dgJson = await dgRes.json();

    if (!dgRes.ok) {
      console.error("Deepgram error:", JSON.stringify(dgJson));
      res.status(200).json({ transcript: "", noSpeechDetected: true, error: true });
      return;
    }

    const transcript = dgJson.results?.channels?.[0]?.alternatives?.[0]?.transcript ?? "";

    res.status(200).json({
      transcript,
      noSpeechDetected: transcript.trim().length === 0,
    });
  } catch (err) {
    console.error("Transcription failed:", String(err));
    res.status(500).json({ error: "Transcription failed", detail: String(err) });
  }
};
