# VoxAgent Studio

Streamlit voice AI assistant: Capture, Clean audio, STT, Intent and emotion, Reason + act (tools), TTS, with a live inspector.

Run: `pip install -r requirements.txt` then `streamlit run app.py`.

Real AI needs an OpenAI API key pasted in the sidebar (kept only in the browser session, never stored). Without a key it runs in free mode: typed input, rule-based intent, live Open-Meteo weather, gTTS voice.

## What is real, simulated, or missing
- **Streaming + pipelined TTS (Voice Studio):** the LLM reply is streamed; each finished sentence is sent to TTS in parallel while the model keeps writing. Latency tab shows a per-step timeline and per-chunk timings. "First audio chunk ready" is server-side; the browser player appears when the turn finishes rendering. The "time saved" line is an estimate.
- **Validated with a mock OpenAI-compatible server only.** The real OpenAI key path has not been run by the author. Set `VOX_OPENAI_BASE` to point at a test server.
- **Realtime tab:** a sentence-chunked TTS demo. Not the OpenAI Realtime API. No duplex and no barge-in (out of scope).
- **Bookings, calendar, tickets, appointments:** simulated in session memory. Nothing real is booked. Weather is live (Open-Meteo).
- **Local Whisper:** offered only if `faster-whisper` is installed (not on the hosted app).
- **Free mode** (no key) is a rule-based fallback, not an LLM.

- **Early playback:** each finished chunk is handed to a small player in the browser page, which starts playing chunk 1 while later sentences are still being generated and chains the rest in order. Tested only with a mock server in headless Chromium (play events logged in `window.__vox.log`: chunk 1 audible at ~3.0 s, last chunk ready at ~5.3 s). Real-key and real-speaker behaviour are untested; browsers may block autoplay without a prior click or keypress.
