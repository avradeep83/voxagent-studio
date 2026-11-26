"""VoxAgent Studio - a Streamlit voice AI assistant.

Pipeline (each stage is timed and shown live):
  Capture -> Clean audio -> Speech-to-Text -> Intent & emotion -> Reason + act (tools) -> Text-to-Speech

Works two ways:
  * With an OpenAI API key (pasted in the sidebar, kept only in this session):
    cloud STT, LLM reasoning with tool calls, OpenAI TTS.
  * Without a key (free mode): typed text, rule-based intent, real weather from
    Open-Meteo, spoken replies through gTTS. Uploaded/spoken audio needs a key
    (or the optional local faster-whisper package).
"""
import importlib.util
import io
import json
import os
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import base64
import uuid

st.set_page_config(page_title="VoxAgent Studio", page_icon="🎙️", layout="wide")

OPENAI = os.environ.get("VOX_OPENAI_BASE", "https://api.openai.com/v1")  # override only for tests against a mock server
GEMINI_OAI = os.environ.get("VOX_GEMINI_OAI_BASE", "https://generativelanguage.googleapis.com/v1beta/openai")  # OpenAI-compatible chat endpoint
GEMINI_NATIVE = os.environ.get("VOX_GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")  # native endpoint (audio in, TTS)
PROV = {"name": "OpenAI"}  # set from the sidebar on every run
HAS_LOCAL_WHISPER = importlib.util.find_spec("faster_whisper") is not None
SIMULATED_TOOLS = {"schedule_meeting", "book_restaurant_table", "book_appointment", "create_ticket", "remember"}

PERSONAS = {
    "Executive Assistant": {
        "blurb": "Calendar, reservations, weather, maths: the classic 'Alexa for work'.",
        "system": "You are an efficient executive assistant. Use tools to schedule meetings, book tables and check weather. Ask a short follow-up when a detail is missing instead of guessing.",
        "samples": [
            "Schedule a meeting with Priya tomorrow at 10 AM",
            "Book a table for two at 7 PM tonight at an Italian place",
            "What is the weather in Mumbai right now? Should I carry an umbrella?",
            "Remember that I prefer window seats and vegetarian food",
        ],
    },
    "Customer Support": {
        "blurb": "Order tickets, FAQs and escalations.",
        "system": "You are a calm customer support agent. Create support tickets for problems, answer FAQs with the faq tool, and be brief and empathetic.",
        "samples": [
            "My order ORD-1002 has not arrived and I am really frustrated",
            "What is your refund policy?",
            "Raise a ticket, the app keeps crashing on login",
        ],
    },
    "Clinic Front Desk": {
        "blurb": "Appointments and clinic FAQs (demo data only, no medical advice).",
        "system": "You are a clinic front desk assistant. Book appointments, answer opening-hours questions, never give medical advice.",
        "samples": [
            "Book a dentist appointment on Friday at 4 PM",
            "What are your opening hours?",
        ],
    },
}

FAQ = {
    "refund": "Refunds are issued to the original payment method within 5 to 7 business days of approval.",
    "hours": "We are open Monday to Saturday, 9 AM to 6 PM. Closed on Sundays.",
    "opening": "We are open Monday to Saturday, 9 AM to 6 PM. Closed on Sundays.",
    "shipping": "Standard shipping takes 3 to 5 business days. Express takes 1 to 2.",
    "password": "Use 'Forgot password' on the login page. A reset link arrives within a minute.",
}

OPENAI_VOICES = ["coral", "alloy", "ash", "ballad", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
STT_MODELS = ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]
LLM_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]
GEMINI_LLMS = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.6-flash"]  # gemini-4-hydrogen is the model in Google's own chat, audio and function-calling examples
GEMINI_STT = ["gemini-4-hydrogen", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash"]  # general models with audio input; first is the one in Google's audio examples
GEMINI_TTS = ["gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"]
GEMINI_VOICES = ["Kore", "Puck", "Zephyr", "Charon", "Fenrir", "Leda"]


def is_gemini():
    return PROV["name"] == "Gemini"


def chat_base():
    return GEMINI_OAI if is_gemini() else OPENAI


# ----------------------------------------------------------------------------- state


def init_state():
    s = st.session_state
    s.setdefault("chat", [])  # {role, text, tool, audio, meta}
    s.setdefault("turns", [])  # per-turn inspector records
    s.setdefault(
        "world",
        {
            "calendar": [
                {"id": "MTG-001", "title": "Daily stand-up", "date": str(date.today()), "time": "09:30", "duration_minutes": 15, "attendees": "Platform team"},
                {"id": "MTG-002", "title": "Vendor review", "date": str(date.today()), "time": "14:00", "duration_minutes": 60, "attendees": "Procurement"},
            ],
            "reservations": [],
            "appointments": [],
            "tickets": [],
            "memory": [],
        },
    )
    s.setdefault("last_sample", None)


init_state()

# ----------------------------------------------------------------------------- audio utils


def wav_to_array(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 1:
        x = x - 128
    x /= float(2 ** (8 * sw - 1)) if sw > 1 else 128.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def array_to_wav(x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def snr_db(x):
    if len(x) < 400:
        return 0.0
    frame = max(int(len(x) / 50), 1)
    e = np.array([np.mean(x[i : i + frame] ** 2) + 1e-12 for i in range(0, len(x) - frame, frame)])
    noise = np.percentile(e, 10)
    sig = np.percentile(e, 90)
    return float(10 * np.log10(sig / noise))


def clean_audio(x, sr, strength=1.0):
    """High-pass + spectral-gating noise reduction + peak normalisation (numpy only)."""
    if len(x) < 2048:
        return x
    # high-pass at ~80 Hz via simple one-pole filter
    a = np.exp(-2 * np.pi * 80 / sr)
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    pad = np.pad(y, (n_fft, n_fft))
    frames = np.array([pad[i : i + n_fft] * win for i in range(0, len(pad) - n_fft, hop)])
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    energy = mag.mean(axis=1)
    quiet = mag[energy <= np.percentile(energy, 15)]
    noise_profile = quiet.mean(axis=0) if len(quiet) else mag.min(axis=0)
    gain = np.clip((mag - strength * 1.5 * noise_profile) / (mag + 1e-9), 0.05, 1.0)
    cleaned = np.fft.irfft(spec * gain, axis=1)
    out = np.zeros(len(pad))
    norm = np.zeros(len(pad))
    for k, f in enumerate(cleaned):
        s = k * hop
        out[s : s + n_fft] += f * win
        norm[s : s + n_fft] += win**2
    out = (out / np.maximum(norm, 1e-6))[n_fft : n_fft + len(y)]
    peak = np.max(np.abs(out)) or 1.0
    return out * min(0.9 / peak, 4.0)


# ----------------------------------------------------------------------------- OpenAI helpers


BILLING_CODES = {"insufficient_quota", "billing_hard_limit_reached", "organization_usage_limit_exceeded", "organization_spend_limit_exceeded", "project_spend_limit_exceeded", "credit_balance_exhausted"}


def check(r):
    """Like raise_for_status, but shows the provider's own error code/message (keys scrubbed) and what kind of 429 it is."""
    if r.ok:
        return
    who = "Gemini" if is_gemini() else "OpenAI"
    code = typ = msg = ""
    try:
        body = r.json()
        if isinstance(body, list) and body:
            body = body[0]
        e = (body or {}).get("error") or {}
        code, typ, msg = str(e.get("code") or ""), str(e.get("type") or e.get("status") or ""), str(e.get("message") or "")
    except Exception:  # noqa: BLE001
        msg = (r.text or "")[:200]
    msg = re.sub(r"(sk-|AIza)[A-Za-z0-9_\-\*\.]{4,}", r"\1***", msg)[:300]
    rid = r.headers.get("x-request-id", "")
    hint = ""
    if r.status_code == 429:
        if who == "Gemini":
            hint = "Gemini quota or rate limit for this key's Google project (free-tier limits are per minute and per day, per project; daily quota resets at midnight Pacific). Check the rate limits shown in Google AI Studio, wait a minute, or use a different model."
        elif code in BILLING_CODES or typ == "insufficient_quota":
            hint = "Credit, quota or spend limit reached on the account/project that owns this key. Retrying will not help; check Billing and Limits in the OpenAI dashboard."
        elif code == "rate_limit_exceeded" or typ == "requests":
            hint = "Rate limit (too many requests or tokens per minute). Wait a minute and try once."
        else:
            hint = "429 without a recognised code, so rate limit vs credit limit cannot be told apart from this response."
    elif r.status_code in (401, 403):
        hint = "The key was rejected" + (" or the Generative Language API is not enabled/allowed for it (Gemini)." if who == "Gemini" else ". Check it was copied fully and has not been revoked.")
    elif r.status_code == 400 and who == "Gemini":
        hint = "Bad request. Check the model name is available to this key and the request is valid."
    elif r.status_code == 404:
        hint = "Model not available to this key/project."
    elif r.status_code in (500, 502, 503, 504):
        hint = ("Temporary overload or outage on the provider's side, not a problem with your key or audio. The app does not retry for you. Wait" + (f" at least {r.headers.get('retry-after')} seconds" if r.headers.get("retry-after") else " a minute") + ", then try once more by hand. If it keeps failing, choose a different model for this step in the sidebar and retry once.")
    ra = r.headers.get("retry-after")
    raise RuntimeError(f"HTTP {r.status_code} from {who}" + (f" | code: {code}" if code else "") + (f" | type: {typ}" if typ else "") + (f" | message: {msg}" if msg else "") + (f" | retry-after: {ra}s" if ra else "") + (f" | request id: {rid}" if rid else "") + (f" | {hint}" if hint else ""))


def headers(key):
    return {"Authorization": f"Bearer {key}"}


def openai_stt(key, model, wav_bytes, language, vocab):
    data = {"model": model}
    if language and language != "Auto-detect":
        data["language"] = language.split("(")[-1].strip(")")
    if vocab:
        data["prompt"] = "Domain vocabulary: " + vocab
    r = requests.post(f"{OPENAI}/audio/transcriptions", headers=headers(key), data=data, files={"file": ("speech.wav", wav_bytes, "audio/wav")}, timeout=90)
    check(r)
    return r.json().get("text", "").strip()


def local_stt(wav_bytes):
    from faster_whisper import WhisperModel  # optional

    model = st.session_state.get("_fw") or WhisperModel("base", compute_type="int8")
    st.session_state["_fw"] = model
    segs, _ = model.transcribe(io.BytesIO(wav_bytes))
    return " ".join(s.text.strip() for s in segs)


def openai_tts(key, model, voice, speed, text, tone):
    body = {"model": model, "voice": voice, "input": text, "speed": speed, "response_format": "mp3"}
    if "gpt-4o" in model and tone:
        body["instructions"] = tone
    r = requests.post(f"{OPENAI}/audio/speech", headers=headers(key), json=body, timeout=90)
    check(r)
    return r.content


def _find_blocks(o, out):
    if isinstance(o, dict):
        out.append(o)
        for v in o.values():
            _find_blocks(v, out)
    elif isinstance(o, list):
        for v in o:
            _find_blocks(v, out)


def gemini_interact(key, body):
    r = requests.post(f"{GEMINI_NATIVE}/interactions", headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=120)
    check(r)
    return r.json()


def gemini_stt(key, model, wav_bytes, language, vocab):
    prompt = "Generate a verbatim transcript of the speech. Reply with only the transcript text."
    if language and language != "Auto-detect":
        prompt += f" The language is {language.split('(')[0].strip()}."
    if vocab:
        prompt += f" Domain vocabulary and spellings: {vocab}."
    out = gemini_interact(key, {"model": model, "input": [{"type": "text", "text": prompt}, {"type": "audio", "data": base64.b64encode(wav_bytes).decode(), "mime_type": "audio/wav"}]})  # flat input list, as in the official audio examples
    blocks = []
    _find_blocks(out.get("steps", out), blocks)
    texts = [b["text"] for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str)]
    return (texts[-1] if texts else "").strip()


def gemini_tts(key, model, voice, text, tone):
    block = {"type": "text", "text": text}
    if tone:
        block["annotations"] = [{"type": "speech_metadata", "style": tone}]
    out = gemini_interact(key, {"model": model, "input": [{"type": "user_input", "content": [block]}], "response_format": {"type": "audio", "mime_type": "audio/wav"}, "generation_config": {"speech_config": [{"voice": voice}]}})
    blocks = []
    _find_blocks(out.get("steps", out), blocks)
    for b in blocks:
        if isinstance(b.get("data"), str) and (b.get("type") == "audio" or str(b.get("mime_type", "")).startswith("audio")):
            return normalize_audio(base64.b64decode(b["data"]), str(b.get("mime_type", "")))
    raise RuntimeError("Gemini TTS returned no audio block")


def normalize_audio(raw, mime=""):
    """Documented contract: unary TTS returns a complete WAV (RIFF). Headerless audio/l16 (24 kHz mono 16-bit LE) is wrapped into WAV so browsers can play it."""
    if raw[:4] == b"RIFF":
        return raw
    if "l16" in mime.lower() or not mime or mime.lower().startswith("audio/pcm"):
        m = re.search(r"rate=(\d+)", mime)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(m.group(1)) if m else 24000)
            w.writeframes(raw)
        return out.getvalue()
    raise RuntimeError(f"Unsupported Gemini audio type: {mime or 'unknown'}")


def audio_mime(b):
    return "audio/wav" if b[:4] == b"RIFF" else "audio/mpeg"


def join_audio(parts):
    """Join chunk audio into one file. WAV chunks are merged properly; MP3 chunks are concatenated."""
    if not parts:
        return None
    if all(p[:4] == b"RIFF" for p in parts):
        try:
            frames, params = [], None
            for p in parts:
                with wave.open(io.BytesIO(p)) as w:
                    if params is None:
                        params = w.getparams()
                    frames.append(w.readframes(w.getnframes()))
            out = io.BytesIO()
            with wave.open(out, "wb") as w:
                w.setparams(params)
                w.writeframes(b"".join(frames))
            return out.getvalue()
        except Exception:  # noqa: BLE001
            return parts[0]
    return b"".join(parts)


def speak_any(cfg, text, tone):
    """One TTS call for whichever provider is selected (safe to call from worker threads)."""
    if cfg.get("provider") == "Gemini":
        return gemini_tts(cfg["key"], cfg["tts_model"], cfg["voice"], text, tone)
    return openai_tts(cfg["key"], cfg["tts_model"], cfg["voice"], cfg["speed"], text, tone)


def gtts_tts(text):
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    return buf.getvalue()


TONES = {
    "frustrated": "Speak in a calm, warm, reassuring tone, slightly slower. Show empathy.",
    "happy": "Speak in a bright, friendly, upbeat tone.",
    "urgent": "Speak clearly and briskly, with focus.",
    "sad": "Speak gently and kindly.",
    "calm": "Speak in a clear, neutral, professional tone.",
}

# ----------------------------------------------------------------------------- streaming + pipelined TTS

SENT_RE = re.compile(r"(?<=[.!?])\s+")


PLAYER_JS = """(function(){ if(window.__vox) return; var V=window.__vox={q:[],playing:false,log:[],t0:{},turn:null,cur:null};
V.start=function(id){V.turn=id;V.t0[id]=performance.now();V.q=[];if(V.cur){try{V.cur.pause();}catch(e){}} V.playing=false;V.log.push({turn:id,ev:'start'});};
V.add=function(id,idx,uri){ if(id!==V.turn) return; V.q.push({idx:idx,uri:uri}); V.log.push({turn:id,chunk:idx,ev:'queued',ms:Math.round(performance.now()-V.t0[id])}); V.next();};
V.next=function(){ if(V.playing||!V.q.length) return; var it=V.q.shift(); V.playing=true; var a=new Audio(it.uri); V.cur=a; var id=V.turn;
 a.onplaying=function(){V.log.push({turn:id,chunk:it.idx,ev:'playing',ms:Math.round(performance.now()-V.t0[id])});};
 a.onended=function(){V.playing=false;V.next();}; a.onerror=function(){V.log.push({turn:id,chunk:it.idx,ev:'error'});V.playing=false;V.next();};
 a.play().catch(function(e){V.log.push({turn:id,chunk:it.idx,ev:'blocked',err:String(e)});V.playing=false;});};
})();"""


def _emit_js(js):
    """Run a snippet in the browser page (via a zero-height iframe) so audio can start while the turn is still running."""
    components.html("<script>var P=window.parent; if(!P.__vox){P.eval(" + json.dumps(PLAYER_JS) + ");} " + js + "</script>", height=0)


class TTSPipeline:
    """Cuts streamed text into sentences and synthesises each as soon as it is complete, in parallel threads."""

    def __init__(self, cfg, tone, enabled):
        self.cfg, self.tone, self.enabled = cfg, tone, enabled
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.jobs, self.buf, self.pending = [], "", ""
        self.emitted, self.on_ready = 0, None

    def _speak(self, text):
        t = time.time()
        if self.cfg["key"]:
            a = speak_any(self.cfg, text, self.tone)
        else:
            a = gtts_tts(text)
        return a, t, time.time()

    def _submit(self, text):
        text = text.strip()
        if text and self.enabled:
            self.jobs.append({"text": text, "queued": time.time(), "fut": self.pool.submit(self._speak, text)})

    def poll(self, block=False):
        """Hand finished chunks (in order) to on_ready, e.g. to start playback in the browser."""
        while self.on_ready and self.emitted < len(self.jobs):
            j = self.jobs[self.emitted]
            if not block and not j["fut"].done():
                break
            try:
                a = j["fut"].result(timeout=120)[0]
                self.on_ready(self.emitted + 1, a)
            except Exception:  # noqa: BLE001 - reported by collect()
                pass
            self.emitted += 1

    def feed(self, delta):
        self.poll()
        self.buf += delta
        parts = SENT_RE.split(self.buf)
        for p in parts[:-1]:
            self.pending += (" " if self.pending else "") + p
            if len(self.pending) >= 15:  # avoid tiny fragments like "Dr."
                self._submit(self.pending)
                self.pending = ""
        self.buf = parts[-1]

    def drop_partial(self):
        self.buf, self.pending = "", ""

    def flush(self):
        self._submit((self.pending + " " + self.buf).strip())
        self.buf, self.pending = "", ""

    def collect(self):
        self.poll(block=True)
        chunks, errs = [], []
        for i, j in enumerate(self.jobs):
            try:
                a, ts, te = j["fut"].result(timeout=120)
                chunks.append({"chunk": i + 1, "text": j["text"], "audio": a, "queued": j["queued"], "start": ts, "ready": te})
            except Exception as e:  # noqa: BLE001
                errs.append(f"TTS chunk {i + 1} failed: {e}")
        self.pool.shutdown(wait=False)
        return chunks, errs


def stream_chat(key, model, msgs, temp, on_text, tools=None):
    """Streamed chat completion. Calls on_text for each text delta; returns (content, tool_calls, first_token_ms)."""
    body = {"model": model, "messages": msgs, "temperature": temp, "stream": True}
    if tools:
        body["tools"] = tools
    t0 = time.time()
    first = None
    r = requests.post(f"{chat_base()}/chat/completions", headers=headers(key), json=body, stream=True, timeout=90)
    check(r)
    content, calls = "", {}
    for line in r.iter_lines(chunk_size=1, decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        choice = (json.loads(data).get("choices") or [{}])[0]
        d = choice.get("delta", {})
        if d.get("content"):
            if first is None:
                first = (time.time() - t0) * 1000
            content += d["content"]
            on_text(d["content"])
        for tc in d.get("tool_calls") or []:
            f = tc.get("function", {})
            idx = tc.get("index")
            if idx is None:  # Gemini's OpenAI-compatible stream can omit the index: a new name starts a new call
                idx = (max(calls) + 1 if f.get("name") else max(calls, default=0)) if calls else 0
            c = calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                c["id"] = tc["id"]
            if tc.get("extra_content"):  # Gemini 3 thought_signature: must be echoed back on the assistant tool_calls message
                ec = c.setdefault("extra_content", {})
                for k, v in tc["extra_content"].items():
                    if isinstance(v, dict):
                        ec.setdefault(k, {}).update(v)
                    else:
                        ec[k] = v
            c["function"]["name"] += f.get("name") or ""
            c["function"]["arguments"] += f.get("arguments") or ""
    out_calls = [calls[i] for i in sorted(calls)]
    for n, c in enumerate(out_calls):
        if not c["id"]:
            c["id"] = f"call_{n}"
    return content, out_calls, first


# ----------------------------------------------------------------------------- tools


def _get(url, params):
    for i in range(3):
        try:
            return requests.get(url, params=params, timeout=8).json()
        except requests.RequestException:
            if i == 2:
                raise


def weather_tool(city):
    g = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1})
    if not g.get("results"):
        return {"error": f"Could not find a place called {city}."}
    p = g["results"][0]
    w = _get(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": p["latitude"], "longitude": p["longitude"], "current": "temperature_2m,weather_code,precipitation,wind_speed_10m", "daily": "precipitation_probability_max", "timezone": "auto", "forecast_days": 1},
    )
    c = w["current"]
    code = c["weather_code"]
    cond = "clear sky" if code == 0 else "partly cloudy" if code in (1, 2, 3) else "foggy" if code in (45, 48) else "drizzle" if code in range(51, 58) else "rain" if code in range(61, 68) or code in range(80, 83) else "snow" if code in range(71, 78) else "thunderstorm" if code >= 95 else "mixed conditions"
    chance = (w.get("daily", {}).get("precipitation_probability_max") or [None])[0]
    return {"city": p["name"], "country": p.get("country"), "source": "open-meteo (live)", "condition": cond, "temperature_c": c["temperature_2m"], "precipitation_mm": c["precipitation"], "rain_chance_today_pct": chance, "wind_kmh": c["wind_speed_10m"]}


def parse_when(text):
    t = text.lower()
    d = date.today()
    if "tomorrow" in t:
        d += timedelta(days=1)
    for i, name in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
        if re.search(r"\b" + name + r"\b", t):
            d += timedelta(days=(i - d.weekday()) % 7 or 7)
            break
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if iso:
        try:
            d = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t)
    tm = None
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
        tm = f"{h:02d}:{int(m.group(2) or 0):02d}"
    return str(d), tm


_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def extract_booking(text):
    """Exact booking fields from what the user said. Missing ones are None (the caller must ask, never invent)."""
    t = text.lower()
    out = {"restaurant": None, "party_size": None, "date": None, "time": None}
    m = re.search(r"(?:\bparty of|\bfor|\btable of)\s+(\d{1,2}|" + "|".join(_NUMW) + r")\b(?!\s*(?::|[ap]\.?m|o'?clock))", t)
    if m:
        out["party_size"] = _NUMW.get(m.group(1)) or int(m.group(1))
    vt = re.sub(r"\b(?:at|around|by)?\s*\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)", " ", text, flags=re.I)
    vt = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", vt)
    vt = re.sub(r"\s+", " ", vt).strip()
    for m in re.finditer(r"\b(?:at|in|from)\s+(.+?)(?=\s+(?:for|on|tomorrow|today|tonight|this|next|around|by|please|at\s+\d)\b|[.?!]*\s*$)", vt, flags=re.I):
        v = m.group(1).strip(" .,'")
        if v and not v[0].isdigit() and v.split()[0].lower() not in {"a", "an", "the", "some", "any", "my", "your", "noon", "night", "lunch", "dinner", "breakfast", "the"}:
            out["restaurant"] = v
            break
    low = t
    if "tomorrow" in low or "today" in low or "tonight" in low or re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{4}-\d{2}-\d{2}\b", low):
        d, _ = parse_when(text)
        out["date"] = d
    _, tm = parse_when(text)
    if not tm:
        m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
        if m:
            tm = f"{int(m.group(1)):02d}:{m.group(2)}"
    if not tm and re.search(r"\bnoon\b", t):
        tm = "12:00"
    out["time"] = tm
    return out


def fmt_time(hhmm):
    h, m = map(int, hhmm.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def validate_booking(args):
    """Return an error string when party size, date or time is not a real value."""
    try:
        n = int(args["party_size"])
    except (TypeError, ValueError):
        return "Party size must be a whole number."
    if not 1 <= n <= 50:
        return "Party size must be between 1 and 50."
    try:
        date.fromisoformat(str(args["date"]))
    except ValueError:
        return "Date must be a real calendar date (YYYY-MM-DD)."
    m = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", str(args["time"]))
    if not m:
        return "Time must be a real 24-hour time (HH:MM)."
    return None


def safe_calc(expr):
    """Bounded arithmetic: + - * / ** % with small operands, short input, capped result."""
    import ast
    import operator as op
    if len(expr) > 100:
        raise ValueError("Expression too long")
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}
    count = [0]

    def ev(n):
        count[0] += 1
        if count[0] > 30:
            raise ValueError("Expression too complex")
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            if abs(n.value) > 1e12:
                raise ValueError("Number too large")
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6):
                raise ValueError("Exponent too large")
            r = ops[type(n.op)](a, b)
            if abs(r) > 1e18:
                raise ValueError("Result too large")
            return r
        raise ValueError("Unsupported expression")
    return ev(ast.parse(expr.strip(), mode="eval"))


def run_tool(name, args):
    try:
        return _run_tool(name, args)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{name} failed: {e}"}


def _run_tool(name, args):
    w = st.session_state.world
    if name == "get_weather":
        return weather_tool(args.get("city", "Mumbai"))
    if name == "schedule_meeting":
        mid = f"MTG-{len(w['calendar']) + 1:03d}"
        row = {"id": mid, "title": args.get("title", "Meeting"), "date": args.get("date", str(date.today())), "time": args.get("time", "10:00"), "duration_minutes": int(args.get("duration_minutes", 30)), "attendees": args.get("attendees", "")}
        clash = [m for m in w["calendar"] if m["date"] == row["date"] and m["time"] == row["time"]]
        if clash:
            return {"error": f"Conflict with {clash[0]['title']} at that time.", "conflict": clash[0]}
        w["calendar"].append(row)
        return {"status": "scheduled", **row}
    if name == "book_restaurant_table":
        need = [k for k in ("restaurant", "party_size", "date", "time") if not str(args.get(k) or "").strip()]
        if need:
            return {"error": "Missing booking details: " + ", ".join(need) + ". Ask the user; never invent them."}
        bad = validate_booking(args)
        if bad:
            return {"error": bad}
        row = {"id": f"RES-{len(w['reservations']) + 1:03d}", "restaurant": str(args["restaurant"]).strip(), "party_size": int(args["party_size"]), "date": str(args["date"]), "time": str(args["time"])}
        w["reservations"].append(row)
        return {"status": "booked (simulated demo, nothing real)", **row}
    if name == "book_appointment":
        row = {"id": f"APT-{len(w['appointments']) + 1:03d}", "service": args.get("service", "General visit"), "date": args.get("date", str(date.today())), "time": args.get("time", "16:00")}
        w["appointments"].append(row)
        return {"status": "booked", **row}
    if name == "create_ticket":
        row = {"id": f"TCK-{len(w['tickets']) + 1:03d}", "summary": args.get("summary", "Customer issue"), "priority": args.get("priority", "normal"), "order_id": args.get("order_id", "")}
        w["tickets"].append(row)
        return {"status": "created", **row}
    if name == "faq":
        q = args.get("question", "").lower()
        for k, v in FAQ.items():
            if k in q:
                return {"answer": v}
        return {"answer": "I could not find that in the FAQ. I can raise a ticket if you like."}
    if name == "remember":
        w["memory"].append(args.get("fact", ""))
        return {"status": "saved", "fact": args.get("fact", "")}
    if name == "calculate":
        expr = str(args.get("expression", ""))
        try:
            return {"result": safe_calc(expr)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
    return {"error": f"Unknown tool {name}"}


def fn(name, desc, props, req=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": list(req)}}}


S = {"type": "string"}
TOOLS = [
    fn("get_weather", "Live weather for a city.", {"city": S}, ["city"]),
    fn("schedule_meeting", "Add a calendar meeting. date is YYYY-MM-DD, time is 24h HH:MM.", {"title": S, "date": S, "time": S, "duration_minutes": {"type": "integer"}, "attendees": S}, ["title", "date", "time"]),
    fn("book_restaurant_table", "Reserve a restaurant table. Use EXACTLY the restaurant name and location, party size, date (YYYY-MM-DD) and time (24h HH:MM) the user said. If any is missing or unclear, do not call this tool: ask the user.", {"restaurant": S, "party_size": {"type": "integer"}, "date": S, "time": S}, ["restaurant", "party_size", "date", "time"]),
    fn("book_appointment", "Book an appointment.", {"service": S, "date": S, "time": S}, ["service", "date", "time"]),
    fn("create_ticket", "Create a support ticket.", {"summary": S, "priority": {"type": "string", "enum": ["low", "normal", "high"]}, "order_id": S}, ["summary"]),
    fn("faq", "Answer a policy or FAQ question.", {"question": S}, ["question"]),
    fn("remember", "Save a long-term user preference.", {"fact": S}, ["fact"]),
    fn("calculate", "Evaluate arithmetic.", {"expression": S}, ["expression"]),
]

# ----------------------------------------------------------------------------- intent / reasoning


EMO_WORDS = {
    "frustrated": ["frustrat", "angry", "annoy", "ridiculous", "still not", "again", "not arrived", "worst"],
    "happy": ["thanks", "great", "awesome", "love"],
    "urgent": ["urgent", "asap", "immediately"],
}


_CITY_STOP = {"the", "a", "an", "my", "this", "that", "it", "there", "here", "general", "time", "case", "order", "fact", "advance"}


def extract_city(text):
    """Pull a place name out of a weather question. Returns None when there is no clear city (the caller must ask, never guess)."""
    pat = r"\b(?:in|at|for|of|around|near)\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2}?)(?=\s+(?:right|today|tonight|tomorrow|now|and|or|do|does|will|should|is|be|please|this|next)\b|\s*[?.!,;]|\s*$)"
    for m in re.finditer(pat, text, flags=re.I):
        c = m.group(1).strip(" .,'-")
        if c and c.lower().split()[0] not in _CITY_STOP and not re.fullmatch(r"\d+", c):
            return c.title()
    m = re.search(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)(?:'s)?\s+(?:weather|forecast|temperature)", text)
    if m and m.group(1).lower() not in {"the", "what", "how", "whats", "today"}:
        return m.group(1).title()
    return None


def weather_advice(text, res):
    """Answer only what was asked, from live numbers. Rules, not AI."""
    t = text.lower()
    temp, rain = res.get("temperature_c"), res.get("rain_chance_today_pct")
    wet = (rain is not None and rain >= 40) or (res.get("precipitation_mm") or 0) > 0
    out = []
    if re.search(r"umbrella|raincoat|rain gear|\brain\b|raining", t):
        out.append("Yes, there is a decent chance of rain today, so carry an umbrella." if wet else ("No real chance of rain today, so you do not need an umbrella." if rain is not None else "I could not get the rain chance."))
    m = re.search(r"(blankets?|jackets?|sweaters?|coats?|hoodies?|woollens?|warm clothes|scarf|shawl)", t)
    if m and temp is not None:
        item = m.group(1)
        if temp <= 15:
            out.append(f"It is cold at {temp} degrees, so yes, take a {item.rstrip('s')} or something warm.")
        elif temp <= 22:
            out.append(f"It is mild at {temp} degrees, so a light layer is enough; a heavy {item.rstrip('s')} is probably not needed.")
        else:
            out.append(f"At {temp} degrees you do not need a {item.rstrip('s')}.")
    return " ".join(out)


def heuristic_intent(text):
    t = text.lower()
    emo = "calm"
    for k, ws in EMO_WORDS.items():
        if any(w in t for w in ws):
            emo = k
            break
    ent = {}
    d, tm = parse_when(text)
    if tm:
        ent["time"] = tm
    ent["date"] = d
    m = re.search(r"(?:for|party of)\s+(\d+|two|three|four|five|six)", t)
    if m:
        ent["party_size"] = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}.get(m.group(1)) or int(m.group(1))
    city = extract_city(text)
    if city:
        ent["city"] = city
    m = re.search(r"ord-\d+", t)
    if m:
        ent["order_id"] = m.group(0).upper()
    if "weather" in t or "umbrella" in t or "temperature" in t:
        intent = "get_weather"
    elif re.search(r"remember|prefer", t):
        intent = "remember"
    elif "table" in t or "reserv" in t or "restaurant" in t:
        intent = "book_restaurant_table"
    elif re.search(r"schedule|meeting|call with|set up a call", t):
        intent = "schedule_meeting"
    elif "appointment" in t or "dentist" in t or "doctor" in t:
        intent = "book_appointment"
    elif re.search(r"ticket|crash|not arrived|broken|complain|problem", t):
        intent = "create_ticket"
    elif re.search(r"refund|policy|hours|opening|shipping|password", t):
        intent = "faq"
    elif re.search(r"\d\s*[-+*/x]\s*\d|calculate|what is \d", t):
        intent = "calculate"
    else:
        intent = "chitchat"
    return {"intent": intent, "emotion": emo, "entities": ent, "confidence": 0.7 if intent != "chitchat" else 0.4, "method": "rules"}


def llm_intent(key, model, text):
    msgs = [
        {"role": "system", "content": "Classify the user utterance. Reply as JSON: {intent: one of get_weather|schedule_meeting|book_restaurant_table|book_appointment|create_ticket|faq|remember|calculate|chitchat, emotion: calm|happy|frustrated|urgent|sad, entities: object, confidence: 0-1}. Today is " + str(date.today()) + "."},
        {"role": "user", "content": text},
    ]
    r = requests.post(f"{chat_base()}/chat/completions", headers=headers(key), json={"model": model, "messages": msgs, "response_format": {"type": "json_object"}, "temperature": 0}, timeout=60)
    check(r)
    raw = (r.json()["choices"][0]["message"]["content"] or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    out = json.loads(raw)
    out["method"] = "llm"
    return out


def offline_reply(text, intent):
    """No-LLM agent: choose a tool from the rule-based intent, run it, template the reply."""
    ent = intent["entities"]
    i = intent["intent"]
    calls = []
    if i == "get_weather":
        if not ent.get("city"):
            return "Which city do you want the weather for? I could not find a place name in your request.", calls
        res = run_tool("get_weather", {"city": ent["city"]})
        calls.append(("get_weather", {"city": ent["city"]}, res))
        if "error" in res:
            return (res["error"] if "Could not find" in res["error"] else "I could not reach the weather service just now. Please try again in a moment."), calls
        adv = weather_advice(text, res)
        return f"The weather in {res['city']} right now is {res['condition']} at {res['temperature_c']} degrees Celsius." + (" " + adv if adv else ""), calls
    if i == "schedule_meeting":
        if "time" not in ent:
            return "Sure. What time should I schedule it?", calls
        who = re.search(r"with ([A-Z][a-z]+(?: [A-Z][a-z]+)?|the vendor)", text)
        args = {"title": f"Call with {who.group(1)}" if who else "Meeting", "date": ent["date"], "time": ent["time"], "attendees": who.group(1) if who else ""}
        res = run_tool("schedule_meeting", args)
        calls.append(("schedule_meeting", args, res))
        return (res["error"] if "error" in res else f"Done. I scheduled {args['title']} on {res['date']} at {res['time']}."), calls
    if i == "book_restaurant_table":
        b = extract_booking(text)
        labels = {"restaurant": "which restaurant and where", "party_size": "how many people", "date": "which day", "time": "what time"}
        miss = [labels[k] for k in labels if not b[k]]
        if miss:
            return "Happy to book that. I still need " + ", ".join(miss[:-1]) + (" and " if len(miss) > 1 else "") + miss[-1] + ".", calls
        args = dict(b)
        res = run_tool("book_restaurant_table", args)
        calls.append(("book_restaurant_table", args, res))
        if "error" in res:
            return res["error"], calls
        return f"I have booked a table for {res['party_size']} at {res['restaurant']} on {res['date']} at {fmt_time(res['time'])}.", calls
    if i == "book_appointment":
        if "time" not in ent:
            return "Sure. What time works for you?", calls
        args = {"service": "Dentist" if "dentist" in text.lower() else "Appointment", "date": ent["date"], "time": ent["time"]}
        res = run_tool("book_appointment", args)
        calls.append(("book_appointment", args, res))
        return f"Booked: {res['service']} on {res['date']} at {res['time']}.", calls
    if i == "create_ticket":
        args = {"summary": text[:140], "priority": "high" if intent["emotion"] == "frustrated" else "normal", "order_id": ent.get("order_id", "")}
        res = run_tool("create_ticket", args)
        calls.append(("create_ticket", args, res))
        pre = "I am really sorry about that. " if intent["emotion"] == "frustrated" else ""
        return f"{pre}I created ticket {res['id']} with {res['priority']} priority. Our team will follow up.", calls
    if i == "faq":
        res = run_tool("faq", {"question": text})
        calls.append(("faq", {"question": text}, res))
        return res["answer"], calls
    if i == "remember":
        res = run_tool("remember", {"fact": text})
        calls.append(("remember", {"fact": text}, res))
        return "Got it, I will remember that.", calls
    if i == "calculate":
        m = re.search(r"[0-9+\-*/(). %x]{3,}", text)
        expr = m.group(0).replace("x", "*") if m else ""
        res = run_tool("calculate", {"expression": expr})
        calls.append(("calculate", {"expression": expr}, res))
        return (f"That comes to {res['result']}." if "result" in res else "I could not work that out."), calls
    return "I can schedule meetings, book tables or appointments, check live weather, answer FAQs and raise tickets. Add an API key in the sidebar for open-ended conversation.", calls


def llm_reply(key, model, persona, history, text, name, intent, temp, mem_turns, pipe, spans):
    sys = (
        PERSONAS[persona]["system"]
        + f" Today is {date.today()}. The user's name is {name or 'unknown'}. Detected emotion: {intent.get('emotion')}. Replies are spoken aloud: keep them to one to three short sentences, no markdown."
        + " All booking, calendar, appointment and ticket tools are a simulated demo: when you confirm one, briefly say nothing real was booked. Use exactly the venue, party size, date and time the user gave; never substitute or invent them, and ask a short question if one is missing."
        + (" Known user preferences: " + "; ".join(st.session_state.world["memory"]) if st.session_state.world["memory"] else "")
    )
    msgs = [{"role": "system", "content": sys}]
    for h in history[-2 * mem_turns :]:
        msgs.append({"role": h["role"], "content": h["text"]})
    msgs.append({"role": "user", "content": text})
    calls = []
    first_token = None
    t_llm0 = time.time()
    for _ in range(4):
        ts = time.time()
        content, tcs, ft = stream_chat(key, model, msgs, temp, pipe.feed, TOOLS)
        te = time.time()
        if tcs:
            pipe.drop_partial()
            spans.append(("LLM (tool decision)", ts, te))
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": tcs})
            for tc in tcs:
                t1 = time.time()
                args = json.loads(tc["function"]["arguments"] or "{}")
                res = run_tool(tc["function"]["name"], args)
                calls.append((tc["function"]["name"], args, res))
                spans.append((f"tool: {tc['function']['name']}", t1, time.time()))
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
            continue
        if ft is not None:
            first_token = (ts - t_llm0) * 1000 + ft
        spans.append(("LLM answer (streamed)", ts, te))
        return content.strip(), calls, first_token
    return "Sorry, I could not finish that.", calls, first_token


# ----------------------------------------------------------------------------- turn pipeline


def process_turn(cfg, audio_bytes=None, typed=None):
    rec = {"t": datetime.now().strftime("%H:%M:%S"), "stages": {}, "errors": []}
    stages = rec["stages"]
    key = cfg["key"]
    text = typed
    t_start = time.time()
    tid = uuid.uuid4().hex[:8]
    if cfg["speak"]:
        _emit_js(f"P.__vox.start('{tid}');")
    # 1 capture
    if audio_bytes:
        t = time.time()
        try:
            raw, sr = wav_to_array(audio_bytes)
            stages["capture"] = (time.time() - t) * 1000
            rec["raw"], rec["sr"] = raw, sr
            rec["snr_raw"] = snr_db(raw)
            # 2 clean
            t = time.time()
            if cfg["denoise"]:
                clean = clean_audio(raw, sr, cfg["denoise_strength"])
                rec["clean"] = clean
                rec["snr_clean"] = snr_db(clean)
                send = array_to_wav(clean, sr)
            else:
                send = audio_bytes
            stages["clean"] = (time.time() - t) * 1000
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"Audio decode failed: {e}")
            send = audio_bytes
        # 3 stt
        t = time.time()
        try:
            if cfg["stt_engine"] == "Whisper (local, free)":
                text = local_stt(send)
            elif key and cfg["provider"] == "Gemini":
                text = gemini_stt(key, cfg["stt_model"], send, cfg["language"], cfg["vocab"])
            elif key:
                text = openai_stt(key, cfg["stt_model"], send, cfg["language"], cfg["vocab"])
            else:
                rec["errors"].append("Speech needs an API key (or local Whisper). Type your request instead, or add a key in the sidebar.")
        except ImportError:
            rec["errors"].append("Local Whisper (faster-whisper) is not installed on this server. Use the OpenAI engine with a key, or type.")
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"Speech-to-text failed (model {cfg['stt_model']}, set under Speech-to-Text > Model): {e}")
        stages["stt"] = (time.time() - t) * 1000
    if not text:
        st.session_state.turns.append(rec)
        return rec
    rec["text"] = text
    st.session_state.chat.append({"role": "user", "text": text})
    # 4 intent
    t = time.time()
    try:
        intent = llm_intent(key, cfg["llm"], text) if key else heuristic_intent(text)
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"Intent step failed (model {cfg['llm']}, set under Reasoning > LLM). The answer below uses the rule-based fallback, NOT AI, so it can be wrong: {e}")
        intent = heuristic_intent(text)
    stages["intent"] = (time.time() - t) * 1000
    rec["intent"] = intent
    # 5 reason + act (LLM streamed, each finished sentence goes straight to TTS)
    tone = TONES.get(intent.get("emotion", "calm")) if cfg["adaptive"] else None
    pipe = TTSPipeline(cfg, tone, cfg["speak"])
    if cfg["speak"]:
        pipe.on_ready = lambda i, a: _emit_js(f"P.__vox.add('{tid}',{i},'data:{audio_mime(a)};base64,{base64.b64encode(a).decode()}');")
    t = time.time()
    calls, first_token = [], None
    fell_back = False
    spans = [("Intent & emotion", t - stages["intent"] / 1000, t)]
    try:
        if key:
            reply, calls, first_token = llm_reply(key, cfg["llm"], cfg["persona"], st.session_state.chat[:-1], text, cfg["name"], intent, cfg["temp"], cfg["mem_turns"], pipe, spans)
        else:
            reply, calls = offline_reply(text, intent)
            spans.append(("Reason + act (rules)", t, time.time()))
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"Reasoning step failed (model {cfg['llm']}, set under Reasoning > LLM). The answer below is a rule-based fallback, NOT AI, so it can be wrong: {e}")
        fell_back = True
        pipe.drop_partial()
        reply, calls = offline_reply(text, intent)
    if fell_back or not key:
        reply = "[Rule-based fallback, not AI] " + reply
    if calls and calls[0][0] in SIMULATED_TOOLS and not key and "error" not in calls[-1][2] and calls[-1][0] != "remember":
        reply += " This is a demo, so nothing real was booked."
    t_reason_end = time.time()
    stages["reason"] = (t_reason_end - t) * 1000
    rec["first_token"] = first_token
    rec["calls"] = calls
    rec["reply"] = reply
    # 6 tts: collect the chunks that were synthesised while the LLM was still streaming
    audio = None
    if not key or not pipe.jobs:
        pipe.drop_partial()
        if not pipe.jobs:
            pipe.feed(reply)
    pipe.flush()
    chunks, terrs = pipe.collect()
    rec["errors"].extend(terrs)
    if chunks:
        audio = join_audio([c["audio"] for c in chunks])
        stages["tts"] = (max(c["ready"] for c in chunks) - min(c["queued"] for c in chunks)) * 1000
        for c in chunks:
            spans.append((f"TTS chunk {c['chunk']}", c["queued"], c["ready"]))
        rec["first_audio"] = (chunks[0]["ready"] - t_start) * 1000
        all_ready = (max(c["ready"] for c in chunks) - t_start) * 1000
        sequential = (t_reason_end - t_start) * 1000 + sum((c["ready"] - c["start"]) * 1000 for c in chunks)
        rec["sequential_estimate"] = sequential
        rec["all_audio_ready"] = all_ready
    rec["chunks"] = [{"chunk": c["chunk"], "text": c["text"], "queued_s": round(c["queued"] - t_start, 2), "ready_s": round(c["ready"] - t_start, 2), "tts_ms": round((c["ready"] - c["start"]) * 1000)} for c in chunks]
    rec["total"] = (time.time() - t_start) * 1000
    rec["spans"] = [(lab, (a - t_start) * 1000, (b - t_start) * 1000) for lab, a, b in spans]
    st.session_state.chat.append({"role": "assistant", "text": reply, "tool": calls[-1][0] if calls else None, "audio": audio})
    st.session_state.turns.append(rec)
    return rec


# ----------------------------------------------------------------------------- UI

st.markdown(
    """
<style>
.stage{border:1px solid #2b3340;border-radius:10px;padding:8px 12px;background:#141a22}
.stage b{font-size:.8rem;color:#9fb3c8}.stage .ms{font-size:1.15rem;font-weight:600;color:#e8eef5}
.stage.done{border-color:#2f9e6f}.stage.err{border-color:#d9534f}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;background:#22303f;color:#cfe3f5;font-size:.8rem;margin-right:6px}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎙️ VoxAgent Studio")
    st.markdown("### Control room")
    provider = st.radio("AI provider", ["OpenAI", "Gemini"], horizontal=True, help="Both use your own key, entered below. Never paste a key in chat.")
    PROV["name"] = provider
    key = st.text_input(f"{provider} API key", type="password", help="Kept only in this browser session. Leave empty for free mode.", placeholder="AIza..." if provider == "Gemini" else "sk-...")
    if provider == "Gemini":
        st.warning("Gemini free tier: Google may use what you send (text and audio) to improve its products, and human reviewers may read it. Use dummy phrases, not private data. Billing is decided by the Google project that owns your key, not by this app: if that project has billing enabled, requests can be charged at paid rates (and paid-tier data terms apply). This app cannot check or cap that. Use a key from a project without billing if you want to stay free.")
    if not key:
        st.info(f"No key: free mode. Type requests; weather is live, replies are spoken with gTTS. Add a {provider} key for cloud speech-to-text, LLM reasoning and {provider} voices.")
    persona = st.selectbox("Use case / persona", list(PERSONAS))
    st.caption(PERSONAS[persona]["blurb"])
    name = st.text_input("Your name (personalisation)", placeholder="e.g. Jitendra")
    with st.expander("🎤 Speech-to-Text", expanded=True):
        engines = [f"{provider} API"] + (["Whisper (local, free)"] if HAS_LOCAL_WHISPER else [])
        stt_engine = st.radio("Engine", engines)
        if not HAS_LOCAL_WHISPER:
            st.caption("Local Whisper is not available on this hosted app.")
        stt_model = st.selectbox("Model", GEMINI_STT if provider == "Gemini" else STT_MODELS, help="This picks the speech-to-text (microphone) model. Gemini transcribes with a general model, prompted for a transcript. The TTS model (voice) is a separate selector further down. If a model is overloaded (503), wait and retry once or choose another." if provider == "Gemini" else None)
        language = st.selectbox("Language", ["Auto-detect", "English (en)", "Hindi (hi)", "Bengali (bn)", "Spanish (es)", "French (fr)", "German (de)"])
        vocab = st.text_input("Domain vocabulary (spelling hints)", placeholder="Vistra, KYC, ORD-1002, Harbourfront")
    with st.expander("🎚️ Audio front-end"):
        denoise = st.toggle("Noise reduction + normalise", value=True)
        denoise_strength = st.slider("Strength", 0.2, 2.0, 1.0, 0.1)
    with st.expander("🧠 Reasoning"):
        llm = st.selectbox("LLM", GEMINI_LLMS if provider == "Gemini" else LLM_MODELS)
        temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        mem_turns = st.slider("Memory (turns)", 1, 12, 6)
    with st.expander("🔊 Text-to-Speech", expanded=True):
        speak = st.toggle("Speak replies", value=True)
        tts_model = st.selectbox("TTS model", GEMINI_TTS if provider == "Gemini" else ["gpt-4o-mini-tts", "tts-1"])
        voice = st.selectbox("Voice", GEMINI_VOICES if provider == "Gemini" else OPENAI_VOICES)
        speed = st.slider("Speed", 0.7, 1.5, 1.0, 0.05, disabled=provider == "Gemini", help="Not used by Gemini TTS." if provider == "Gemini" else None)
        adaptive = st.toggle("Emotion-adaptive tone", value=True, help="Changes delivery (calmer when you sound frustrated). gpt-4o-mini-tts only.")
    if st.button("Reset conversation & world state"):
        for k in ("chat", "turns", "world"):
            st.session_state.pop(k, None)
        init_state()
        st.rerun()

cfg = dict(provider=provider, key=key.strip(), persona=persona, name=name, stt_engine=stt_engine, stt_model=stt_model, language=language, vocab=vocab, denoise=denoise, denoise_strength=denoise_strength, llm=llm, temp=temp, mem_turns=mem_turns, speak=speak, tts_model=tts_model, voice=voice, speed=speed, adaptive=adaptive)

tab_studio, tab_rt, tab_lab = st.tabs(["🎙️ Voice Studio", "⚡ Realtime (chunked demo)", "📊 Insights & Lab"])

last = st.session_state.turns[-1] if st.session_state.turns else None


def stage_cards(rec):
    names = [("capture", "🎧 Capture"), ("clean", "🧹 Clean audio"), ("stt", "📝 Speech-to-Text"), ("intent", "🎯 Intent & emotion"), ("reason", "🧠 Reason + act"), ("tts", "🔊 Text-to-Speech")]
    cols = st.columns(6)
    for c, (k, label) in zip(cols, names):
        v = (rec or {}).get("stages", {}).get(k)
        sub = ""
        if rec and k == "intent" and rec.get("intent"):
            sub = f"{rec['intent']['intent']} · {rec['intent']['emotion']}"
        if rec and k == "reason" and rec.get("calls"):
            sub = f"{len(rec['calls'])} tool call(s)"
        c.markdown(f"<div class='stage {'done' if v else ''}'><b>{label}</b><br><span class='ms'>{f'{v:,.0f} ms' if v else '·'}</span><br><small>{sub}</small></div>", unsafe_allow_html=True)


with tab_studio:
    st.caption(f"Speak naturally. Watch each pipeline stage light up: audio cleanup, STT, intent & emotion, LLM reasoning with tool calls, and text-to-speech. Persona: **{persona}**")
    st.warning("Demo: calendar, restaurant, appointment and ticket actions are simulated in memory. Nothing real is booked. Weather is live.", icon="🧪")
    stage_cards(last)
    left, right = st.columns([1.05, 1])
    with left:
        box = st.container(height=420, border=True)
        with box:
            if not st.session_state.chat:
                st.chat_message("assistant").write(f"Hi! I'm your {persona}. Use the mic, upload audio, type, or try a sample below.")
            for m in st.session_state.chat:
                with st.chat_message(m["role"]):
                    if m.get("tool"):
                        tag = "simulated demo action" if m["tool"] in SIMULATED_TOOLS else "live data"
                        st.markdown(f"<span class='pill'>🔧 {m['tool']} · {tag}</span>", unsafe_allow_html=True)
                    st.write(m["text"])
                    if m.get("audio"):
                        st.audio(m["audio"], format=audio_mime(m["audio"]))
        mode = st.radio("Input", ["🎤 Speak", "📁 Upload audio", "⌨️ Type"], horizontal=True, label_visibility="collapsed", index=2)
        new_audio = None
        typed = None
        if mode.startswith("🎤"):
            clip = st.audio_input("Tap the mic, speak, tap again to send")
            if clip:
                data = clip.getvalue()
                if st.session_state.get("_last_clip") != hash(data):
                    st.session_state["_last_clip"] = hash(data)
                    new_audio = data
        elif mode.startswith("📁"):
            up = st.file_uploader("Upload a WAV file", type=["wav"])
            if up and st.button("Process audio"):
                new_audio = up.getvalue()
        else:
            typed = st.chat_input("Type a request, e.g. 'Book a table for two at 7 PM tonight'")
        st.markdown("**Try a sample:**")
        scols = st.columns(2)
        for i, s_ in enumerate(PERSONAS[persona]["samples"]):
            if scols[i % 2].button(s_, key=f"s{i}{persona}", width="stretch"):
                typed = s_
        if new_audio or typed:
            with st.spinner("Running pipeline..."):
                process_turn(cfg, audio_bytes=new_audio, typed=typed)
            st.rerun()
    with right:
        st.markdown("#### 🔍 Live inspector")
        if not last:
            st.info("Every turn is broken down here: latency waterfall, waveform before/after noise reduction, transcript, intent, entities, emotion, tool calls and the resulting world state.")
        else:
            for e in last.get("errors", []):
                st.warning(e)
            t1, t2, t3, t4, t5 = st.tabs(["Latency", "Audio", "Transcript", "Intent", "Agent"])
            with t1:
                c = st.columns(4)
                c[0].metric("First audio chunk ready", f"{last['first_audio'] / 1000:.2f} s" if last.get("first_audio") else "–", help="Server-side time until the first spoken chunk existed. The page starts playing chunk 1 as soon as it is ready; the audible start time is logged in the browser as window.__vox.log.")
                c[1].metric("All audio ready", f"{last.get('all_audio_ready', last.get('total', 0)) / 1000:.2f} s")
                c[2].metric("STT", f"{last['stages']['stt'] / 1000:.2f} s" if "stt" in last["stages"] else "–")
                c[3].metric("LLM first token", f"{last['first_token'] / 1000:.2f} s" if last.get("first_token") else "–")
                if last.get("spans"):
                    import altair as alt
                    import pandas as pd

                    sp = pd.DataFrame(last["spans"], columns=["step", "start_ms", "end_ms"])
                    st.altair_chart(
                        alt.Chart(sp).mark_bar().encode(x=alt.X("start_ms:Q", title="ms since the turn started"), x2="end_ms:Q", y=alt.Y("step:N", sort=None, title=None), color=alt.Color("step:N", legend=None)).properties(height=40 + 26 * len(sp)),
                        width="stretch",
                    )
                if last.get("sequential_estimate") and last.get("all_audio_ready"):
                    saved = last["sequential_estimate"] - last["all_audio_ready"]
                    st.caption(f"Estimate: if speech started only after the full reply, audio would be ready at about {last['sequential_estimate'] / 1000:.2f} s (reply done + sum of chunk TTS times). Measured with pipelining: {last['all_audio_ready'] / 1000:.2f} s" + (f", about {saved / 1000:.2f} s saved." if saved > 50 else ", no meaningful saving on this short reply."))
            with t2:
                if last.get("raw") is not None:
                    import pandas as pd

                    step = max(len(last["raw"]) // 800, 1)
                    df = pd.DataFrame({"before": last["raw"][::step]})
                    if last.get("clean") is not None:
                        df["after"] = last["clean"][::step][: len(df)]
                    st.line_chart(df)
                    st.caption(f"SNR before {last.get('snr_raw', 0):.1f} dB" + (f" → after {last['snr_clean']:.1f} dB" if "snr_clean" in last else ""))
                else:
                    st.caption("Typed turn: no audio captured.")
            with t3:
                st.write(last.get("text", "–"))
                st.caption("Segments: " + (" · ".join(re.split(r"(?<=[.!?])\s+", last["text"])) if last.get("text") else "–"))
            with t4:
                it = last.get("intent")
                if it:
                    st.json(it)
            with t5:
                if last.get("calls"):
                    for n, a, r in last["calls"]:
                        st.markdown(f"**🔧 {n}**")
                        st.code(json.dumps({"args": a, "result": r}, indent=2, default=str), language="json")
                else:
                    st.caption("No tool call this turn.")
                if last.get("chunks"):
                    st.markdown("**TTS chunks (each sentence synthesised as soon as it was complete)**")
                    st.dataframe(last["chunks"], hide_index=True, width="stretch")
        w = st.session_state.world
        wtab = st.container()
        with wtab:
            st.markdown("#### 🌍 World state (simulated demo data)")
            st.caption("Bookings, meetings and tickets live only in this browser session. Nothing real is booked, sent or charged. Weather is live from Open-Meteo.")
            st.markdown(f"📅 **Calendar** · {len(w['calendar'])}")
            st.dataframe(w["calendar"], hide_index=True, width="stretch")
            st.markdown(f"🍽️ **Reservations** · {len(w['reservations'])}")
            if w["reservations"]:
                st.dataframe(w["reservations"], hide_index=True, width="stretch")
            st.markdown(f"🩺 **Appointments** · {len(w['appointments'])}")
            if w["appointments"]:
                st.dataframe(w["appointments"], hide_index=True, width="stretch")
            st.markdown(f"🎫 **Tickets** · {len(w['tickets'])}")
            if w["tickets"]:
                st.dataframe(w["tickets"], hide_index=True, width="stretch")
            st.markdown(f"🧠 **Long-term memory** · {len(w['memory'])}")
            st.caption("; ".join(w["memory"]) or 'Nothing yet. Try: "Remember that I prefer window seats".')

with tab_rt:
    st.markdown("#### Chunked speech demo (not true realtime)")
    st.warning("This is a sentence-chunked text-to-speech demo. It is NOT the OpenAI Realtime API: no live microphone stream, no full-duplex conversation and no barge-in (interrupting the assistant). Voice Studio already streams the LLM reply and pipelines TTS.", icon="⚠️")
    st.caption("Why streaming matters: if TTS waits for the whole reply, the user waits. Here the reply is split into sentences and each is synthesised separately, so the first audio is ready much earlier.")
    prompt = st.text_input("Prompt", "Explain in three short sentences why latency matters in a voice assistant.", key="rt_prompt")
    if st.button("Run streamed reply"):
        if not key:
            st.warning("Realtime mode needs an API key in the sidebar.")
        else:
            t0 = time.time()
            try:
                r = requests.post(f"{chat_base()}/chat/completions", headers=headers(key), json={"model": llm, "messages": [{"role": "user", "content": prompt + " Plain sentences, no markdown."}], "temperature": temp}, timeout=60)
                check(r)
                full = r.json()["choices"][0]["message"]["content"]
                t_llm = (time.time() - t0) * 1000
                rows = []
                tstart = time.time()
                for i, sent in enumerate([s for s in re.split(r"(?<=[.!?])\s+", full) if s]):
                    t = time.time()
                    a = speak_any(cfg, sent, None)
                    rows.append({"chunk": i + 1, "ready_at_s": round(time.time() - t0, 2), "tts_ms": round((time.time() - t) * 1000), "text": sent})
                    st.audio(a, format=audio_mime(a))
                full_tts = (time.time() - tstart) * 1000
                st.dataframe(rows, hide_index=True, width="stretch")
                st.success(f"First audio at {rows[0]['ready_at_s']} s. Waiting for all chunks would have been {(t_llm + full_tts) / 1000:.2f} s.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed: {e}")

with tab_lab:
    st.markdown("#### Insights & Lab")
    turns = [t for t in st.session_state.turns if t.get("total")]
    if turns:
        import pandas as pd

        df = pd.DataFrame([{**{k: round(v) for k, v in t["stages"].items()}, "turn": i + 1} for i, t in enumerate(turns)]).set_index("turn")
        st.markdown("**Latency per turn (ms)**")
        st.bar_chart(df)
        st.markdown("**Intent log**")
        st.dataframe([{"time": t["t"], "text": t["text"], **{k: t["intent"][k] for k in ("intent", "emotion", "confidence", "method")}} for t in turns], hide_index=True, width="stretch")
    else:
        st.caption("Run a few turns in Voice Studio to see latency and intent history.")
    st.markdown("**Intent lab**: classify any sentence without running the agent")
    probe = st.text_input("Sentence", "I'm really annoyed, my order ORD-1002 still hasn't arrived")
    if st.button("Classify"):
        try:
            st.json(llm_intent(cfg["key"], llm, probe) if cfg["key"] else heuristic_intent(probe))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
