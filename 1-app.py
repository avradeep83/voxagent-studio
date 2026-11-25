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
import io
import json
import re
import time
import wave
from datetime import date, datetime, timedelta

import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="VoxAgent Studio", page_icon="🎙️", layout="wide")

OPENAI = "https://api.openai.com/v1"

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


def headers(key):
    return {"Authorization": f"Bearer {key}"}


def openai_stt(key, model, wav_bytes, language, vocab):
    data = {"model": model}
    if language and language != "Auto-detect":
        data["language"] = language.split("(")[-1].strip(")")
    if vocab:
        data["prompt"] = "Domain vocabulary: " + vocab
    r = requests.post(f"{OPENAI}/audio/transcriptions", headers=headers(key), data=data, files={"file": ("speech.wav", wav_bytes, "audio/wav")}, timeout=90)
    r.raise_for_status()
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
    r.raise_for_status()
    return r.content


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
        if name in t:
            d += timedelta(days=(i - d.weekday()) % 7 or 7)
            break
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t)
    tm = None
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
        tm = f"{h:02d}:{int(m.group(2) or 0):02d}"
    return str(d), tm


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
        row = {"id": f"RES-{len(w['reservations']) + 1:03d}", "restaurant": args.get("restaurant", "Trattoria Lucca (Italian)"), "party_size": int(args.get("party_size", 2)), "date": args.get("date", str(date.today())), "time": args.get("time", "19:00")}
        w["reservations"].append(row)
        return {"status": "booked", **row}
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
        if re.fullmatch(r"[0-9+\-*/(). %]+", expr):
            try:
                return {"result": eval(expr, {"__builtins__": {}}, {})}  # noqa: S307 (charset-restricted)
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}
        return {"error": "Unsupported expression"}
    return {"error": f"Unknown tool {name}"}


def fn(name, desc, props, req=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": list(req)}}}


S = {"type": "string"}
TOOLS = [
    fn("get_weather", "Live weather for a city.", {"city": S}, ["city"]),
    fn("schedule_meeting", "Add a calendar meeting. date is YYYY-MM-DD, time is 24h HH:MM.", {"title": S, "date": S, "time": S, "duration_minutes": {"type": "integer"}, "attendees": S}, ["title", "date", "time"]),
    fn("book_restaurant_table", "Reserve a restaurant table.", {"restaurant": S, "party_size": {"type": "integer"}, "date": S, "time": S}, ["party_size", "time"]),
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
    m = re.search(r"weather (?:in|at|for) ([a-zA-Z ]+?)(?: right| today| now|\?|$|,)", t)
    if m:
        ent["city"] = m.group(1).strip().title()
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
    r = requests.post(f"{OPENAI}/chat/completions", headers=headers(key), json={"model": model, "messages": msgs, "response_format": {"type": "json_object"}, "temperature": 0}, timeout=60)
    r.raise_for_status()
    out = json.loads(r.json()["choices"][0]["message"]["content"])
    out["method"] = "llm"
    return out


def offline_reply(text, intent):
    """No-LLM agent: choose a tool from the rule-based intent, run it, template the reply."""
    ent = intent["entities"]
    i = intent["intent"]
    calls = []
    if i == "get_weather":
        res = run_tool("get_weather", {"city": ent.get("city", "Mumbai")})
        calls.append(("get_weather", {"city": ent.get("city", "Mumbai")}, res))
        if "error" in res:
            return "I could not reach the weather service just now. Please try again in a moment.", calls
        rain = res.get("rain_chance_today_pct")
        umb = "" if rain is None else (" There is a decent chance of rain today, so carry an umbrella." if rain >= 40 or res["precipitation_mm"] > 0 else " There is no real chance of rain today, so you do not need an umbrella.")
        return f"The weather in {res['city']} right now is {res['condition']} at {res['temperature_c']} degrees Celsius.{umb}", calls
    if i == "schedule_meeting":
        if "time" not in ent:
            return "Sure. What time should I schedule it?", calls
        who = re.search(r"with ([A-Z][a-z]+(?: [A-Z][a-z]+)?|the vendor)", text)
        args = {"title": f"Call with {who.group(1)}" if who else "Meeting", "date": ent["date"], "time": ent["time"], "attendees": who.group(1) if who else ""}
        res = run_tool("schedule_meeting", args)
        calls.append(("schedule_meeting", args, res))
        return (res["error"] if "error" in res else f"Done. I scheduled {args['title']} on {res['date']} at {res['time']}."), calls
    if i == "book_restaurant_table":
        if "time" not in ent:
            return "Happy to book it. What time, and for how many people?", calls
        args = {"party_size": ent.get("party_size", 2), "date": ent["date"], "time": ent["time"], "restaurant": "Trattoria Lucca (Italian)" if "italian" in text.lower() else "Harbourfront Grill"}
        res = run_tool("book_restaurant_table", args)
        calls.append(("book_restaurant_table", args, res))
        return f"I have booked a table for {res['party_size']} at {res['time']} at {res['restaurant']}. Let me know if you need anything else.", calls
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
    return "I can schedule meetings, book tables or appointments, check live weather, answer FAQs and raise tickets. Add an OpenAI key in the sidebar for open-ended conversation.", calls


def llm_reply(key, model, persona, history, text, name, intent, temp, mem_turns):
    sys = (
        PERSONAS[persona]["system"]
        + f" Today is {date.today()}. The user's name is {name or 'unknown'}. Detected emotion: {intent.get('emotion')}. Replies are spoken aloud: keep them to one to three short sentences, no markdown."
        + (" Known user preferences: " + "; ".join(st.session_state.world["memory"]) if st.session_state.world["memory"] else "")
    )
    msgs = [{"role": "system", "content": sys}]
    for h in history[-2 * mem_turns :]:
        msgs.append({"role": h["role"], "content": h["text"]})
    msgs.append({"role": "user", "content": text})
    calls = []
    first_token = None
    t0 = time.time()
    for _ in range(4):
        r = requests.post(f"{OPENAI}/chat/completions", headers=headers(key), json={"model": model, "messages": msgs, "tools": TOOLS, "temperature": temp}, timeout=90)
        r.raise_for_status()
        if first_token is None:
            first_token = (time.time() - t0) * 1000
        m = r.json()["choices"][0]["message"]
        if m.get("tool_calls"):
            msgs.append(m)
            for tc in m["tool_calls"]:
                args = json.loads(tc["function"]["arguments"] or "{}")
                res = run_tool(tc["function"]["name"], args)
                calls.append((tc["function"]["name"], args, res))
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
            continue
        return (m.get("content") or "").strip(), calls, first_token
    return "Sorry, I could not finish that.", calls, first_token


# ----------------------------------------------------------------------------- turn pipeline


def process_turn(cfg, audio_bytes=None, typed=None):
    rec = {"t": datetime.now().strftime("%H:%M:%S"), "stages": {}, "errors": []}
    stages = rec["stages"]
    key = cfg["key"]
    text = typed
    t_start = time.time()
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
            elif key:
                text = openai_stt(key, cfg["stt_model"], send, cfg["language"], cfg["vocab"])
            else:
                rec["errors"].append("Speech needs an OpenAI key (or local Whisper). Type your request instead, or add a key in the sidebar.")
        except ImportError:
            rec["errors"].append("Local Whisper (faster-whisper) is not installed on this server. Use the OpenAI engine with a key, or type.")
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"Speech-to-text failed: {e}")
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
        rec["errors"].append(f"LLM intent failed, used rules: {e}")
        intent = heuristic_intent(text)
    stages["intent"] = (time.time() - t) * 1000
    rec["intent"] = intent
    # 5 reason + act
    t = time.time()
    calls, first_token = [], None
    try:
        if key:
            reply, calls, first_token = llm_reply(key, cfg["llm"], cfg["persona"], st.session_state.chat[:-1], text, cfg["name"], intent, cfg["temp"], cfg["mem_turns"])
        else:
            reply, calls = offline_reply(text, intent)
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"Reasoning failed: {e}")
        reply, calls = offline_reply(text, intent)
    stages["reason"] = (time.time() - t) * 1000
    rec["first_token"] = first_token
    rec["calls"] = calls
    rec["reply"] = reply
    # 6 tts
    audio = None
    if cfg["speak"] and reply:
        t = time.time()
        try:
            tone = TONES.get(intent.get("emotion", "calm")) if cfg["adaptive"] else None
            if key:
                audio = openai_tts(key, cfg["tts_model"], cfg["voice"], cfg["speed"], reply, tone)
            else:
                audio = gtts_tts(reply)
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"Text-to-speech failed: {e}")
        stages["tts"] = (time.time() - t) * 1000
    rec["total"] = (time.time() - t_start) * 1000
    rec["first_audio"] = rec["total"] if audio else None
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
    key = st.text_input("OpenAI API key", type="password", help="Kept only in this browser session. Leave empty for free mode.", placeholder="sk-...")
    if not key:
        st.info("No key: free mode. Type requests; weather is live, replies are spoken with gTTS. Add a key for cloud speech-to-text, LLM reasoning and OpenAI voices.")
    persona = st.selectbox("Use case / persona", list(PERSONAS))
    st.caption(PERSONAS[persona]["blurb"])
    name = st.text_input("Your name (personalisation)", placeholder="e.g. Jitendra")
    with st.expander("🎤 Speech-to-Text", expanded=True):
        stt_engine = st.radio("Engine", ["OpenAI API", "Whisper (local, free)"], help="Local Whisper needs the optional faster-whisper package; not installed on the hosted demo.")
        stt_model = st.selectbox("Model", STT_MODELS)
        language = st.selectbox("Language", ["Auto-detect", "English (en)", "Hindi (hi)", "Bengali (bn)", "Spanish (es)", "French (fr)", "German (de)"])
        vocab = st.text_input("Domain vocabulary (spelling hints)", placeholder="Vistra, KYC, ORD-1002, Harbourfront")
    with st.expander("🎚️ Audio front-end"):
        denoise = st.toggle("Noise reduction + normalise", value=True)
        denoise_strength = st.slider("Strength", 0.2, 2.0, 1.0, 0.1)
    with st.expander("🧠 Reasoning"):
        llm = st.selectbox("LLM", LLM_MODELS)
        temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        mem_turns = st.slider("Memory (turns)", 1, 12, 6)
    with st.expander("🔊 Text-to-Speech", expanded=True):
        speak = st.toggle("Speak replies", value=True)
        tts_model = st.selectbox("TTS model", ["gpt-4o-mini-tts", "tts-1"])
        voice = st.selectbox("Voice", OPENAI_VOICES)
        speed = st.slider("Speed", 0.7, 1.5, 1.0, 0.05)
        adaptive = st.toggle("Emotion-adaptive tone", value=True, help="Changes delivery (calmer when you sound frustrated). gpt-4o-mini-tts only.")
    if st.button("Reset conversation & world state"):
        for k in ("chat", "turns", "world"):
            st.session_state.pop(k, None)
        init_state()
        st.rerun()

cfg = dict(key=key.strip(), persona=persona, name=name, stt_engine=stt_engine, stt_model=stt_model, language=language, vocab=vocab, denoise=denoise, denoise_strength=denoise_strength, llm=llm, temp=temp, mem_turns=mem_turns, speak=speak, tts_model=tts_model, voice=voice, speed=speed, adaptive=adaptive)

tab_studio, tab_rt, tab_lab = st.tabs(["🎙️ Voice Studio", "⚡ Realtime Voice", "📊 Insights & Lab"])

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
                        st.markdown(f"<span class='pill'>🔧 {m['tool']}</span>", unsafe_allow_html=True)
                    st.write(m["text"])
                    if m.get("audio"):
                        st.audio(m["audio"], format="audio/mp3")
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
                c[0].metric("First audio ready", f"{last['first_audio'] / 1000:.2f} s" if last.get("first_audio") else "–")
                c[1].metric("End-to-end", f"{last.get('total', 0) / 1000:.2f} s")
                c[2].metric("STT", f"{last['stages']['stt'] / 1000:.2f} s" if "stt" in last["stages"] else "–")
                c[3].metric("LLM first token", f"{last['first_token'] / 1000:.2f} s" if last.get("first_token") else "–")
                import pandas as pd

                wf = pd.DataFrame({"stage": [k for k in last["stages"]], "ms": [round(v) for v in last["stages"].values()]}).set_index("stage")
                st.bar_chart(wf, horizontal=True)
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
        w = st.session_state.world
        wtab = st.container()
        with wtab:
            st.markdown("#### 🌍 World state")
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
    st.markdown("#### Realtime voice: streamed, sentence-by-sentence speech")
    st.caption("Why streaming matters: if TTS waits for the whole reply, the user waits. Here the reply is split into sentences and each is synthesised separately, so the first audio is ready much earlier.")
    prompt = st.text_input("Prompt", "Explain in three short sentences why latency matters in a voice assistant.", key="rt_prompt")
    if st.button("Run streamed reply"):
        if not key:
            st.warning("Realtime mode needs an OpenAI key in the sidebar.")
        else:
            t0 = time.time()
            try:
                r = requests.post(f"{OPENAI}/chat/completions", headers=headers(key), json={"model": llm, "messages": [{"role": "user", "content": prompt + " Plain sentences, no markdown."}], "temperature": temp}, timeout=60)
                r.raise_for_status()
                full = r.json()["choices"][0]["message"]["content"]
                t_llm = (time.time() - t0) * 1000
                rows = []
                tstart = time.time()
                for i, sent in enumerate([s for s in re.split(r"(?<=[.!?])\s+", full) if s]):
                    t = time.time()
                    a = openai_tts(key, tts_model, voice, speed, sent, None)
                    rows.append({"chunk": i + 1, "ready_at_s": round(time.time() - t0, 2), "tts_ms": round((time.time() - t) * 1000), "text": sent})
                    st.audio(a, format="audio/mp3")
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
