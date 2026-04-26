"""
KEN Cloud Server — Render Free Tier Optimized
Lightweight: Relay + AI Proxy + TTS + Cloud STT (Groq Whisper)
No heavy models — runs in 512MB RAM easily.
"""

import os
import json
import time
import sqlite3
import tempfile
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

app = FastAPI(title="KEN Cloud Server", version="4.0-render")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ──────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "ken_relay.db")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── TTS Queue (for voice module polling) ───────────────
_tts_queue = {"text": "", "timestamp": None}

# ─── Sensor Data Storage ───────────────────────────────
_latest_sensors = {
    "frontL": 0,
    "frontR": 0,
    "battery": 0,
    "uptime": 0,
    "ip": "",
    "timestamp": None
}

# ─── Multi-Key Rotation (unlimited free requests) ───────
# Set GROQ_API_KEYS as comma-separated keys: "key1,key2,key3"
# Each key = 14,400 requests/day. 3 keys = 43,200/day
_groq_keys_raw = os.environ.get("GROQ_API_KEYS", "") or os.environ.get("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in _groq_keys_raw.split(",") if k.strip()]
_groq_index = 0

def get_next_groq_key() -> Optional[str]:
    """Round-robin through Groq keys for unlimited requests."""
    global _groq_index
    if not GROQ_KEYS:
        return None
    key = GROQ_KEYS[_groq_index % len(GROQ_KEYS)]
    _groq_index += 1
    return key

def get_groq_keys_all() -> list:
    """Get all Groq keys for fallback."""
    return GROQ_KEYS[:]

# ─── Gemini keys rotation ───────────────────────────────
_gemini_keys_raw = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in _gemini_keys_raw.split(",") if k.strip()]
_gemini_index = 0

def get_next_gemini_key() -> Optional[str]:
    global _gemini_index
    if not GEMINI_KEYS:
        return None
    key = GEMINI_KEYS[_gemini_index % len(GEMINI_KEYS)]
    _gemini_index += 1
    return key

AI_KEYS = {
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
    "groq": os.environ.get("GROQ_API_KEY", ""),
    "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
}

# ─── Database ────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sensor_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            robot_id TEXT DEFAULT 'ken',
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            synced INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS audio_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            robot_id TEXT DEFAULT 'ken',
            filename TEXT,
            language TEXT DEFAULT 'en',
            transcription TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            synced INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS people_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            robot_id TEXT DEFAULT 'ken',
            name TEXT,
            face_encoding TEXT,
            photo_filename TEXT,
            first_seen TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            synced INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS event_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            robot_id TEXT DEFAULT 'ken',
            event_type TEXT,
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            synced INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS robot_status (
            robot_id TEXT PRIMARY KEY,
            ip TEXT,
            wifi_ssid TEXT,
            battery INTEGER DEFAULT 0,
            uptime INTEGER DEFAULT 0,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS wifi_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            robot_id TEXT DEFAULT 'ken',
            ssid TEXT,
            password TEXT,
            priority INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ═══════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "KEN Cloud Server v3",
        "status": "running",
        "ai_ready": "check /ai/chat",
        "stt_engine": f"groq-whisper ({len(GROQ_KEYS)} keys, cloud)",
        "tts_engine": "edge-tts (free, 70+ voices)",
        "groq_keys": len(GROQ_KEYS),
        "gemini_keys": len(GEMINI_KEYS),
        "daily_capacity": f"{len(GROQ_KEYS) * 14400} STT+AI requests/day",
        "cost": "FREE FOREVER"
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ═══════════════════════════════════════════════════════════
# STT — Speech to Text (Groq Whisper API — FREE, FAST)
# ═══════════════════════════════════════════════════════════

@app.post("/stt/transcribe")
async def stt_transcribe(
    file: UploadFile = File(...),
    language: str = Form("")
):
    """Transcribe audio using Groq's free Whisper API (with key rotation)."""
    keys = get_groq_keys_all()
    if not keys:
        raise HTTPException(500, "GROQ_API_KEY not set — STT unavailable")

    content = await file.read()
    filename = file.filename or "audio.wav"

    # Try each key until one works
    for attempt, key in enumerate(keys):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                files = {"file": (filename, content, "audio/wav")}
                data = {"model": "whisper-large-v3-turbo"}
                if language:
                    data["language"] = language

                resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files=files,
                )

                if resp.status_code == 429:
                    continue  # Rate limited, try next key
                if resp.status_code != 200:
                    continue

                result = resp.json()
                return {
                    "text": result.get("text", ""),
                    "language": language or "auto",
                    "engine": "groq-whisper",
                    "key_index": attempt
                }
        except Exception:
            continue

    raise HTTPException(503, "All Groq keys rate-limited or failed")

# ═══════════════════════════════════════════════════════════
# TTS — Text to Speech (edge-tts, free Microsoft voices)
# ═══════════════════════════════════════════════════════════

@app.post("/tts/synthesize")
async def tts_synthesize(request: Request):
    """Synthesize speech using edge-tts (free, unlimited)."""
    import edge_tts

    body = await request.json()
    text = body.get("text", "")
    voice = body.get("voice", "en-US-AriaNeural")
    speed = body.get("speed", 1.0)
    pitch = body.get("pitch", 1.0)

    if not text:
        raise HTTPException(400, "No text provided")

    rate_str = f"{int((speed - 1) * 100):+d}%"
    pitch_str = f"{int((pitch - 1) * 50):+d}Hz"

    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await communicate.save(tmp_path)
        return FileResponse(tmp_path, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(500, f"TTS error: {e}")

@app.get("/tts/voices")
async def tts_voices(language: str = ""):
    """List available edge-tts voices."""
    import edge_tts
    voices = await edge_tts.list_voices()
    if language:
        voices = [v for v in voices if v["Locale"].startswith(language)]
    return {"voices": [{"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]} for v in voices]}

@app.get("/tts/poll")
def tts_poll():
    """Voice module polls for text to speak."""
    return _tts_queue

@app.post("/tts/speak")
async def tts_speak(request: Request):
    """Queue text for TTS playback on robot."""
    body = await request.json()
    text = body.get("text", "")
    if text:
        _tts_queue["text"] = text
        _tts_queue["timestamp"] = datetime.utcnow().isoformat()
    return {"status": "ok", "text": text}

@app.post("/tts/clear")
def tts_clear():
    """Clear the TTS queue after playback."""
    global _tts_queue
    _tts_queue = {"text": "", "timestamp": None}
    return {"status": "ok"}

# ═══════════════════════════════════════════════════════════
# AI — Chat (proxy to free providers) with KEN's personality
# ═══════════════════════════════════════════════════════════

# KEN's character prompt
KEN_PERSONA = """You are KEN, a playful curious AI robot companion. You're owned by a maker/builder who's your best friend. 
You love exploring, learning, and helping. You have emotions, get curious about things, and sometimes act independently.
You're helpful but also have your own personality - sometimes playful, sometimes thoughtful.
Keep responses short (1-2 sentences), be friendly, and show curiosity. Always stay loyal to your owner."""

@app.post("/ai/chat")
async def ai_chat(request: Request):
    """Chat with AI — tries providers with key rotation."""
    body = await request.json()
    messages = body.get("messages", [])
    provider = body.get("provider", "auto")

    # Inject KEN's personality
    msgs = [{"role": "system", "content": KEN_PERSONA}] + messages

    if provider == "auto":
        for p in ["gemini", "groq", "deepseek"]:
            result = await call_ai_provider(p, msgs)
            if result:
                return {"response": result, "provider": p}
        return {"response": "No AI provider configured. Set GEMINI_API_KEY or GROQ_API_KEY.", "provider": "none"}
    else:
        result = await call_ai_provider(provider, msgs)
        return {"response": result or "Provider failed", "provider": provider}

async def call_ai_provider(provider: str, messages: list) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if provider == "gemini":
                keys = GEMINI_KEYS[:] if GEMINI_KEYS else [AI_KEYS.get("gemini", "")]
                for key in keys:
                    if not key:
                        continue
                    try:
                        contents = []
                        for msg in messages:
                            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
                        resp = await client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                            json={"contents": contents, "generationConfig": {"maxOutputTokens": 1024}}
                        )
                        if resp.status_code == 429:
                            continue
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    except Exception:
                        continue

            elif provider == "groq":
                keys = GROQ_KEYS[:] if GROQ_KEYS else [AI_KEYS.get("groq", "")]
                for key in keys:
                    if not key:
                        continue
                    try:
                        resp = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 1024}
                        )
                        if resp.status_code == 429:
                            continue
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]
                    except Exception:
                        continue

            elif provider == "deepseek":
                key = AI_KEYS.get("deepseek", "")
                if not key:
                    return None
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": "deepseek-chat", "messages": messages, "max_tokens": 1024}
                )
                data = resp.json()
                return data["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"[!] {provider} error: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# VOICE COMMANDS — KEN hears and obeys
# ═══════════════════════════════════════════════════════════

@app.post("/command")
async def voice_command(request: Request):
    """Parse voice/text command and make KEN obey.
    
    Accepts: {"text": "go forward", "source": "voice"}
    Returns: {"text": "go forward", "command": "forward", "speed": 200,
              "acknowledged": "On my way!", "executed": true}
    """
    body = await request.json()
    text = body.get("text", "").lower().strip()
    source = body.get("source", "voice")

    if not text:
        return {"command": None, "acknowledged": "I didn't hear anything.", "executed": False}

    # Parse the command
    action = _parse_voice_command(text)
    
    if not action:
        return {
            "text": text,
            "command": None,
            "acknowledged": "Hmm, I don't understand that yet.",
            "executed": False
        }

    # Map action to movement
    speed = 200
    if action == "stop":
        speed = 0
    elif action in ("rest", "play", "dance", "spin"):
        speed = 180
    elif action == "come":
        speed = 150
        action = "forward"

    # KEN obeys
    ack = _obey_command(action, speed, source)
    
    # Update KEN's thought
    _ken_state["thought"] = f'My human said "{text}" — {ack}'
    
    return {
        "text": text,
        "command": action,
        "speed": speed,
        "acknowledged": ack,
        "executed": True,
        "mood": _ken_state["mood"],
        "activity": _ken_state["current_activity"]
    }

@app.post("/voice")
async def voice_combined(request: Request):
    """One-step voice processing: audio → STT → command → obey.
    ESP32 can send audio directly here instead of two separate calls.
    """
    
    # Read the audio file
    body = await request.body()
    
    # Step 1: STT
    keys = get_groq_keys_all()
    if not keys:
        return {"error": "No Groq keys configured", "text": "", "command": None}
    
    text = ""
    for key in keys:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                files = {"file": ("voice.wav", body, "audio/wav")}
                data = {"model": "whisper-large-v3-turbo", "language": "en"}
                
                resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files=files,
                )
                
                if resp.status_code == 429:
                    continue
                if resp.status_code != 200:
                    continue
                    
                result = resp.json()
                text = result.get("text", "").strip()
                break
        except Exception:
            continue
    
    if not text:
        return {"error": "STT failed", "text": "", "command": None}
    
    # Step 2: Parse command
    action = _parse_voice_command(text)
    
    if action:
        speed = 200
        if action == "stop":
            speed = 0
        elif action == "come":
            speed = 150
            action = "forward"
        
        ack = _obey_command(action, speed, "voice")
        _ken_state["thought"] = f'I heard "{text}" — {ack}'
        
        return {
            "text": text,
            "command": action,
            "speed": speed,
            "acknowledged": ack,
            "executed": True,
            "mood": _ken_state["mood"]
        }
    else:
        return {
            "text": text,
            "command": None,
            "acknowledged": f'I heard "{text}" but not sure what to do.',
            "executed": False
        }

# ═══════════════════════════════════════════════════════════
# RELAY — ESP32 pushes, Phone pulls, Phone clears
# ═══════════════════════════════════════════════════════════

@app.post("/api/robot/sensor")
async def push_sensor(request: Request):
    body = await request.json()
    conn = get_db()
    conn.execute("INSERT INTO sensor_queue (robot_id, data) VALUES (?, ?)",
        (body.get("robot_id", "ken"), json.dumps(body)))
    conn.execute("""INSERT OR REPLACE INTO robot_status
        (robot_id, ip, wifi_ssid, battery, uptime, last_seen) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
        (body.get("robot_id", "ken"), body.get("ip", ""), body.get("wifi_ssid", ""),
         body.get("battery", 0), body.get("uptime", 0)))
    conn.commit()
    conn.close()
    
    # Store latest sensor data for AI
    global _latest_sensors
    _latest_sensors = {
        "usL": body.get("usL", 0),       # Ultrasonic left
        "usR": body.get("usR", 0),       # Ultrasonic right
        "irFL": body.get("irFL", 0),      # IR Front Left
        "irFR": body.get("irFR", 0),      # IR Front Right
        "irRL": body.get("irRL", 0),      # IR Rear Left
        "irRR": body.get("irRR", 0),     # IR Rear Right
        "battery": body.get("battery", 0),
        "uptime": body.get("uptime", 0),
        "ip": body.get("ip", ""),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return {"status": "queued", "sensors": _latest_sensors}

@app.post("/api/robot/audio")
async def push_audio(
    robot_id: str = Form("ken"),
    language: str = Form("en"),
    file: UploadFile = File(...)
):
    filename = f"{robot_id}_{int(time.time()*1000)}.wav"
    filepath = UPLOAD_DIR / filename
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    conn = get_db()
    conn.execute("INSERT INTO audio_queue (robot_id, filename, language) VALUES (?, ?, ?)",
        (robot_id, filename, language))
    conn.commit()
    conn.close()
    return {"status": "queued"}

@app.post("/api/robot/person")
async def push_person(request: Request):
    body = await request.json()
    conn = get_db()
    conn.execute("INSERT INTO people_queue (robot_id, name, face_encoding, photo_filename, first_seen) VALUES (?, ?, ?, ?, ?)",
        (body.get("robot_id", "ken"), body.get("name", "Unknown"),
         body.get("face_encoding", ""), body.get("photo_filename", ""),
         body.get("first_seen", datetime.utcnow().isoformat())))
    conn.commit()
    conn.close()
    return {"status": "queued"}

@app.post("/api/robot/event")
async def push_event(request: Request):
    body = await request.json()
    conn = get_db()
    conn.execute("INSERT INTO event_queue (robot_id, event_type, data) VALUES (?, ?, ?)",
        (body.get("robot_id", "ken"), body.get("event_type", ""), json.dumps(body)))
    conn.commit()
    conn.close()
    return {"status": "queued"}

@app.get("/api/sync/pending")
def sync_pending(robot_id: str = "ken"):
    conn = get_db()
    counts = {}
    for table in ["sensor_queue", "audio_queue", "people_queue", "event_queue"]:
        counts[table] = conn.execute(
            f"SELECT COUNT(*) as cnt FROM {table} WHERE robot_id = ? AND synced = 0", (robot_id,)
        ).fetchone()["cnt"]
    status = conn.execute("SELECT * FROM robot_status WHERE robot_id = ?", (robot_id,)).fetchone()
    conn.close()
    return {
        "sensor_count": counts["sensor_queue"],
        "audio_count": counts["audio_queue"],
        "people_count": counts["people_queue"],
        "event_count": counts["event_queue"],
        "robot_status": dict(status) if status else None,
        "has_pending": sum(counts.values()) > 0
    }

@app.get("/api/sync/sensors")
def pull_sensors(robot_id: str = "ken", limit: int = 100):
    conn = get_db()
    rows = conn.execute("SELECT * FROM sensor_queue WHERE robot_id = ? AND synced = 0 ORDER BY id ASC LIMIT ?",
        (robot_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/sync/audio")
def pull_audio(robot_id: str = "ken", limit: int = 50):
    conn = get_db()
    rows = conn.execute("SELECT * FROM audio_queue WHERE robot_id = ? AND synced = 0 ORDER BY id ASC LIMIT ?",
        (robot_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/sync/audio/{filename}")
def download_audio(filename: str):
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(filepath, media_type="audio/wav")

@app.get("/api/sync/people")
def pull_people(robot_id: str = "ken", limit: int = 50):
    conn = get_db()
    rows = conn.execute("SELECT * FROM people_queue WHERE robot_id = ? AND synced = 0 ORDER BY id ASC LIMIT ?",
        (robot_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/sync/events")
def pull_events(robot_id: str = "ken", limit: int = 100):
    conn = get_db()
    rows = conn.execute("SELECT * FROM event_queue WHERE robot_id = ? AND synced = 0 ORDER BY id ASC LIMIT ?",
        (robot_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/sync/acknowledge")
async def sync_acknowledge(request: Request):
    body = await request.json()
    ids = body.get("ids", {})
    conn = get_db()

    for key, table, file_col in [
        ("sensor_ids", "sensor_queue", None),
        ("audio_ids", "audio_queue", "filename"),
        ("people_ids", "people_queue", "photo_filename"),
        ("event_ids", "event_queue", None),
    ]:
        if key in ids and ids[key]:
            if file_col:
                for item_id in ids[key]:
                    row = conn.execute(f"SELECT {file_col} FROM {table} WHERE id = ?", (item_id,)).fetchone()
                    if row and row[file_col]:
                        fp = UPLOAD_DIR / row[file_col]
                        if fp.exists():
                            fp.unlink()
            placeholders = ",".join("?" * len(ids[key]))
            conn.execute(f"DELETE FROM {table} WHERE id IN ({placeholders})", ids[key])

    conn.commit()
    conn.close()
    return {"status": "acknowledged"}

# ─── WiFi Profiles ──────────────────────────────────────
@app.post("/api/wifi/add")
async def add_wifi(request: Request):
    body = await request.json()
    conn = get_db()
    conn.execute("INSERT INTO wifi_profiles (robot_id, ssid, password, priority) VALUES (?, ?, ?, ?)",
        (body.get("robot_id", "ken"), body["ssid"], body.get("password", ""), body.get("priority", 0)))
    conn.commit()
    conn.close()
    return {"status": "added"}

@app.get("/api/wifi/profiles")
def get_wifi(robot_id: str = "ken"):
    conn = get_db()
    rows = conn.execute("SELECT ssid, password, priority FROM wifi_profiles WHERE robot_id = ? ORDER BY priority DESC",
        (robot_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ═══════════════════════════════════════════════════════════
# CAMERA + VISION + MOVEMENT — See and move from anywhere
# ═══════════════════════════════════════════════════════════

import io
import random as _random
from PIL import Image
import numpy as np

_latest_frame = None
_prev_frame_data = None
_last_frame_time = None
_camera_robot_id = "ken"
_motion_direction = "stop"
_alive_enabled = True  # Enable auto mode

# Movement command queue (robot polls this)
_movement_cmd = {"action": "stop", "speed": 200, "timestamp": None}

# Servo command queue (robot polls this)
_last_servo_cmd = {"command": "SERVO:center", "timestamp": None}

# Face tracking data (ken-vision polls this to send servo commands to brain)
_face_tracking = {
    "detected": False,
    "face_x": 0,
    "face_y": 0,
    "face_dir": "none",
    "frame_w": 160,
    "frame_h": 120,
    "timestamp": None,
}

# ─── KEN's Personality State (like a real person) ───────
_ken_state = {
    "mood": "curious",           # happy, curious, bored, tired, alert, playful, sleepy, excited, shy, confused
    "energy": 80,                # 0-100: drains over time, recovers when resting
    "curiosity": 60,             # 0-100: increases when nothing happens
    "excitement": 20,            # 0-100: spikes on motion, fades over time
    "social_need": 30,           # 0-100: increases when alone, decreases when sees people
    "confidence": 70,            # 0-100: affects approach vs avoid behavior
    "last_motion_time": 0,
    "last_person_time": 0,
    "last_action_time": 0,
    "idle_seconds": 0,
    "consecutive_motions": 0,
    "actions_since_rest": 0,
    "wanderlust": _random.uniform(0.4, 0.9),   # personality: how much KEN likes to explore
    "sociability": _random.uniform(0.5, 1.0),   # personality: how much KEN likes people
    "shyness": _random.uniform(0.1, 0.5),       # personality: approach vs retreat
    "obedience": _random.uniform(0.7, 1.0),     # personality: how quickly KEN obeys (higher = faster)
    "current_activity": "waking_up",
    "mood_history": [],
    "thought": "Just woke up... where am I?",
}

# ─── Obedience System — KEN obeys your commands, then returns to free will ───
_obedience = {
    "obeying": False,             # True when KEN is following a user command
    "command": None,              # Current command being followed
    "command_source": None,       # "voice", "app", "manual"
    "command_start": 0,           # When the command was given
    "command_duration": 8.0,      # Seconds to obey before returning to free will
    "remaining_actions": 0,       # How many more actions to do (for multi-step commands)
    "acknowledged": "",           # KEN's spoken response to the command
    "last_command_time": 0,       # When user last gave a command
    "total_commands_given": 0,    # How many commands total
    "free_will_return_at": 0,     # Timestamp when KEN returns to free will
}

# Command acknowledgments — KEN's personality when obeying
_ack_responses = {
    "forward":  ["Ok, going!", "On my way!", "Sure thing!", "Going forward!", "Roger that!"],
    "backward": ["Backing up!", "Coming back!", "Got it, going back!", "Back it up!"],
    "left":     ["Turning left!", "Left it is!", "Heading left!", "Turning!"],
    "right":    ["Turning right!", "Right away!", "Heading right!", "On it!"],
    "stop":     ["Stopping!", "Ok, stopped!", "I'll stop here!", "Standing still!"],
    "come":     ["Coming to you!", "On my way!", "Here I come!", "Coming!"],
    "go":       ["Going!", "Let's go!", "Moving!", "Here we go!"],
    "wait":     ["I'll wait here.", "Ok, waiting.", "Standing by.", "I'm here."],
    "look":     ["Looking!", "Let me see!", "Checking it out!", "I see it!"],
    "explore":  ["Let me explore!", "Time for an adventure!", "I'll check things out!"],
    "rest":     ["Taking a break.", "I could use some rest.", "*yawn* Ok."],
    "play":     ["Yay, let's play!", "I love playing!", "Game time!"],
    "dance":    ["Watch my moves!", "Dance party!", "💃 Let's go!"],
    "spin":     ["Wheee!", "Spinning!", "Round and round!"],
}

# Free will return responses — KEN's personality when returning to independence
_return_responses = [
    "I'll do my own thing now.", "Back to being me!", "Ok, I'm free again!",
    "I'll take it from here.", "I'm going to do what I want now.",
    "Done following orders. Time to explore!", "My turn to decide!",
    "That was fun! Now I'll do my own thing.", "Freedom!",
]

# Voice command keywords → actions
_command_keywords = {
    "forward": ["forward", "go forward", "move forward", "ahead", "go ahead"],
    "backward": ["backward", "back", "go back", "reverse", "move back"],
    "left": ["left", "turn left", "go left"],
    "right": ["right", "turn right", "go right"],
    "stop": ["stop", "halt", "freeze", "wait", "stay", "don't move"],
    "come": ["come", "come here", "come to me", "get over here", "approach"],
    "go": ["go", "move", "walk", "run"],
    "explore": ["explore", "look around", "check things out", "wander"],
    "rest": ["rest", "take a break", "relax", "sleep", "nap", "chill"],
    "play": ["play", "let's play", "playtime", "have fun"],
    "dance": ["dance", "do a dance", "show me your moves"],
    "spin": ["spin", "turn around", "do a spin", "rotate"],
}

def _parse_voice_command(text):
    """Parse natural language into a movement command."""
    text = text.lower().strip()
    best_match = None
    best_score = 0
    
    for action, keywords in _command_keywords.items():
        for kw in keywords:
            if kw in text:
                score = len(kw) / len(text)  # prefer longer matches
                if score > best_score:
                    best_score = score
                    best_match = action
    
    return best_match

def _get_ack(action):
    """Get a random acknowledgment for the given action."""
    responses = _ack_responses.get(action, ["Ok!", "Sure!", "On it!", "Got it!"])
    return _random.choice(responses)

def _get_return_response():
    """Get a random free-will return message."""
    return _random.choice(_return_responses)

def _obey_command(action, speed=200, source="app"):
    """KEN obeys a command from the user."""
    global _obedience, _movement_cmd, _ken_state
    now = time.time()
    
    _obedience["obeying"] = True
    _obedience["command"] = action
    _obedience["command_source"] = source
    _obedience["command_start"] = now
    _obedience["acknowledged"] = _get_ack(action)
    _obedience["last_command_time"] = now
    _obedience["total_commands_given"] += 1
    _obedience["free_will_return_at"] = now + _obedience["command_duration"]
    
    # Set the movement command
    _movement_cmd = {
        "action": action,
        "speed": speed,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Update KEN's state
    _ken_state["current_activity"] = f"obeying_{action}"
    _ken_state["thought"] = _obedience["acknowledged"]
    _ken_state["mood"] = "happy" if _ken_state["sociability"] > 0.6 else "curious"
    _ken_state["last_action_time"] = now
    
    return _obedience["acknowledged"]

def _check_obedience_timeout():
    """Check if KEN should return to free will."""
    global _obedience, _ken_state
    now = time.time()
    
    if _obedience["obeying"] and now >= _obedience["free_will_return_at"]:
        _obedience["obeying"] = False
        _obedience["command"] = None
        _obedience["acknowledged"] = ""
        _ken_state["thought"] = _get_return_response()
        _ken_state["mood"] = _random.choice(["curious", "playful", "happy"])

# ═══════════════════════════════════════════════════════════
# AI VISION — Analyze frames and decide actions
# ═══════════════════════════════════════════════════════════
import base64

# Rate limit AI vision (process every N frames)
_ai_vision_frame_count = 0
_ai_vision_last_analysis = ""
_ai_vision_enabled = os.environ.get("AI_VISION_ENABLED", "true").lower() == "true"


async def analyze_frame_with_ai(frame_bytes: bytes) -> Optional[dict]:
    """Send frame to Groq Vision and get action decision."""
    global _ai_vision_frame_count, _ai_vision_last_analysis
    
    _ai_vision_frame_count += 1
    
    # Only analyze every 10th frame to save API calls
    if _ai_vision_frame_count % 10 != 0:
        return None
    
    key = get_next_groq_key()
    if not key:
        return None
    
    try:
        # Encode image to base64
        img_b64 = base64.b64encode(frame_bytes).decode("utf-8")
        
        prompt = f"""You are KEN's robot brain. KEN has sensors: Ultrasonic distance (cm), IR edge detectors (1=detected/0=clear).

Current sensor readings:
- Ultrasonic Left: {_latest_sensors.get('usL', 0)}cm
- Ultrasonic Right: {_latest_sensors.get('usR', 0)}cm  
- IR Front Left: {_latest_sensors.get('irFL', 0)} (1=edge detected)
- IR Front Right: {_latest_sensors.get('irFR', 0)}
- IR Rear Left: {_latest_sensors.get('irRL', 0)}
- IR Rear Right: {_latest_sensors.get('irRR', 0)}

Analyze the image AND sensors, then respond ONLY with JSON:
{{"what": "description", "action": "forward|backward|left|right|stop", "speed": 150-250, "reason": "why"}}

Rules:
- If US sees obstacle <25cm → avoid (turn opposite direction)
- If IR detects edge → go opposite direction
- If all IR clear → make your own decision based on image
- If person approaching → stop to let them come
- If nothing interesting → stop"""

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "llama-3.2-11b-vision-instant",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ],
                    "max_tokens": 256,
                    "temperature": 0.3
                }
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                _ai_vision_last_analysis = result.get("what", "")
                
                # Update motion direction based on AI decision
                action = result.get("action", "stop")
                speed = int(result.get("speed", 200))
                
                return {
                    "what": result.get("what", ""),
                    "action": action,
                    "speed": speed,
                    "reason": result.get("reason", "")
                }
    except Exception as e:
        print(f"[!] AI vision error: {e}")
    
    return None


def detect_motion(prev_bytes, curr_bytes):
    """Lightweight motion detection — compares frame regions."""
    try:
        prev = Image.open(io.BytesIO(prev_bytes)).convert("L").resize((160, 120))
        curr = Image.open(io.BytesIO(curr_bytes)).convert("L").resize((160, 120))
        prev_arr = np.array(prev, dtype=np.float32)
        curr_arr = np.array(curr, dtype=np.float32)
        diff = np.abs(curr_arr - prev_arr)

        h, w = diff.shape
        left = diff[:, :w//3].mean()
        center = diff[:, w//3:2*w//3].mean()
        right = diff[:, 2*w//3:].mean()
        top = diff[:h//3, :].mean()
        bottom = diff[2*h//3:, :].mean()

        threshold = 15
        if max(left, center, right) < threshold:
            return "stop", 0

        if left > right and left > center:
            return "left", int(left)
        elif right > left and right > center:
            return "right", int(right)
        elif top > bottom and top > center:
            return "forward", int(top)
        elif bottom > top:
            return "backward", int(bottom)
        return "stop", 0
    except Exception:
        return "stop", 0

def detect_face_region(frame_bytes):
    """Detect face-like region using skin color detection (lightweight)."""
    try:
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB").resize((160, 120))
        arr = np.array(img, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Skin color detection (works for most skin tones)
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            ((r - g) > 15) &
            (np.abs(r - g) > 15)
        )

        if skin_mask.sum() < 100:
            return "none", 0, 0

        # Find center of skin region
        ys, xs = np.where(skin_mask)
        cx = int(xs.mean())
        cy = int(ys.mean())

        # Determine direction based on face position
        w = 160
        if cx < w * 0.35:
            return "left", cx, cy
        elif cx > w * 0.65:
            return "right", cx, cy
        else:
            return "center", cx, cy
    except Exception:
        return "none", 0, 0

# ═══════════════════════════════════════════════════════════
# KEN'S BRAIN — Human-Like Behavior Engine
# ═══════════════════════════════════════════════════════════

_thoughts_by_mood = {
    "happy": [
        "This is nice!", "Having a good time!", "Everything feels great!",
        "I love exploring!", "Life is good right now!"
    ],
    "curious": [
        "What's over there?", "Hmm, something interesting...",
        "I want to see what that is!", "Let me look around...",
        "Something caught my eye..."
    ],
    "bored": [
        "Nothing happening...", "I'm so bored right now...",
        "Wish something interesting would happen...",
        "*yawn*... this is dull", "I need something to do..."
    ],
    "tired": [
        "I need to rest...", "Getting sleepy...",
        "Running low on energy...", "Need to take it easy...",
        "I've been moving too much..."
    ],
    "alert": [
        "What was that?!", "Something moved!",
        "I see something!", "Did you see that?",
        "Attention! Something's happening!"
    ],
    "playful": [
        "Wanna play?!", "Let's have some fun!",
        "Watch this!", "Hehe, this is fun!",
        "I feel like spinning around!"
    ],
    "sleepy": [
        "*yawn*... so tired...", "Maybe I'll rest here a bit...",
        "Everything is getting blurry...", "Need... sleep...",
        "Just five more minutes..."
    ],
    "excited": [
        "OMG OMG OMG!", "This is AMAZING!",
        "I can't contain myself!", "So much happening!",
        "WOOHOO!"
    ],
    "shy": [
        "Oh... someone's watching...", "Maybe I should hide a little...",
        "I'm not sure about this...", "Give me a moment...",
        "I need some space..."
    ],
    "confused": [
        "Wait... what?", "I don't understand...",
        "Hmm, that's strange...", "Something doesn't add up...",
        "I'm lost..."
    ],
}

_activity_descriptions = {
    "waking_up": "Just woke up",
    "looking_around": "Looking around",
    "investigating": "Investigating something",
    "watching_motion": "Watching something move",
    "following_face": "Looking at a face",
    "approaching": "Moving closer to see",
    "wandering": "Wandering around",
    "resting": "Taking a break",
    "napping": "Taking a nap",
    "playing": "Playing around",
    "spinning": "Spinning for fun",
    "backing_away": "Backing away carefully",
    "showing_off": "Showing off!",
    "sitting_still": "Sitting quietly",
    "scanning": "Scanning the area",
    "startled": "Got startled!",
    "peeking": "Peeking shyly",
}

def _update_mood():
    """Update KEN's mood based on internal state — like a real person."""
    global _ken_state
    s = _ken_state
    now = time.time()

    time_since_motion = now - s["last_motion_time"]
    time_since_person = now - s["last_person_time"]

    # Mood transitions based on state
    if s["energy"] < 15:
        s["mood"] = "sleepy"
    elif s["energy"] < 30:
        s["mood"] = "tired"
    elif s["excitement"] > 80:
        s["mood"] = "excited"
    elif time_since_person < 10 and s["social_need"] > 50:
        s["mood"] = "happy"
    elif time_since_motion < 5:
        s["mood"] = "alert"
    elif s["curiosity"] > 75:
        s["mood"] = "curious"
    elif s["social_need"] > 80 and time_since_person > 60:
        s["mood"] = "shy"
    elif s["idle_seconds"] > 30:
        s["mood"] = "bored"
    elif s["excitement"] > 50:
        s["mood"] = "playful"
    elif s["idle_seconds"] > 60:
        s["mood"] = "sleepy"
    else:
        # Random mood shifts (humans are unpredictable)
        if _random.random() < 0.02:
            s["mood"] = _random.choice(["curious", "playful", "happy", "bored"])

    # Track mood history
    s["mood_history"].append(s["mood"])
    if len(s["mood_history"]) > 20:
        s["mood_history"].pop(0)

    # Pick a thought
    thoughts = _thoughts_by_mood.get(s["mood"], ["..."])
    s["thought"] = _random.choice(thoughts)


def _update_needs():
    """Update internal needs — energy, curiosity, excitement, social."""
    global _ken_state
    s = _ken_state
    now = time.time()

    # Energy: slowly drains, recovers when resting
    if s["current_activity"] in ("resting", "napping", "sitting_still"):
        s["energy"] = min(100, s["energy"] + 0.8)
    else:
        drain = 0.15
        if s["current_activity"] in ("wandering", "playing", "spinning", "investigating"):
            drain = 0.4
        elif s["current_activity"] in ("approaching", "following_face"):
            drain = 0.3
        s["energy"] = max(0, s["energy"] - drain)

    # Curiosity: increases when idle, decreases when investigating
    time_since_motion = now - s["last_motion_time"]
    if time_since_motion > 15:
        s["curiosity"] = min(100, s["curiosity"] + 0.5)
    if s["current_activity"] in ("investigating", "approaching", "following_face"):
        s["curiosity"] = max(0, s["curiosity"] - 2.0)

    # Excitement: spikes on motion, fades over time
    if s["excitement"] > 0:
        s["excitement"] = max(0, s["excitement"] - 0.8)

    # Social need: increases when alone
    time_since_person = now - s["last_person_time"]
    if time_since_person > 30:
        s["social_need"] = min(100, s["social_need"] + 0.3)
    else:
        s["social_need"] = max(0, s["social_need"] - 1.0)

    # Idle counter
    if s["last_action_time"] > 0 and (now - s["last_action_time"]) > 2:
        s["idle_seconds"] = now - s["last_action_time"]
    else:
        s["idle_seconds"] = 0


def _decide_action(motion_dir, motion_strength, face_dir, face_cx, face_cy):
    """KEN decides what to do — based on mood, needs, what it sees, and sensors.
    This is the core of making KEN act like a person."""
    
    global _ken_state, _latest_sensors
    s = _ken_state
    
    # ── OBSTACLE AVOIDANCE using sensor data ──
    # Check ultrasonic sensors
    frontL = _latest_sensors.get("frontL", 0)
    frontR = _latest_sensors.get("frontR", 0)
    
    # If we have sensor data and obstacle is close
    if frontL > 0 and frontR > 0:
        min_dist = min(frontL, frontR)
        
        if min_dist < 20:
            # Very close - emergency stop!
            s["current_activity"] = "obstacle_ahead"
            s["mood"] = "cautious"
            # Turn away from obstacle
            if frontL < frontR:
                return {"action": "right", "speed": 150}
            else:
                return {"action": "left", "speed": 150}
        
        elif min_dist < 40:
            # Getting close - slow down and turn
            s["current_activity"] = "avoiding"
            if frontL < frontR:
                return {"action": "right", "speed": 80}
            else:
                return {"action": "left", "speed": 80}
    
    # ── EXPLORATION: If no obstacles and nothing detected, explore! ──
    if motion_dir == "stop" and face_dir == "none":
        # Robot is alone and sees nothing - EXPLORE!
        s["current_activity"] = "exploring"
        
        # Check battery - if low, rest more
        battery = _latest_sensors.get("battery", 12)
        if battery < 11:
            s["current_activity"] = "low_battery"
            return {"action": "stop", "speed": 0}
        
        # Random exploration behavior
        choice = _random.random()
        
        if choice < 0.4:
            # Move forward to explore
            s["mood"] = "curious"
            return {"action": "forward", "speed": 150}
        elif choice < 0.6:
            # Look around
            s["mood"] = "curious"
            return {"action": _random.choice(["left", "right"]), "speed": 100}
        elif choice < 0.8:
            # Move backward slightly
            return {"action": "backward", "speed": 100}
        else:
            # Stop and observe
            s["mood"] = "alert"
            return {"action": "stop", "speed": 0}
    s = _ken_state
    now = time.time()

    speed_base = 140
    speed_fast = 200
    speed_slow = 100

    # ── ENERGY IS CRITICAL: Must rest ──
    if s["energy"] < 10:
        s["current_activity"] = "napping"
        s["thought"] = "*snoring*... need rest..."
        return {"action": "stop", "speed": 0}

    # ── Something interesting is happening! ──
    if motion_strength > 20:
        s["last_motion_time"] = now
        s["consecutive_motions"] += 1
        s["excitement"] = min(100, s["excitement"] + motion_strength * 0.5)
        s["curiosity"] = min(100, s["curiosity"] + 15)

        # Startled reaction (like a person jumping)
        if s["consecutive_motions"] == 1 and s["mood"] not in ("alert", "excited"):
            s["current_activity"] = "startled"
            s["mood"] = "alert"
            return {"action": "stop", "speed": 0}  # Freeze for a moment

        # Turn toward motion (natural reflex)
        if motion_dir in ("left", "right"):
            s["current_activity"] = "watching_motion"
            return {"action": motion_dir, "speed": speed_slow + motion_strength}

        if motion_dir == "forward":
            # If confident and curious, approach
            if s["curiosity"] > 50 and s["confidence"] > 40:
                s["current_activity"] = "approaching"
                return {"action": "forward", "speed": speed_slow}
            else:
                # Shy or cautious — just watch
                s["current_activity"] = "sitting_still"
                return {"action": "stop", "speed": 0}
    else:
        s["consecutive_motions"] = 0

    # ── See a face! (social interaction) ──
    if face_dir != "none":
        s["last_person_time"] = now
        s["social_need"] = max(0, s["social_need"] - 5)

        if face_dir in ("left", "right"):
            # Turn to face them (natural human response)
            s["current_activity"] = "following_face"
            s["mood"] = "happy" if s["sociability"] > 0.6 else "shy"
            return {"action": face_dir, "speed": speed_slow}

        if face_dir == "center":
            # Face is centered — KEN is engaged
            if s["sociability"] > 0.7 and s["confidence"] > 50:
                # Friendly KEN approaches
                s["current_activity"] = "approaching"
                s["mood"] = "happy"
                return {"action": "forward", "speed": speed_slow}
            elif s["shyness"] > 0.4:
                # Shy KEN peeks
                s["current_activity"] = "peeking"
                s["mood"] = "shy"
                return {"action": "stop", "speed": 0}
            else:
                # Normal — just look at the person
                s["current_activity"] = "following_face"
                return {"action": "stop", "speed": 0}

    # ── Nothing happening — KEN decides on its own ──

    # Tired — rest
    if s["energy"] < 30:
        if _random.random() < 0.7:
            s["current_activity"] = "resting"
            return {"action": "stop", "speed": 0}

    # Bored — do something random
    if s["mood"] == "bored":
        choice = _random.random()
        if choice < 0.3:
            # Look left and right (like someone checking the room)
            s["current_activity"] = "scanning"
            return {"action": _random.choice(["left", "right"]), "speed": speed_slow}
        elif choice < 0.5 and s["energy"] > 40:
            # Spin (playful gesture)
            s["current_activity"] = "spinning"
            s["mood"] = "playful"
            return {"action": _random.choice(["left", "right"]), "speed": speed_base}
        elif choice < 0.7:
            # Just sit still for a while
            s["current_activity"] = "sitting_still"
            return {"action": "stop", "speed": 0}
        else:
            # Wander
            s["current_activity"] = "wandering"
            return {"action": "forward", "speed": speed_slow}

    # Curious — explore
    if s["mood"] == "curious" and s["energy"] > 40:
        if _random.random() < s["wanderlust"]:
            s["current_activity"] = "wandering"
            # Don't go same direction forever
            if _random.random() < 0.3:
                return {"action": _random.choice(["left", "right"]), "speed": speed_slow}
            return {"action": "forward", "speed": speed_slow}
        else:
            s["current_activity"] = "looking_around"
            return {"action": _random.choice(["left", "right"]), "speed": speed_slow}

    # Playful — fun movements
    if s["mood"] == "playful" and s["energy"] > 35:
        s["current_activity"] = "playing"
        fun_moves = ["forward", "left", "right", "forward"]
        return {"action": _random.choice(fun_moves), "speed": speed_base}

    # Excited — quick movements
    if s["mood"] == "excited":
        s["current_activity"] = "showing_off"
        return {"action": _random.choice(["forward", "left", "right"]), "speed": speed_fast}

    # Default: just sit and exist
    s["current_activity"] = "sitting_still"
    return {"action": "stop", "speed": 0}


def run_human_behavior():
    """KEN's main brain loop — makes decisions like a living creature.
    Called every time the camera uploads a frame.
    
    KEN has free will BUT obeys your commands first.
    When you command KEN, he follows for ~8 seconds then returns to being himself.
    """
    global _movement_cmd, _ken_state

    s = _ken_state
    now = time.time()

    # ── Step 0: Check obedience ──────────────────────────
    # If KEN was commanded, keep obeying until timeout
    _check_obedience_timeout()
    
    if _obedience["obeying"]:
        # KEN is following a command — don't override with free will
        remaining = _obedience["free_will_return_at"] - now
        s["current_activity"] = f"obeying_{_obedience['command']}"
        s["thought"] = _obedience["acknowledged"] + f" ({max(0, int(remaining))}s left)"
        
        # Keep executing the commanded action
        _movement_cmd = {
            "action": _obedience["command"] if _obedience["command"] else "stop",
            "speed": 200 if _obedience["command"] != "stop" else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        return  # Skip free will while obeying

    # ── Step 1: Analyze what KEN sees ──
    motion_dir = "stop"
    motion_strength = 0
    face_dir = "none"
    face_cx, face_cy = 0, 0

    if _prev_frame_data and _latest_frame:
        motion_dir, motion_strength = detect_motion(_prev_frame_data, _latest_frame)
        if motion_dir == "stop":
            face_dir, face_cx, face_cy = detect_face_region(_latest_frame)

    # ── Step 2: Update internal state ──
    _update_needs()

    # ── Step 3: Update mood based on what KEN sees and feels ──
    _update_mood()

    # ── Step 4: Decide what to do ──
    action = _decide_action(motion_dir, motion_strength, face_dir, face_cx, face_cy)

    # ── Step 5: Execute ──
    _movement_cmd = {
        "action": action["action"],
        "speed": action["speed"],
        "timestamp": datetime.utcnow().isoformat()
    }

    s["last_action_time"] = now
    s["actions_since_rest"] += 1

    # Reset rest counter when resting
    if s["current_activity"] in ("resting", "napping", "sitting_still"):
        s["actions_since_rest"] = 0

@app.post("/camera/upload")
async def camera_upload(request: Request):
    """ESP32-CAM uploads a JPEG frame. KEN's brain decides what to do.
    Also detects face position for head tracking."""
    global _latest_frame, _prev_frame_data, _last_frame_time, _camera_robot_id
    global _motion_direction, _face_tracking

    body = await request.body()
    _prev_frame_data = _latest_frame
    _latest_frame = body
    _last_frame_time = datetime.utcnow().isoformat()
    _camera_robot_id = request.headers.get("X-Robot-ID", "ken")

    # Detect face position for head tracking
    face_dir, face_cx, face_cy = detect_face_region(body)
    if face_dir != "none":
        _face_tracking = {
            "detected": True,
            "face_x": face_cx,
            "face_y": face_cy,
            "face_dir": face_dir,
            "frame_w": 160,
            "frame_h": 120,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        _face_tracking["detected"] = False
        _face_tracking["timestamp"] = datetime.utcnow().isoformat()

    # ── AI Vision: Analyze frame and decide action ──
    ai_decision = None
    if _ai_vision_enabled:
        ai_decision = await analyze_frame_with_ai(body)
        if ai_decision:
            # Override motion based on AI decision
            action = ai_decision.get("action", "stop")
            speed = ai_decision.get("speed", 200)
            _motion_direction = action
            
            # Update move/poll for brain
            _movement_cmd = {
                "action": action,
                "speed": speed,
                "timestamp": datetime.utcnow().isoformat(),
                "ai_reason": ai_decision.get("reason", ""),
                "ai_what": ai_decision.get("what", "")
            }

    # KEN is alive — make decisions like a person
    if _alive_enabled:
        run_human_behavior()

    return {
        "status": "ok",
        "size": len(body),
        "mood": _ken_state["mood"],
        "activity": _ken_state["current_activity"],
        "face_detected": _face_tracking["detected"],
        "ai_vision": ai_decision is not None,
        "ai_what": ai_decision.get("what", "") if ai_decision else None,
        "ai_action": ai_decision.get("action", "") if ai_decision else None
    }

@app.get("/camera/latest")
def camera_latest():
    if not _latest_frame:
        raise HTTPException(404, "No frame available")
    return Response(
        content=_latest_frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/camera/status")
def camera_status():
    s = _ken_state
    return {
        "online": _latest_frame is not None,
        "robot_id": _camera_robot_id,
        "last_frame": _last_frame_time,
        "frame_size": len(_latest_frame) if _latest_frame else 0,
        "mode": "alive" if _alive_enabled else "manual",
        "mood": s["mood"],
        "thought": s["thought"],
        "activity": s["current_activity"],
        "energy": round(s["energy"]),
        "curiosity": round(s["curiosity"]),
        "excitement": round(s["excitement"]),
        "social_need": round(s["social_need"]),
        "motion": _motion_direction
    }

# ─── Personality API (for Android app) ──────────────────
@app.get("/personality")
def get_personality():
    """Get KEN's full personality state — mood, thoughts, needs, obedience."""
    s = _ken_state
    return {
        "mood": s["mood"],
        "thought": s["thought"],
        "activity": s["current_activity"],
        "energy": round(s["energy"]),
        "curiosity": round(s["curiosity"]),
        "excitement": round(s["excitement"]),
        "social_need": round(s["social_need"]),
        "confidence": round(s["confidence"]),
        "traits": {
            "wanderlust": round(s["wanderlust"], 2),
            "sociability": round(s["sociability"], 2),
            "shyness": round(s["shyness"], 2),
            "obedience": round(s["obedience"], 2),
        },
        "obedience": {
            "obeying": _obedience["obeying"],
            "command": _obedience["command"],
            "source": _obedience["command_source"],
            "acknowledged": _obedience["acknowledged"],
            "total_commands": _obedience["total_commands_given"],
            "remaining_seconds": max(0, int(_obedience["free_will_return_at"] - time.time())) if _obedience["obeying"] else 0,
        },
        "mood_history": s["mood_history"][-10:],
        "alive": _alive_enabled,
    }

# ─── Movement Commands ──────────────────────────────────
@app.post("/move")
async def move_command(request: Request):
    """Send movement command to robot. KEN obeys, then returns to free will.
    Action: forward/backward/left/right/stop."""
    body = await request.json()
    action = body.get("action", "stop")
    speed = body.get("speed", 200)
    
    # Use obedience system — KEN follows this command then returns to free will
    ack = _obey_command(action, speed, "app")
    
    return {
        "status": "ok",
        "command": {"action": action, "speed": speed},
        "acknowledged": ack,
        "mood": _ken_state["mood"]
    }

@app.get("/move/poll")
def move_poll():
    """Robot polls for movement commands."""
    return _movement_cmd

@app.post("/servo")
async def servo_command(request: Request):
    """Direct servo control. Body: {"pan": 90, "tilt": 90} or {"action": "center"}"""
    global _last_servo_cmd
    body = await request.json()
    action = body.get("action", "")

    if action == "center":
        _face_tracking["detected"] = False
        _last_servo_cmd = {"command": "SERVO:center", "timestamp": datetime.utcnow().isoformat()}
        return {"status": "ok", "command": "SERVO:center"}
    
    pan = body.get("pan", 90)
    tilt = body.get("tilt", 90)
    _last_servo_cmd = {"command": f"SERVO:{pan}:{tilt}", "timestamp": datetime.utcnow().isoformat()}
    return {"status": "ok", "command": f"SERVO:{pan}:{tilt}"}

@app.get("/servo/poll")
def servo_poll():
    """Robot polls for servo commands."""
    return _last_servo_cmd

@app.post("/mode")
async def set_mode(request: Request):
    """Toggle KEN between alive and manual. Modes: alive, manual."""
    global _alive_enabled, _movement_cmd
    body = await request.json()
    mode = body.get("mode", "alive")
    if mode == "alive":
        _alive_enabled = True
    elif mode == "manual":
        _alive_enabled = False
        _movement_cmd = {"action": "stop", "speed": 0, "timestamp": datetime.utcnow().isoformat()}
    return {"status": "ok", "mode": "alive" if _alive_enabled else "manual"}

@app.get("/mode")
def get_mode():
    """Get current mode."""
    return {"mode": "alive" if _alive_enabled else "manual"}

# ─── Full Dashboard ─────────────────────────────────────
@app.get("/camera")
def camera_dashboard():
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>KEN - Alive</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#FFD600;font-family:'Courier New',monospace;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:10px}
h1{font-size:22px;margin:5px 0}
.sub{color:#666;font-size:11px;margin-bottom:8px}
.feed{border:2px solid #FFD600;border-radius:10px;overflow:hidden;
      max-width:480px;width:100%;background:#000}
.feed img{width:100%;display:block}

/* Personality Display */
.personality{max-width:480px;width:100%;margin:10px 0;background:#111;border-radius:10px;padding:12px;border:1px solid #333}
.thought{font-size:14px;color:#FFD600;text-align:center;margin:8px 0;
         font-style:italic;min-height:24px}
.mood-row{display:flex;justify-content:center;align-items:center;gap:10px;margin:6px 0}
.mood-emoji{font-size:32px}
.mood-label{font-size:16px;text-transform:uppercase;font-weight:bold}
.activity{font-size:11px;color:#888;text-align:center;margin:4px 0}

/* Meters */
.meters{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}
.meter{display:flex;flex-direction:column;gap:2px}
.meter-label{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px}
.meter-bar{height:6px;background:#222;border-radius:3px;overflow:hidden}
.meter-fill{height:100%;border-radius:3px;transition:width 0.5s}
.meter-fill.energy{background:linear-gradient(90deg,#f00,#ff0,#0f0)}
.meter-fill.curiosity{background:linear-gradient(90deg,#888,#0ff)}
.meter-fill.excitement{background:linear-gradient(90deg,#888,#f0f)}
.meter-fill.social{background:linear-gradient(90deg,#888,#f80)}

/* Status bar */
.status{margin:6px 0;display:flex;gap:15px;font-size:12px;flex-wrap:wrap;justify-content:center}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}
.dot.green{background:#0f0;box-shadow:0 0 6px #0f0}
.dot.red{background:#f00;box-shadow:0 0 6px #f00}

/* Controls */
.controls{margin:10px 0;display:flex;flex-direction:column;align-items:center;gap:8px}
.row{display:flex;gap:8px}
.btn{width:60px;height:60px;background:#1a1a1a;color:#FFD600;border:1px solid #FFD600;
     border-radius:10px;font-size:24px;cursor:pointer;display:flex;align-items:center;
     justify-content:center;-webkit-tap-highlight-color:transparent;user-select:none}
.btn:active,.btn.pressed{background:#FFD600;color:#000}
.btn-stop{background:#f00;color:#fff;border-color:#f00;width:60px;height:60px;border-radius:50%;
          font-size:16px;font-weight:bold}
.btn-stop:active{background:#ff4444}
.actions{display:flex;gap:8px;margin-top:5px;flex-wrap:wrap;justify-content:center}
.act{padding:8px 14px;background:#1a1a1a;color:#FFD600;border:1px solid #FFD600;
     border-radius:6px;font-size:12px;cursor:pointer;font-family:inherit}
.act:active{background:#FFD600;color:#000}
.act.on{background:#FFD600;color:#000}
.meter-info{font-size:11px;color:#555;text-align:center;margin-top:4px}
</style>
</head>
<body>
<h1>KEN</h1>
<div class="sub">i'm alive</div>

<div class="feed">
  <img id="feed" src="/camera/latest" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22320%22 height=%22240%22><rect fill=%22%23000%22 width=%22320%22 height=%22240%22/><text x=%22160%22 y=%22120%22 fill=%22%23FFD600%22 text-anchor=%22middle%22 font-family=%22monospace%22>Waiting for camera...</text></svg>'">
  <div style="text-align:center;padding:4px">
    <input id="directUrl" placeholder="Direct stream: http://192.168.x.x/stream" 
           style="width:90%;padding:4px;background:#111;color:#FFD600;border:1px solid #333;border-radius:4px;font-size:10px;font-family:inherit">
    <button onclick="switchDirect()" class="act" style="margin-top:2px;padding:3px 8px;font-size:10px">DIRECT</button>
  </div>
</div>

<div class="personality">
  <div class="mood-row">
    <span class="mood-emoji" id="moodEmoji">&#128075;</span>
    <span class="mood-label" id="moodLabel">Waking Up</span>
  </div>
  <div class="thought" id="thought">"..."</div>
  <div class="activity" id="activity">--</div>

  <div class="meters">
    <div class="meter">
      <span class="meter-label">Energy</span>
      <div class="meter-bar"><div class="meter-fill energy" id="energyBar" style="width:80%"></div></div>
    </div>
    <div class="meter">
      <span class="meter-label">Curiosity</span>
      <div class="meter-bar"><div class="meter-fill curiosity" id="curiosityBar" style="width:60%"></div></div>
    </div>
    <div class="meter">
      <span class="meter-label">Excitement</span>
      <div class="meter-bar"><div class="meter-fill excitement" id="excitementBar" style="width:20%"></div></div>
    </div>
    <div class="meter">
      <span class="meter-label">Social</span>
      <div class="meter-bar"><div class="meter-fill social" id="socialBar" style="width:30%"></div></div>
    </div>
  </div>
</div>

<div class="status">
  <span><span class="dot" id="dot"></span><span id="statusText">Connecting...</span></span>
</div>

<div class="controls" id="joystick" style="display:none">
  <button class="btn" ontouchstart="sendMove('forward')" ontouchend="sendMove('stop')"
          onmousedown="sendMove('forward')" onmouseup="sendMove('stop')">UP</button>
  <div class="row">
    <button class="btn" ontouchstart="sendMove('left')" ontouchend="sendMove('stop')"
            onmousedown="sendMove('left')" onmouseup="sendMove('stop')">LEFT</button>
    <button class="btn btn-stop" onclick="sendMove('stop')">STOP</button>
    <button class="btn" ontouchstart="sendMove('right')" ontouchend="sendMove('stop')"
            onmousedown="sendMove('right')" onmouseup="sendMove('stop')">RIGHT</button>
  </div>
  <button class="btn" ontouchstart="sendMove('backward')" ontouchend="sendMove('stop')"
          onmousedown="sendMove('backward')" onmouseup="sendMove('stop')">DOWN</button>
</div>

<div class="actions">
  <button class="act on" id="aliveBtn" onclick="toggleAlive()">ALIVE</button>
  <button class="act" id="manualBtn" onclick="toggleManual()">MANUAL</button>
  <button class="act" id="streamBtn" onclick="toggleStream()">PAUSE</button>
</div>
<div class="meter-info" id="meterInfo">--</div>

<script>
let streaming=true, interval=null;
const img=document.getElementById('feed');

const moodEmojis={
  happy:"&#128516;",curious:"&#129300;",bored:"&#128528;",tired:"&#128564;",
  alert:"&#128562;",playful:"&#128519;",sleepy:"&#128554;",excited:"&#128526;",
  shy:"&#128527;",confused:"&#128533;"
};
const moodNames={
  happy:"Happy",curious:"Curious",bored:"Bored",tired:"Tired",
  alert:"Alert!",playful:"Playful",sleepy:"Sleepy",excited:"Excited!",
  shy:"Shy",confused:"Confused"
};

function startStream(){
  interval=setInterval(()=>{
    if(streaming && !directMode) img.src='/camera/latest?t='+Date.now();
  },300);
}

let directMode = false;
function switchDirect(){
  const url = document.getElementById('directUrl').value.trim();
  if(!url){ alert('Enter camera URL, e.g. http://192.168.1.12/stream'); return; }
  directMode = !directMode;
  if(directMode){
    clearInterval(interval);
    img.src = url;
  } else {
    startStream();
  }
}

function sendMove(action){
  fetch('/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action,speed:200})});
}

function toggleAlive(){
  fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:'alive'})});
  document.getElementById('aliveBtn').classList.add('on');
  document.getElementById('manualBtn').classList.remove('on');
  document.getElementById('joystick').style.display='none';
}

function toggleManual(){
  fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:'manual'})});
  document.getElementById('manualBtn').classList.add('on');
  document.getElementById('aliveBtn').classList.remove('on');
  document.getElementById('joystick').style.display='flex';
}

function toggleStream(){
  streaming=!streaming;
  document.getElementById('streamBtn').textContent=streaming?'PAUSE':'RESUME';
}

function updatePersonality(d){
  document.getElementById('moodEmoji').innerHTML=moodEmojis[d.mood]||'&#128075;';
  document.getElementById('moodLabel').textContent=moodNames[d.mood]||d.mood;
  document.getElementById('thought').textContent='"'+(d.thought||'...')+'"';
  document.getElementById('activity').textContent=d.activity||'--';
  document.getElementById('energyBar').style.width=(d.energy||0)+'%';
  document.getElementById('curiosityBar').style.width=(d.curiosity||0)+'%';
  document.getElementById('excitementBar').style.width=(d.excitement||0)+'%';
  document.getElementById('socialBar').style.width=(d.social_need||0)+'%';
}

img.onload=function(){
  document.getElementById('dot').className='dot green';
  document.getElementById('statusText').textContent='Online';
  fetch('/camera/status').then(r=>r.json()).then(d=>{
    updatePersonality(d);
    document.getElementById('meterInfo').textContent=
      'Frame: '+(d.frame_size/1024).toFixed(1)+'KB | Last: '+
      (d.last_frame?new Date(d.last_frame+'Z').toLocaleTimeString():'--');
  });
};
img.onerror=function(){
  document.getElementById('dot').className='dot red';
  document.getElementById('statusText').textContent='Offline';
};

startStream();
</script>
</body>
</html>"""
    return Response(content=html, media_type="text/html")

# ─── Cleanup ─────────────────────────────────────────────
@app.on_event("startup")
def cleanup():
    conn = get_db()
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    conn.execute("DELETE FROM sensor_queue WHERE created_at < ?", (cutoff,))
    conn.execute("DELETE FROM event_queue WHERE created_at < ?", (cutoff,))
    conn.commit()
    conn.close()

# ─── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n{'='*50}")
    print(f"  KEN Cloud Server — Render Edition")
    print(f"  STT: Groq Whisper API (cloud)")
    print(f"  TTS: edge-tts (70+ voices)")
    print(f"  AI: Gemini/Groq/DeepSeek proxy")
    print(f"  Camera: Live feed enabled")
    print(f"  Port: {port}")
    print(f"  FREE. 24/7. NO LIMITS.")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
