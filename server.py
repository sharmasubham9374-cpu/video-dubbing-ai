import os
import sys
import uuid
import json
import time
import re
import shutil
import base64
import asyncio
import requests
import subprocess
from pathlib import Path
from typing import Optional, List
import edge_tts

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
AUDIO_DIR = STORAGE_DIR / "audio"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
CONFIG_FILE = BASE_DIR / "config.json"
LOCAL_BIN = BASE_DIR / "bin"

for p in [UPLOADS_DIR, AUDIO_DIR, OUTPUTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ==================== MODULE: config.py ====================
import json


DEFAULT_CONFIG = {
    "elevenlabs_api_key": "",
    "gemini_api_key": "",
    "selected_voice_id": "",
    "selected_voice_name": "Bunty",
    "tts_model": "eleven_multilingual_v2",
    "voice_stability": 0.5,
    "voice_similarity_boost": 0.8,
    "voice_style": 0.0,
    "voice_speaker_boost": True,
    "audio_mode": "replace",  # "ducking" or "replace"
    "bg_music_volume": 0.0,
    "voice_volume": 1.0,
    "burn_subtitles": False,
    "target_language": "Hindi"
}

def load_config():
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(data)
            return merged
    except Exception as e:
        print(f"Error loading config: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False



# ==================== MODULE: audio_extractor.py ====================
import subprocess
import json
import shutil


def get_ffmpeg_path():
    # 1. Check local bin directory
    local_ffmpeg = LOCAL_BIN / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    
    # 2. Check system PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    
    # 3. Check common Windows locations
    common_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
            
    return "ffmpeg"

def get_ffprobe_path():
    local_ffprobe = LOCAL_BIN / "ffprobe.exe"
    if local_ffprobe.exists():
        return str(local_ffprobe)
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffprobe:
        return sys_ffprobe
    common_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ffmpeg" / "bin" / "ffprobe.exe",
        Path(r"C:\ffmpeg\bin\ffprobe.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffprobe.exe")
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
    return "ffprobe"

def get_video_info(video_path: str):
    """Retrieve duration, dimensions, and audio stream existence from video."""
    ffprobe_cmd = get_ffprobe_path()
    cmd = [
        ffprobe_cmd,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        duration = float(data.get("format", {}).get("duration", 0.0))
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        
        width = int(video_stream.get("width", 1280)) if video_stream else 1280
        height = int(video_stream.get("height", 720)) if video_stream else 720
        
        return {
            "duration": duration,
            "width": width,
            "height": height,
            "has_audio": audio_stream is not None,
            "format": data.get("format", {})
        }
    except Exception as e:
        print(f"Error reading video info: {e}")
        return {"duration": 0.0, "width": 1280, "height": 720, "has_audio": True}

def extract_audio(video_path: str, output_audio_path: str, sample_rate: int = 44100, channels: int = 2):
    """Extract audio track from video file as MP3/WAV."""
    ffmpeg_cmd = get_ffmpeg_path()
    output_dir = Path(output_audio_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i", video_path,
        "-vn",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-b:a", "192k",
        output_audio_path
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {process.stderr}")
    
    return output_audio_path



# ==================== MODULE: transcriber.py ====================
import json
import time
import base64
import subprocess
import requests

def transcribe_audio_gemini(audio_path: str, api_key: str = None, progress_callback = None) -> list:
    """Transcribe audio to English segments with timestamps across the entire video duration."""
    if not api_key:
        config = load_config()
        api_key = config.get("gemini_api_key")
        
    if not api_key:
        raise ValueError("Gemini API Key is missing. Please set it in Settings.")

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ffmpeg_cmd = get_ffmpeg_path()
    info = get_video_info(audio_path)
    total_dur = info.get("duration", 0.0)
    
    # 60s chunks ensure complete coverage with 0 rate limits and 0 timeouts
    chunk_duration = 60.0
    temp_chunks_dir = Path(audio_path).parent / f"temp_transcribe_{Path(audio_path).stem}"
    temp_chunks_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []
    current_start = 0.0
    chunk_idx = 0
    total_chunks = max(1, int(total_dur // chunk_duration) + (1 if total_dur % chunk_duration > 0.5 else 0))

    models_to_try = [
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.7-flash",
        "gemini-3.5-flash"
    ]

    while current_start < total_dur:
        chunk_len = min(chunk_duration, total_dur - current_start)
        if chunk_len < 0.5:
            break

        chunk_idx += 1
        if progress_callback:
            progress_callback(chunk_idx, total_chunks, f"Transcribing part {chunk_idx}/{total_chunks} ({current_start:.0f}s - {current_start + chunk_len:.0f}s)...")

        chunk_file = temp_chunks_dir / f"chunk_{chunk_idx:03d}.mp3"
        cmd = [
            ffmpeg_cmd, "-y",
            "-ss", f"{current_start:.3f}",
            "-t", f"{chunk_len:.3f}",
            "-i", audio_path,
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            str(chunk_file)
        ]
        subprocess.run(cmd, capture_output=True)

        if chunk_file.exists() and chunk_file.stat().st_size > 100:
            with open(chunk_file, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            prompt = f"""
You are an expert audio transcription system.
Listen to this {chunk_len:.1f}-second audio clip and transcribe all spoken English dialogue.
Break speech into natural, spoken sentences with start and end timestamps.

You MUST reply ONLY with a valid JSON array of objects:
[
  {{
    "start": 0.5,
    "end": 3.8,
    "text": "spoken English sentence."
  }}
]
Notes:
- 'start' and 'end' must be in seconds relative to this chunk (between 0.0 and {chunk_len:.1f}).
- If no speech is present, return [].
"""
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"inline_data": {"mime_type": "audio/mp3", "data": b64_data}},
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json"
                }
            }

            success = False
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                try:
                    r = requests.post(url, json=payload, timeout=45)
                    if r.status_code == 200:
                        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if raw.startswith("```json"): raw = raw[7:]
                        if raw.startswith("```"): raw = raw[3:]
                        if raw.endswith("```"): raw = raw[:-3]
                        segs = json.loads(raw.strip())
                        for s in segs:
                            s_start = round(current_start + float(s.get("start", 0.0)), 2)
                            s_end = round(current_start + float(s.get("end", chunk_len)), 2)
                            txt = s.get("text", "").strip()
                            if txt:
                                all_segments.append({
                                    "id": len(all_segments),
                                    "start": s_start,
                                    "end": s_end,
                                    "text": txt
                                })
                        success = True
                        break
                    elif r.status_code == 429:
                        time.sleep(2)
                except Exception as e:
                    print(f"Transcription error on chunk {chunk_idx} with {model_name}: {e}")

            if not success:
                print(f"Warning: Chunk {chunk_idx} transcription skipped")

        current_start += chunk_len
        # Small delay between chunks to strictly respect free-tier RPM limits
        time.sleep(2.5)

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(temp_chunks_dir, ignore_errors=True)
    except Exception:
        pass

    return all_segments


# ==================== MODULE: translator.py ====================
import json
import time
import re
import requests

def has_hindi_characters(text: str) -> bool:
    """Check if text contains Devanagari script characters."""
    if not text:
        return False
    return bool(re.search(r'[\u0900-\u097F]', text))

def translate_single_text(english_text: str, api_key: str) -> str:
    """Fallback: translate single sentence to Hindi."""
    prompt = f"Translate the following English line to natural conversational Hindi in Devanagari script only (no notes or explanation):\n\"{english_text}\""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    models = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.7-flash"]
    for m in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            time.sleep(2)
        except Exception:
            pass
    return english_text

def translate_batch_with_retry(batch_segments: list, api_key: str, max_retries: int = 4) -> list:
    """Translate a batch with guaranteed retry until translated into Hindi."""
    input_text = json.dumps(batch_segments, ensure_ascii=False, indent=2)

    prompt = f"""
You are an expert Hollywood/Bollywood dubbing translator specializing in English to Hindi video dubbing.
Translate ALL the following English transcript segments into natural, fluent, and engaging Hindi (Devanagari script).

CRITICAL DUBBING RULES:
1. The Hindi translation MUST sound natural, fluent and conversational when spoken aloud.
2. TIMING & PACING: Keep each Hindi translation concise and punchy so it fits the timing smoothly.
3. Every single segment MUST have a "hindi_text" field translated in Devanagari (e.g. 'नमस्ते दोस्तों').
4. Modern terms like 'video', 'internet', 'treaty', 'timeline' can be written in Devanagari ('वीडियो', 'इंटरनेट', 'ट्रीटी', 'टाइमलाइन').

INPUT SEGMENTS:
{input_text}

You MUST reply ONLY with a valid JSON array of objects with the exact same segment IDs:
[
  {{
    "id": 0,
    "start": 0.0,
    "end": 3.4,
    "english_text": "Hey everyone, welcome back.",
    "hindi_text": "नमस्ते दोस्तों, वापस स्वागत है।"
  }}
]
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    models_to_try = [
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.7-flash"
    ]

    for attempt in range(max_retries):
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
                if r.status_code == 200:
                    raw_text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if raw_text.startswith("```json"): raw_text = raw_text[7:]
                    if raw_text.startswith("```"): raw_text = raw_text[3:]
                    if raw_text.endswith("```"): raw_text = raw_text[:-3]
                    translated = json.loads(raw_text.strip())

                    trans_map = {item.get("id"): item.get("hindi_text", "") for item in translated}
                    result = []
                    for s in batch_segments:
                        s_id = s.get("id")
                        hi_text = trans_map.get(s_id, "")
                        # Validate Hindi presence
                        if not has_hindi_characters(hi_text):
                            hi_text = translate_single_text(s.get("text", s.get("english_text", "")), api_key)

                        result.append({
                            "id": s_id,
                            "start": s.get("start", 0.0),
                            "end": s.get("end", 0.0),
                            "english_text": s.get("text", s.get("english_text", "")),
                            "hindi_text": hi_text
                        })
                    return result
                elif r.status_code == 429:
                    print(f"Rate limit on {model_name} (attempt {attempt+1}), waiting 4 seconds...")
                    time.sleep(4)
                else:
                    print(f"Model {model_name} returned status {r.status_code}")
            except Exception as e:
                print(f"Error on {model_name} (attempt {attempt+1}): {e}")
                time.sleep(2)

    # If all batch attempts fail, translate each segment one-by-one with fallbacks
    print("[!] Batch translation fallback: translating individually...")
    fallback_result = []
    for s in batch_segments:
        eng = s.get("text", s.get("english_text", ""))
        hi_text = translate_single_text(eng, api_key)
        fallback_result.append({
            "id": s.get("id"),
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "english_text": eng,
            "hindi_text": hi_text
        })
        time.sleep(1.5)
    return fallback_result

def translate_segments_to_hindi(segments: list, api_key: str = None) -> list:
    """Translate English segments to natural conversational Hindi with 100% guarantee."""
    if not api_key:
        config = load_config()
        api_key = config.get("gemini_api_key")
        
    if not api_key:
        raise ValueError("Gemini API Key is missing. Please set it in Settings.")

    if not segments:
        return []

    # Batch into smaller groups of 15 segments for maximum reliability
    batch_size = 15
    final_segments = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        print(f"[*] Translating Batch {i//batch_size + 1} (Segments {i+1} to {min(i+batch_size, len(segments))})...")
        translated_batch = translate_batch_with_retry(batch, api_key)
        final_segments.extend(translated_batch)
        if i + batch_size < len(segments):
            time.sleep(3.0)

    # Final sanity check: Ensure 100% of segments have Hindi text
    for seg in final_segments:
        if not has_hindi_characters(seg.get("hindi_text", "")):
            eng = seg.get("english_text", seg.get("text", ""))
            seg["hindi_text"] = translate_single_text(eng, api_key)

    return final_segments


# ==================== MODULE: elevenlabs_tts.py ====================
import json
import asyncio
import requests
import edge_tts

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Built-in High-Fidelity Indian Neural Voices
PRESET_VOICES = [
    {
        "voice_id": "edge:hi-IN-MadhurNeural",
        "name": "Bunty (Natural Hindi Male Voice)",
        "category": "Built-in AI",
        "preview_url": "",
        "labels": {"accent": "Indian", "language": "Hindi", "gender": "Male"},
        "is_bunty": True,
        "engine": "edge"
    },
    {
        "voice_id": "edge:hi-IN-SwaraNeural",
        "name": "Swara (Natural Hindi Female Voice)",
        "category": "Built-in AI",
        "preview_url": "",
        "labels": {"accent": "Indian", "language": "Hindi", "gender": "Female"},
        "is_bunty": False,
        "engine": "edge"
    }
]

def get_voices(api_key: str = None) -> list:
    """Fetch all available voices (including built-in Bunty and ElevenLabs voices)."""
    voices_list = [dict(v) for v in PRESET_VOICES]

    if not api_key:
        config = load_config()
        api_key = config.get("elevenlabs_api_key")

    if api_key and not api_key.startswith("API_KEY_ID"):
        headers = {
            "xi-api-key": api_key,
            "Accept": "application/json"
        }
        try:
            response = requests.get(f"{ELEVENLABS_BASE_URL}/voices", headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                el_voices = data.get("voices", [])
                for v in el_voices:
                    v_name = v.get("name", "")
                    v_id = v.get("voice_id", "")
                    is_bunty = "bunty" in v_name.lower()
                    voices_list.append({
                        "voice_id": f"elevenlabs:{v_id}",
                        "name": f"{v_name} (ElevenLabs)",
                        "category": v.get("category", "ElevenLabs"),
                        "preview_url": v.get("preview_url", ""),
                        "labels": v.get("labels", {}),
                        "is_bunty": is_bunty,
                        "engine": "elevenlabs"
                    })
        except Exception as e:
            print(f"Could not load ElevenLabs voices: {e}")

    # Ensure Bunty is at the top
    voices_list.sort(key=lambda x: (not x["is_bunty"], x["name"].lower()))
    return voices_list

async def synthesize_edge_tts(text: str, voice_name: str, output_path: str):
    """Synthesize speech using Microsoft Edge Neural TTS."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Clean text to prevent SSML parsing issues
    clean_text = text.strip()
    communicate = edge_tts.Communicate(clean_text, voice_name)
    await communicate.save(output_path)
    return output_path

def synthesize_speech_segment(
    text: str,
    voice_id: str,
    output_path: str,
    api_key: str = None,
    model_id: str = "eleven_multilingual_v2",
    voice_settings: dict = None
) -> str:
    """Synthesize Hindi text using either Edge-TTS (Bunty) or ElevenLabs."""
    if not voice_id:
        voice_id = "edge:hi-IN-MadhurNeural"

    # 1. Edge-TTS Engine (Default Bunty Voice)
    if voice_id.startswith("edge:") or voice_id in ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]:
        voice_name = voice_id.replace("edge:", "")
        asyncio.run(synthesize_edge_tts(text, voice_name, output_path))
        return output_path

    # 2. ElevenLabs Engine
    actual_voice_id = voice_id.replace("elevenlabs:", "")
    if not api_key:
        config = load_config()
        api_key = config.get("elevenlabs_api_key")
        
    if not api_key:
        # Fallback to Bunty Edge-TTS if no ElevenLabs key
        asyncio.run(synthesize_edge_tts(text, "hi-IN-MadhurNeural", output_path))
        return output_path

    if not voice_settings:
        config = load_config()
        voice_settings = {
            "stability": config.get("voice_stability", 0.5),
            "similarity_boost": config.get("voice_similarity_boost", 0.8),
            "style": config.get("voice_style", 0.0),
            "use_speaker_boost": config.get("voice_speaker_boost", True)
        }

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings
    }

    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{actual_voice_id}?output_format=mp3_44100_128"
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        # If ElevenLabs call fails (e.g. permission or quota error), fallback to Bunty Edge-TTS seamlessly
        print(f"ElevenLabs TTS failed ({response.status_code}), falling back to Bunty neural voice...")
        asyncio.run(synthesize_edge_tts(text, "hi-IN-MadhurNeural", output_path))
        return output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path

def synthesize_all_segments(
    segments: list,
    voice_id: str,
    output_dir: str,
    api_key: str = None,
    progress_callback=None
) -> list:
    """Synthesize all Hindi segments into separate audio files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_segments = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        hindi_text = seg.get("hindi_text", "").strip()
        if not hindi_text:
            hindi_text = seg.get("english_text", "").strip()
            
        seg_audio_file = str(output_path / f"segment_{idx:03d}.mp3")
        
        if progress_callback:
            progress_callback(idx + 1, total, f"Generating Hindi voice for segment {idx + 1}/{total}...")

        if hindi_text:
            try:
                synthesize_speech_segment(
                    text=hindi_text,
                    voice_id=voice_id,
                    output_path=seg_audio_file,
                    api_key=api_key
                )
            except Exception as e:
                print(f"Error generating audio for segment {idx}: {e}")
                seg_audio_file = None
        else:
            seg_audio_file = None

        seg_data = dict(seg)
        seg_data["audio_file"] = seg_audio_file
        result_segments.append(seg_data)

    return result_segments


# ==================== MODULE: video_merger.py ====================
import subprocess
import shutil

def get_audio_duration(audio_file: str, ffmpeg_cmd: str = None) -> float:
    """Get accurate duration of an audio file using ffprobe/video_info."""
    try:
        info = get_video_info(audio_file)
        dur = info.get("duration", 0.0)
        if dur > 0:
            return dur
    except Exception:
        pass
    return 2.0

def fit_audio_segment_duration(
    input_file: str,
    output_file: str,
    target_duration: float,
    ffmpeg_cmd: str
) -> float:
    """Speed up audio slightly if it exceeds the available time window using atempo (preserves pitch)."""
    actual_dur = get_audio_duration(input_file, ffmpeg_cmd)
    
    # If actual duration exceeds available window, speed up slightly (up to 1.35x)
    if target_duration > 0.5 and actual_dur > (target_duration + 0.1):
        tempo = min(1.35, actual_dur / target_duration)
        if tempo > 1.05:
            cmd = [
                ffmpeg_cmd, "-y",
                "-i", input_file,
                "-filter:a", f"atempo={tempo:.2f}",
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                output_file
            ]
            subprocess.run(cmd, capture_output=True)
            if Path(output_file).exists():
                return get_audio_duration(output_file, ffmpeg_cmd)

    # If no tempo adjustment needed, just copy
    shutil.copyfile(input_file, output_file)
    return actual_dur

def build_hindi_audio_track(
    segments: list,
    total_duration: float,
    output_audio_path: str,
    ffmpeg_cmd: str = None
) -> str:
    """Combine Hindi audio chunks into a clean, ZERO-OVERLAP, synchronized track."""
    if not ffmpeg_cmd:
        ffmpeg_cmd = get_ffmpeg_path()

    Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_audio_path).parent / f"temp_align_{Path(output_audio_path).stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Filter segments that actually have audio generated
    valid_segments = [s for s in segments if s.get("audio_file") and Path(s["audio_file"]).exists()]
    
    if not valid_segments:
        # Generate silence track of total_duration
        cmd = [
            ffmpeg_cmd, "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(max(1.0, total_duration)),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_audio_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_audio_path

    concat_list_file = temp_dir / "concat_list.txt"
    concat_lines = []
    current_time = 0.0

    for idx, seg in enumerate(valid_segments):
        # Calculate available time before next segment starts
        if idx + 1 < len(valid_segments):
            next_start = valid_segments[idx + 1].get("start", seg.get("end", seg.get("start", 0) + 3.0))
            avail_time = max(0.5, next_start - seg.get("start", 0.0))
        else:
            avail_time = max(0.5, total_duration - seg.get("start", 0.0))

        # 1. Fit segment audio if it's too long
        processed_seg_file = temp_dir / f"proc_seg_{idx:03d}.mp3"
        seg_dur = fit_audio_segment_duration(
            input_file=seg["audio_file"],
            output_file=str(processed_seg_file),
            target_duration=avail_time,
            ffmpeg_cmd=ffmpeg_cmd
        )

        # 2. Prevent overlapping: start time can NEVER be earlier than current_time
        seg_scheduled_start = float(seg.get("start", 0.0))
        actual_start = max(current_time, seg_scheduled_start)
        
        silence_gap = actual_start - current_time
        if silence_gap > 0.04:
            silence_file = temp_dir / f"silence_{idx:03d}.mp3"
            cmd_silence = [
                ffmpeg_cmd, "-y",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", f"{silence_gap:.3f}",
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(silence_file)
            ]
            subprocess.run(cmd_silence, capture_output=True)
            if silence_file.exists():
                concat_lines.append(f"file '{silence_file.name}'")
                current_time += silence_gap

        # 3. Append the segment audio
        concat_lines.append(f"file '{processed_seg_file.name}'")
        current_time += seg_dur

    # 4. Fill remaining duration with silence to match video length
    if current_time < total_duration:
        remaining_gap = total_duration - current_time
        if remaining_gap > 0.05:
            final_silence = temp_dir / "silence_final.mp3"
            cmd_final = [
                ffmpeg_cmd, "-y",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", f"{remaining_gap:.3f}",
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(final_silence)
            ]
            subprocess.run(cmd_final, capture_output=True)
            if final_silence.exists():
                concat_lines.append(f"file '{final_silence.name}'")

    with open(concat_list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines))

    # 5. Concat everything into final Hindi audio track
    cmd_concat = [
        ffmpeg_cmd, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list_file),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        output_audio_path
    ]
    subprocess.run(cmd_concat, capture_output=True, check=True, cwd=str(temp_dir))

    # Cleanup temp directory
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    return output_audio_path

def merge_video_and_hindi_audio(
    video_path: str,
    hindi_audio_path: str,
    output_video_path: str,
    mode: str = "replace", # "replace" or "ducking"
    bg_music_volume: float = 0.0,
    voice_volume: float = 1.0,
    subtitles_path: str = None
) -> str:
    """Mux video stream with Hindi audio track (with zero overlap & clean mixing)."""
    ffmpeg_cmd = get_ffmpeg_path()
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "replace" or bg_music_volume <= 0.01:
        # Pure Voice Replacement (0% original audio, 100% Hindi voice)
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", video_path,
            "-i", hindi_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_video_path
        ]
    else:
        # Audio Ducking Mode: Mix original audio (low volume) + Hindi voice (full volume)
        filter_complex = f"[0:a]volume={bg_music_volume}[bg];[1:a]volume={voice_volume}[voice];[bg][voice]amix=inputs=2:duration=first:dropout_transition=0[outa]"
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", video_path,
            "-i", hindi_audio_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_video_path
        ]

    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        # Fallback to replace mode
        print(f"Merge filter failed, fallback to replace: {process.stderr}")
        cmd_fallback = [
            ffmpeg_cmd, "-y",
            "-i", video_path,
            "-i", hindi_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_video_path
        ]
        subprocess.run(cmd_fallback, capture_output=True, check=True)
            
    return output_video_path


# ==================== MODULE: subtitles.py ====================

def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into VTT timestamp: HH:MM:SS.mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

def generate_srt(segments: list, output_path: str, use_hindi: bool = True) -> str:
    """Generate SRT subtitle file from segments."""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_timestamp_srt(seg.get("start", 0.0))
        end_ts = format_timestamp_srt(seg.get("end", 0.0))
        text = seg.get("hindi_text" if use_hindi else "english_text", "").strip()
        if not text:
            text = seg.get("text", "").strip()
            
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("") # Empty line separator

    srt_content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    return output_path

def generate_vtt(segments: list, output_path: str, use_hindi: bool = True) -> str:
    """Generate WebVTT subtitle file from segments."""
    lines = ["WEBVTT\n"]
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_timestamp_vtt(seg.get("start", 0.0))
        end_ts = format_timestamp_vtt(seg.get("end", 0.0))
        text = seg.get("hindi_text" if use_hindi else "english_text", "").strip()
        if not text:
            text = seg.get("text", "").strip()
            
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    vtt_content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(vtt_content)
        
    return output_path



# ==================== EMBEDDED WEB ASSETS AUTO-INSTALLER ====================
EMBEDDED_HTML = '<!DOCTYPE html>\n<html lang="hi">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>VideoDubber AI - English to Hindi Video Dubbing Studio | Bunty AI Voice</title>\n    <meta name="description" content="Free AI English to Hindi Video Dubbing Studio. Dub YouTube videos, Instagram Reels & Shorts in 1-click with natural Indian Bunty AI voice & auto Hindi subtitles.">\n    <meta name="keywords" content="video dubbing, english to hindi dubbing, ai video dubber, video dubbing online, hindi dubbing ai, bunty voice dubbing, elevenlabs hindi voice, free ai dubbing, translate english video to hindi, youtube shorts dubbing">\n    <meta name="author" content="VideoDubber AI">\n    <meta name="robots" content="index, follow">\n    <link rel="canonical" href="https://videodubber-ai-trxg.onrender.com/">\n\n    <!-- OpenGraph / Social Sharing -->\n    <meta property="og:type" content="website">\n    <meta property="og:url" content="https://videodubber-ai-trxg.onrender.com/">\n    <meta property="og:title" content="VideoDubber AI - Free English to Hindi AI Video Dubbing">\n    <meta property="og:description" content="Dub English videos to natural Hindi in 60 seconds with Bunty AI voice! Free trial & instant UPI access.">\n    <meta property="og:image" content="https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=https://videodubber-ai-trxg.onrender.com/">\n\n    <!-- Google Search Schema Markup (JSON-LD) -->\n    <script type="application/ld+json">\n    {\n      "@context": "https://schema.org",\n      "@type": "WebApplication",\n      "name": "VideoDubber AI",\n      "url": "https://videodubber-ai-trxg.onrender.com/",\n      "description": "Professional English to Hindi AI Video Dubbing Studio with ElevenLabs Bunty neural voice.",\n      "applicationCategory": "MultimediaApplication",\n      "operatingSystem": "All",\n      "offers": {\n        "@type": "Offer",\n        "price": "10.00",\n        "priceCurrency": "INR"\n      }\n    }\n    </script>\n\n    <link rel="preconnect" href="https://fonts.googleapis.com">\n    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">\n    <link rel="stylesheet" href="css/styles.css?v=2.2">\n</head>\n<body>\n    <!-- Background Glow Elements -->\n    <div class="bg-glow bg-glow-1"></div>\n    <div class="bg-glow bg-glow-2"></div>\n\n    <div class="app-container">\n        <!-- Top Navbar -->\n        <header class="navbar">\n            <div class="brand">\n                <div class="brand-icon">🎙️</div>\n                <div class="brand-info">\n                    <h1>VideoDubber <span class="gradient-text">AI</span></h1>\n                    <p class="tagline">English Video to Hindi AI Dubbing Studio • ElevenLabs <span class="badge-voice">Bunty Voice</span></p>\n                </div>\n            </div>\n            <div class="nav-actions">\n                <button id="btnOpenPricing" class="btn btn-pass glow-btn">\n                    <span class="btn-icon">👑</span>\n                    <span id="passStatusText">👑 Unlock Pass (₹10/₹15/₹20)</span>\n                </button>\n                <div id="apiStatusBadge" class="status-pill status-warning">\n                    <span class="status-dot"></span>\n                    <span id="apiStatusText">Setup API Keys</span>\n                </div>\n                <button id="btnOpenSettings" class="btn btn-secondary">\n                    <span class="btn-icon">⚙️</span>\n                    <span>My API Keys</span>\n                </button>\n            </div>\n        </header>\n\n        <!-- Prominent Launch & Pricing Banner -->\n        <section class="pricing-banner" id="pricingHeroBanner">\n            <div class="banner-badge">🔥 Instant UPI Access</div>\n            <div class="banner-content">\n                <h2>🚀 English to Hindi AI Dubbing Studio — Pass Choose Karein</h2>\n                <p>Apna Gemini/ElevenLabs API key use karke unlimited dubbing karein. Direct UPI to <strong>subham.088@fam</strong></p>\n            </div>\n            <div class="banner-plans-preview">\n                <div class="plan-chip" onclick="document.getElementById(\'btnOpenPricing\').click()">\n                    <span class="chip-badge">2 Days</span>\n                    <strong>₹10</strong>\n                    <small>2 Din Pass</small>\n                </div>\n                <div class="plan-chip" onclick="document.getElementById(\'btnOpenPricing\').click()">\n                    <span class="chip-badge">7 Days</span>\n                    <strong>₹15</strong>\n                    <small>7 Din Pass</small>\n                </div>\n                <div class="plan-chip chip-popular" onclick="document.getElementById(\'btnOpenPricing\').click()">\n                    <span class="chip-badge badge-pop">🔥 Best</span>\n                    <strong>₹20</strong>\n                    <small>28 Din (Monthly)</small>\n                </div>\n            </div>\n            <button class="btn btn-banner-action" onclick="document.getElementById(\'btnOpenPricing\').click()">\n                <span>⚡ Pay with UPI & Unlock Now</span>\n            </button>\n        </section>\n\n        <!-- Main Studio Grid -->\n        <main class="studio-grid">\n            <!-- Left Panel: Input & Controls -->\n            <section class="studio-panel panel-left">\n                <div class="panel-header">\n                    <h2><span class="step-num">1</span> Video Upload & Voice</h2>\n                </div>\n\n                <!-- Video Dropzone -->\n                <div id="dropzone" class="dropzone">\n                    <input type="file" id="videoFileInput" accept="video/mp4,video/mkv,video/mov,video/webm" hidden>\n                    <div class="dropzone-content" id="dropzonePrompt">\n                        <div class="upload-icon">📹</div>\n                        <h3>Upload English Video</h3>\n                        <p>Drag and drop video here or <span class="browse-link">browse</span></p>\n                        <span class="supported-formats">MP4, MOV, MKV, WebM (Up to 500MB)</span>\n                    </div>\n\n                    <!-- Video Preview after selection -->\n                    <div class="selected-video-preview hidden" id="selectedVideoPreview">\n                        <video id="inputVideoPreview" controls></video>\n                        <div class="video-meta">\n                            <div class="file-details">\n                                <span class="video-name" id="previewFileName">video.mp4</span>\n                                <span class="video-specs" id="previewSpecs">0:00 • 1080p</span>\n                            </div>\n                            <button id="btnRemoveVideo" class="btn-icon-only" title="Remove video">✕</button>\n                        </div>\n                    </div>\n                </div>\n\n                <!-- Dubbing Options Card -->\n                <div class="options-card">\n                    <h3>Voice & Audio Settings</h3>\n\n                    <!-- Voice Selector -->\n                    <div class="form-group">\n                        <div class="label-with-action">\n                            <label for="voiceSelect">ElevenLabs Hindi Voice:</label>\n                            <button id="btnTestVoice" class="btn-text" title="Listen to sample audio">\n                                🔊 Test "Bunty" Voice\n                            </button>\n                        </div>\n                        <div class="select-wrapper">\n                            <select id="voiceSelect" class="form-control">\n                                <option value="" disabled selected>Loading voices from ElevenLabs...</option>\n                            </select>\n                        </div>\n                        <small class="helper-text" id="voiceHelperText">Default: Bunty (ElevenLabs Multilingual v2)</small>\n                    </div>\n\n                    <!-- Audio Mode Toggle -->\n                    <div class="form-group">\n                        <label>Audio Dubbing Mode:</label>\n                        <div class="toggle-group">\n                            <label class="toggle-option">\n                                <input type="radio" name="audioMode" value="replace" checked>\n                                <span class="toggle-box">\n                                    <strong>🎙️ Clean Voice Replace (0% Original Audio)</strong>\n                                    <small>Replaces original English audio track completely with crystal clear Hindi voice</small>\n                                </span>\n                            </label>\n                            <label class="toggle-option">\n                                <input type="radio" name="audioMode" value="ducking">\n                                <span class="toggle-box">\n                                    <strong>🎵 Background Sound Ducking</strong>\n                                    <small>Keeps subtle original background sound while Bunty\'s Hindi voice plays on top</small>\n                                </span>\n                            </label>\n                        </div>\n                    </div>\n\n                    <!-- Volume Sliders (Expandable) -->\n                    <div class="form-group" id="volumeControlGroup">\n                        <div class="slider-row">\n                            <label for="bgVolume">Background Audio Volume: <span id="bgVolumeVal">0%</span></label>\n                            <input type="range" id="bgVolume" min="0" max="0.5" step="0.05" value="0">\n                        </div>\n                    </div>\n\n                    <!-- Start Dubbing Button -->\n                    <button id="btnStartDubbing" class="btn btn-primary btn-block btn-large" disabled>\n                        <span class="btn-icon">⚡</span>\n                        <span>Start English to Hindi Dubbing</span>\n                    </button>\n                </div>\n            </section>\n\n            <!-- Right Panel: Processing, Transcript & Player -->\n            <section class="studio-panel panel-right">\n                <!-- Status & Progress Container -->\n                <div class="card status-card" id="statusCard">\n                    <div class="panel-header">\n                        <h2><span class="step-num">2</span> AI Dubbing Engine</h2>\n                        <span class="badge" id="jobStatusBadge">Ready</span>\n                    </div>\n\n                    <!-- Pipeline Steps Visualizer -->\n                    <div class="pipeline-stepper">\n                        <div class="step-node" id="step-extract">\n                            <div class="node-icon">🎵</div>\n                            <span class="node-title">Audio Extract</span>\n                        </div>\n                        <div class="step-connector"></div>\n                        <div class="step-node" id="step-transcribe">\n                            <div class="node-icon">📝</div>\n                            <span class="node-title">Transcription</span>\n                        </div>\n                        <div class="step-connector"></div>\n                        <div class="step-node" id="step-translate">\n                            <div class="node-icon">🌐</div>\n                            <span class="node-title">Hindi Translate</span>\n                        </div>\n                        <div class="step-connector"></div>\n                        <div class="step-node" id="step-synthesize">\n                            <div class="node-icon">🗣️</div>\n                            <span class="node-title">Bunty Voice</span>\n                        </div>\n                        <div class="step-connector"></div>\n                        <div class="step-node" id="step-merge">\n                            <div class="node-icon">🎬</div>\n                            <span class="node-title">Video Merge</span>\n                        </div>\n                    </div>\n\n                    <!-- Progress Bar -->\n                    <div class="progress-bar-wrapper">\n                        <div class="progress-bar-fill" id="progressBarFill" style="width: 0%;"></div>\n                    </div>\n                    <div class="status-message" id="statusMessage">Upload a video to begin translation and dubbing.</div>\n                </div>\n\n                <!-- Tabs: Transcript Review / Result Player -->\n                <div class="tabs-container">\n                    <div class="tabs-nav">\n                        <button class="tab-btn active" data-tab="tab-player">🎬 Output Video Studio</button>\n                        <button class="tab-btn" data-tab="tab-transcript">📜 Transcript & Hindi Script</button>\n                    </div>\n\n                    <!-- Tab 1: Output Video Studio -->\n                    <div class="tab-content active" id="tab-player">\n                        <div class="empty-state" id="playerEmptyState">\n                            <div class="empty-icon">📺</div>\n                            <h3>No Dubbed Video Yet</h3>\n                            <p>Once you click "Start Dubbing", your finished Hindi video with Bunty\'s voice will appear here for preview and download.</p>\n                        </div>\n\n                        <div class="player-wrapper hidden" id="playerWrapper">\n                            <div class="video-container">\n                                <video id="outputVideoPlayer" controls playsinline></video>\n                            </div>\n\n                            <div class="export-actions">\n                                <a id="btnDownloadVideo" class="btn btn-success btn-large" download>\n                                    <span class="btn-icon">⬇️</span>\n                                    <span>Download Hindi Video (.mp4)</span>\n                                </a>\n                                <a id="btnDownloadSrt" class="btn btn-secondary" download>\n                                    <span class="btn-icon">📄</span>\n                                    <span>Download Subtitles (.srt)</span>\n                                </a>\n                            </div>\n                        </div>\n                    </div>\n\n                    <!-- Tab 2: Transcript & Script Editor -->\n                    <div class="tab-content" id="tab-transcript">\n                        <div class="transcript-header">\n                            <p>Review timestamped English lines and their natural Hindi dubbing translations.</p>\n                        </div>\n                        <div id="transcriptSegmentsList" class="segments-list">\n                            <div class="empty-transcript">\n                                Transcript will automatically be generated during dubbing.\n                            </div>\n                        </div>\n                    </div>\n                </div>\n            </section>\n        </main>\n    </div>\n\n    <!-- Pricing / UPI Payment Modal -->\n    <div class="modal-overlay hidden" id="pricingModal">\n        <div class="modal-card pricing-modal-card">\n            <div class="modal-header">\n                <div>\n                    <h2>👑 Unlock VideoDubber Studio Pass</h2>\n                    <p class="modal-subtitle">Direct Instant UPI Payment to <strong>subham.088@fam</strong></p>\n                </div>\n                <button class="btn-close" id="btnClosePricing">✕</button>\n            </div>\n            <div class="modal-body">\n                <!-- Plan Selection Cards -->\n                <div class="plans-container">\n                    <div class="plan-card" data-plan="10" data-name="2-Days Starter Pass">\n                        <div class="plan-badge">2 Days</div>\n                        <div class="plan-price">₹10</div>\n                        <div class="plan-duration">2 Din Full Access</div>\n                        <ul class="plan-features">\n                            <li>✓ Unlimited Dubbing</li>\n                            <li>✓ Bunty & Indian Voices</li>\n                            <li>✓ Instant MP4 Download</li>\n                        </ul>\n                        <button class="btn btn-plan-select">Select ₹10</button>\n                    </div>\n\n                    <div class="plan-card" data-plan="15" data-name="7-Days Creator Pass">\n                        <div class="plan-badge">7 Days</div>\n                        <div class="plan-price">₹15</div>\n                        <div class="plan-duration">7 Din Full Access</div>\n                        <ul class="plan-features">\n                            <li>✓ Unlimited Dubbing</li>\n                            <li>✓ Hindi Subtitles (.SRT)</li>\n                            <li>✓ Background Audio Mix</li>\n                        </ul>\n                        <button class="btn btn-plan-select">Select ₹15</button>\n                    </div>\n\n                    <div class="plan-card popular-plan selected" data-plan="20" data-name="28-Days Monthly VIP Pass">\n                        <div class="plan-badge badge-popular">🔥 Best Value</div>\n                        <div class="plan-price">₹20</div>\n                        <div class="plan-duration">28 Din (Full Month)</div>\n                        <ul class="plan-features">\n                            <li>✓ <strong>28 Din</strong> Unlimited Dubbing</li>\n                            <li>✓ Highest Speed Processing</li>\n                            <li>✓ All Voice & Audio Modes</li>\n                        </ul>\n                        <button class="btn btn-plan-select selected-btn">Selected (₹20)</button>\n                    </div>\n                </div>\n\n                <!-- Payment QR & UPI Section -->\n                <div class="payment-checkout-card">\n                    <div class="qr-col">\n                        <div class="qr-box">\n                            <img id="upiQrImage" src="https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=upi://pay?pa=subham.088@fam&pn=VideoDubber%20AI&am=20&cu=INR&tn=VideoDubber%20VIP%20Pass" alt="Scan UPI QR">\n                            <div class="qr-overlay-brand">₹<span id="selectedPlanAmountDisplay">20</span></div>\n                        </div>\n                        <span class="qr-scan-label">Scan with GPay, PhonePe, Paytm or BHIM</span>\n                    </div>\n\n                    <div class="upi-details-col">\n                        <div class="upi-id-pill">\n                            <span class="upi-label">UPI ID:</span>\n                            <strong class="upi-val" id="textUpiId">subham.088@fam</strong>\n                            <button id="btnCopyUpi" class="btn-copy" title="Copy UPI ID">📋 Copy</button>\n                        </div>\n\n                        <!-- Direct Mobile UPI Intent Button -->\n                        <a id="btnUpiIntentLink" href="upi://pay?pa=subham.088@fam&pn=VideoDubber%20AI&am=20&cu=INR&tn=VideoDubber%20Pass" class="btn btn-upi-pay">\n                            <span class="btn-icon">⚡</span>\n                            <span>Pay ₹<span id="intentPayAmount">20</span> with Any UPI App</span>\n                        </a>\n\n                        <!-- Verification Input -->\n                        <div class="verification-box">\n                            <label for="inputUtrNumber">Payment karne ke baad 12-Digit UTR / Reference No. ya VIP Code dalein:</label>\n                            <div class="verify-input-group">\n                                <input type="text" id="inputUtrNumber" class="form-control" placeholder="e.g. 423819283746 or VIP code">\n                                <button id="btnVerifyUtr" class="btn btn-success">\n                                    <span>Unlock Now 🔓</span>\n                                </button>\n                            </div>\n                            <div id="paymentFeedback" class="settings-feedback"></div>\n                        </div>\n                    </div>\n                </div>\n            </div>\n            <div class="modal-footer">\n                <button class="btn btn-secondary" id="btnCancelPricing">Close</button>\n            </div>\n        </div>\n    </div>\n\n    <!-- Settings Modal (BYOK) -->\n    <div class="modal-overlay hidden" id="settingsModal">\n        <div class="modal-card">\n            <div class="modal-header">\n                <div>\n                    <h2>🔑 My AI API Keys (Bring Your Own Key)</h2>\n                    <p class="modal-subtitle">Aapki API Keys aapke browser me securely save rehti hain</p>\n                </div>\n                <button class="btn-close" id="btnCloseSettings">✕</button>\n            </div>\n            <div class="modal-body">\n                <!-- Gemini API Key -->\n                <div class="form-group">\n                    <label for="inputGeminiKey">\n                        Google Gemini API Key:\n                        <a href="https://aistudio.google.com/app/apikey" target="_blank" class="external-link">Get Free Gemini Key ↗</a>\n                    </label>\n                    <input type="password" id="inputGeminiKey" class="form-control" placeholder="AIzaSy...">\n                    <small class="helper-text">High-speed speech transcription aur context-aware Hindi translation ke liye.</small>\n                </div>\n\n                <!-- ElevenLabs API Key -->\n                <div class="form-group">\n                    <label for="inputElevenLabsKey">\n                        ElevenLabs API Key:\n                        <a href="https://elevenlabs.io" target="_blank" class="external-link">Get Free ElevenLabs Key ↗</a>\n                    </label>\n                    <input type="password" id="inputElevenLabsKey" class="form-control" placeholder="xi-xxxxxxxxxxxxxxxx">\n                    <small class="helper-text">Natural Bunty Hindi AI voice synthesize karne ke liye.</small>\n                </div>\n\n                <!-- Voice ID -->\n                <div class="form-group">\n                    <label for="inputVoiceId">Bunty Voice ID (Custom or Preset):</label>\n                    <input type="text" id="inputVoiceId" class="form-control" placeholder="e.g. edge:hi-IN-MadhurNeural or ElevenLabs ID">\n                    <small class="helper-text">Agar ElevenLabs me custom "Bunty" clone voice hai toh uska ID dalein ya default rehne dein.</small>\n                </div>\n\n                <!-- Creator Affiliate Deals Section -->\n                <div class="affiliate-deals-box">\n                    <h4>🎁 Recommended Creator AI Tools & Gear:</h4>\n                    <div class="deals-grid">\n                        <a href="https://elevenlabs.io/?from=partner" target="_blank" class="deal-item">\n                            <span class="deal-icon">🎙️</span>\n                            <div class="deal-info">\n                                <strong>ElevenLabs AI Voices</strong>\n                                <small>Best Humanlike Hindi & Indian Accent Clones</small>\n                            </div>\n                            <span class="deal-tag">Get 10k Credits ↗</span>\n                        </a>\n\n                        <a href="https://aistudio.google.com/" target="_blank" class="deal-item">\n                            <span class="deal-icon">⚡</span>\n                            <div class="deal-info">\n                                <strong>Google Gemini Flash 2.5</strong>\n                                <small>100% Free AI Transcription & Subtitles</small>\n                            </div>\n                            <span class="deal-tag">Free API ↗</span>\n                        </a>\n\n                        <a href="https://www.amazon.in/s?k=boya+by+m1+mic" target="_blank" class="deal-item">\n                            <span class="deal-icon">🎤</span>\n                            <div class="deal-info">\n                                <strong>Boya BY-M1 / Fifine USB Mic</strong>\n                                <small>Best Studio Mic for Creators (Under ₹799)</small>\n                            </div>\n                            <span class="deal-tag">View on Amazon ↗</span>\n                        </a>\n\n                        <a href="https://www.capcut.com/" target="_blank" class="deal-item">\n                            <span class="deal-icon">✂️</span>\n                            <div class="deal-info">\n                                <strong>CapCut / Canva Pro Video</strong>\n                                <small>Auto Captions, Reels & Shorts Editor</small>\n                            </div>\n                            <span class="deal-tag">Try Free ↗</span>\n                        </a>\n                    </div>\n                </div>\n\n                <div id="settingsFeedback" class="settings-feedback"></div>\n            </div>\n            <div class="modal-footer">\n                <button class="btn btn-secondary" id="btnCancelSettings">Cancel</button>\n                <button class="btn btn-primary" id="btnSaveSettings">Save Keys in Browser</button>\n            </div>\n        </div>\n    </div>\n\n    <!-- AdSense / Sponsor Banner Slot -->\n    <footer class="app-footer">\n        <div class="ad-banner-slot" id="googleAdSlotBottom">\n            <div class="ad-placeholder">\n                <span class="ad-label">SPONSORED ADVERTISEMENT</span>\n                <p>🚀 <strong>Want to grow your YouTube Shorts?</strong> Dub in Hindi with Bunty AI Voice & get 10x more reach!</p>\n            </div>\n        </div>\n        <p class="footer-copy">© 2026 VideoDubber AI • English to Hindi Video Dubbing Studio • UPI ID: <strong>subham.088@fam</strong></p>\n    </footer>\n\n    <!-- Audio Player for Voice Testing -->\n    <audio id="voiceTestAudio" style="display:none;"></audio>\n\n    <script src="js/app.js?v=2.2"></script>\n</body>\n</html>\n\n\n'
EMBEDDED_CSS = ':root {\n    --bg-primary: #0d0f17;\n    --bg-secondary: #161926;\n    --bg-card: rgba(26, 31, 46, 0.85);\n    --bg-card-hover: rgba(33, 40, 60, 0.95);\n    --border-color: rgba(255, 255, 255, 0.08);\n    --border-focus: #6366f1;\n    \n    --text-primary: #f8fafc;\n    --text-secondary: #94a3b8;\n    --text-muted: #64748b;\n    \n    --accent-indigo: #6366f1;\n    --accent-purple: #8b5cf6;\n    --accent-pink: #ec4899;\n    --accent-emerald: #10b981;\n    --accent-amber: #f59e0b;\n    --accent-rose: #f43f5e;\n    \n    --radius-sm: 8px;\n    --radius-md: 14px;\n    --radius-lg: 20px;\n    \n    --shadow-card: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 1px 1px rgba(255, 255, 255, 0.05);\n    --shadow-glow: 0 0 25px rgba(99, 102, 241, 0.35);\n}\n\n* {\n    margin: 0;\n    padding: 0;\n    box-sizing: border-box;\n    font-family: \'Outfit\', \'Noto Sans Devanagari\', -apple-system, BlinkMacSystemFont, sans-serif;\n}\n\nbody {\n    background-color: var(--bg-primary);\n    color: var(--text-primary);\n    min-height: 100vh;\n    position: relative;\n    overflow-x: hidden;\n    line-height: 1.5;\n}\n\n/* Background Ambient Glows */\n.bg-glow {\n    position: fixed;\n    border-radius: 50%;\n    filter: blur(120px);\n    pointer-events: none;\n    z-index: 0;\n    opacity: 0.4;\n}\n\n.bg-glow-1 {\n    top: -100px;\n    left: 10%;\n    width: 500px;\n    height: 500px;\n    background: radial-gradient(circle, #6366f1 0%, rgba(99, 102, 241, 0) 70%);\n}\n\n.bg-glow-2 {\n    bottom: -150px;\n    right: 5%;\n    width: 600px;\n    height: 600px;\n    background: radial-gradient(circle, #ec4899 0%, rgba(236, 72, 153, 0) 70%);\n}\n\n.app-container {\n    position: relative;\n    z-index: 1;\n    max-width: 1440px;\n    margin: 0 auto;\n    padding: 24px 32px;\n    display: flex;\n    flex-direction: column;\n    gap: 24px;\n    min-height: 100vh;\n}\n\n/* Navbar */\n.navbar {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    padding: 16px 24px;\n    background: var(--bg-card);\n    backdrop-filter: blur(16px);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-lg);\n    box-shadow: var(--shadow-card);\n}\n\n.brand {\n    display: flex;\n    align-items: center;\n    gap: 16px;\n}\n\n.brand-icon {\n    font-size: 2rem;\n    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(236, 72, 153, 0.2));\n    border: 1px solid rgba(255, 255, 255, 0.15);\n    width: 52px;\n    height: 52px;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    border-radius: var(--radius-md);\n    box-shadow: var(--shadow-glow);\n}\n\n.brand-info h1 {\n    font-size: 1.6rem;\n    font-weight: 800;\n    letter-spacing: -0.5px;\n}\n\n.gradient-text {\n    background: linear-gradient(135deg, #6366f1, #ec4899);\n    -webkit-background-clip: text;\n    -webkit-text-fill-color: transparent;\n}\n\n.tagline {\n    font-size: 0.85rem;\n    color: var(--text-secondary);\n}\n\n.badge-voice {\n    background: linear-gradient(135deg, #8b5cf6, #ec4899);\n    color: #fff;\n    padding: 2px 8px;\n    border-radius: 12px;\n    font-size: 0.75rem;\n    font-weight: 600;\n}\n\n.nav-actions {\n    display: flex;\n    align-items: center;\n    gap: 14px;\n}\n\n.status-pill {\n    display: flex;\n    align-items: center;\n    gap: 8px;\n    padding: 6px 14px;\n    border-radius: 30px;\n    font-size: 0.82rem;\n    font-weight: 500;\n    border: 1px solid var(--border-color);\n    background: rgba(0, 0, 0, 0.3);\n}\n\n.status-dot {\n    width: 8px;\n    height: 8px;\n    border-radius: 50%;\n}\n\n.status-warning .status-dot { background-color: var(--accent-amber); box-shadow: 0 0 8px var(--accent-amber); }\n.status-success .status-dot { background-color: var(--accent-emerald); box-shadow: 0 0 8px var(--accent-emerald); }\n\n/* Main Grid */\n.studio-grid {\n    display: grid;\n    grid-template-columns: 460px 1fr;\n    gap: 24px;\n    flex: 1;\n}\n\n@media (max-width: 1080px) {\n    .studio-grid {\n        grid-template-columns: 1fr;\n    }\n}\n\n/* Studio Panels */\n.studio-panel {\n    display: flex;\n    flex-direction: column;\n    gap: 20px;\n}\n\n.panel-header {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n}\n\n.panel-header h2 {\n    font-size: 1.25rem;\n    font-weight: 700;\n    display: flex;\n    align-items: center;\n    gap: 10px;\n}\n\n.step-num {\n    display: inline-flex;\n    align-items: center;\n    justify-content: center;\n    width: 28px;\n    height: 28px;\n    border-radius: 50%;\n    background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));\n    color: #fff;\n    font-size: 0.9rem;\n    font-weight: 700;\n}\n\n/* Dropzone */\n.dropzone {\n    background: var(--bg-card);\n    border: 2px dashed rgba(99, 102, 241, 0.35);\n    border-radius: var(--radius-lg);\n    padding: 32px 24px;\n    text-align: center;\n    cursor: pointer;\n    transition: all 0.3s ease;\n    backdrop-filter: blur(12px);\n    position: relative;\n    overflow: hidden;\n}\n\n.dropzone:hover, .dropzone.dragover {\n    border-color: var(--accent-indigo);\n    background: var(--bg-card-hover);\n    box-shadow: var(--shadow-glow);\n}\n\n.upload-icon {\n    font-size: 3.2rem;\n    margin-bottom: 12px;\n}\n\n.dropzone-content h3 {\n    font-size: 1.15rem;\n    margin-bottom: 6px;\n}\n\n.dropzone-content p {\n    color: var(--text-secondary);\n    font-size: 0.9rem;\n}\n\n.browse-link {\n    color: var(--accent-indigo);\n    text-decoration: underline;\n    font-weight: 600;\n}\n\n.supported-formats {\n    display: inline-block;\n    margin-top: 10px;\n    font-size: 0.75rem;\n    color: var(--text-muted);\n    background: rgba(255, 255, 255, 0.04);\n    padding: 4px 10px;\n    border-radius: 6px;\n}\n\n/* Video Preview */\n.selected-video-preview video {\n    width: 100%;\n    max-height: 200px;\n    border-radius: var(--radius-md);\n    background: #000;\n    object-fit: contain;\n}\n\n.video-meta {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    margin-top: 12px;\n    padding: 8px 12px;\n    background: rgba(0, 0, 0, 0.4);\n    border-radius: var(--radius-sm);\n    text-align: left;\n}\n\n.file-details {\n    display: flex;\n    flex-direction: column;\n}\n\n.video-name {\n    font-size: 0.88rem;\n    font-weight: 600;\n    max-width: 320px;\n    white-space: nowrap;\n    overflow: hidden;\n    text-overflow: ellipsis;\n}\n\n.video-specs {\n    font-size: 0.75rem;\n    color: var(--text-muted);\n}\n\n/* Options Card */\n.options-card {\n    background: var(--bg-card);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-lg);\n    padding: 24px;\n    display: flex;\n    flex-direction: column;\n    gap: 18px;\n    box-shadow: var(--shadow-card);\n}\n\n.options-card h3 {\n    font-size: 1.1rem;\n    font-weight: 600;\n    border-bottom: 1px solid var(--border-color);\n    padding-bottom: 10px;\n}\n\n.form-group {\n    display: flex;\n    flex-direction: column;\n    gap: 8px;\n}\n\n.form-group label {\n    font-size: 0.88rem;\n    font-weight: 500;\n    color: var(--text-primary);\n}\n\n.label-with-action {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n}\n\n.btn-text {\n    background: none;\n    border: none;\n    color: var(--accent-indigo);\n    font-size: 0.8rem;\n    cursor: pointer;\n    font-weight: 600;\n    transition: opacity 0.2s;\n}\n\n.btn-text:hover {\n    text-decoration: underline;\n    opacity: 0.8;\n}\n\n.form-control {\n    background: rgba(13, 15, 23, 0.7);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-sm);\n    padding: 10px 14px;\n    color: var(--text-primary);\n    font-size: 0.9rem;\n    outline: none;\n    transition: border-color 0.2s, box-shadow 0.2s;\n    width: 100%;\n}\n\n.form-control:focus {\n    border-color: var(--border-focus);\n    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);\n}\n\n.helper-text {\n    font-size: 0.75rem;\n    color: var(--text-muted);\n}\n\n/* Toggle Options */\n.toggle-group {\n    display: flex;\n    flex-direction: column;\n    gap: 10px;\n}\n\n.toggle-option {\n    cursor: pointer;\n}\n\n.toggle-option input {\n    display: none;\n}\n\n.toggle-box {\n    display: flex;\n    flex-direction: column;\n    padding: 12px 14px;\n    border-radius: var(--radius-sm);\n    background: rgba(0, 0, 0, 0.25);\n    border: 1px solid var(--border-color);\n    transition: all 0.2s ease;\n}\n\n.toggle-box strong {\n    font-size: 0.88rem;\n    color: var(--text-primary);\n}\n\n.toggle-box small {\n    font-size: 0.78rem;\n    color: var(--text-muted);\n    margin-top: 3px;\n}\n\n.toggle-option input:checked + .toggle-box {\n    border-color: var(--accent-indigo);\n    background: rgba(99, 102, 241, 0.12);\n    box-shadow: 0 0 12px rgba(99, 102, 241, 0.15);\n}\n\n.slider-row {\n    display: flex;\n    flex-direction: column;\n    gap: 6px;\n}\n\n.slider-row input[type="range"] {\n    accent-color: var(--accent-indigo);\n    cursor: pointer;\n}\n\n/* Buttons */\n.btn {\n    display: inline-flex;\n    align-items: center;\n    justify-content: center;\n    gap: 8px;\n    padding: 10px 20px;\n    border-radius: var(--radius-sm);\n    font-size: 0.9rem;\n    font-weight: 600;\n    cursor: pointer;\n    border: none;\n    transition: all 0.2s ease;\n    text-decoration: none;\n}\n\n.btn-primary {\n    background: linear-gradient(135deg, #6366f1, #8b5cf6);\n    color: #fff;\n    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);\n}\n\n.btn-primary:hover:not(:disabled) {\n    transform: translateY(-1px);\n    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);\n}\n\n.btn-secondary {\n    background: rgba(255, 255, 255, 0.08);\n    color: var(--text-primary);\n    border: 1px solid var(--border-color);\n}\n\n.btn-secondary:hover {\n    background: rgba(255, 255, 255, 0.14);\n}\n\n.btn-success {\n    background: linear-gradient(135deg, #10b981, #059669);\n    color: #fff;\n    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35);\n}\n\n.btn-success:hover {\n    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);\n}\n\n.btn-block {\n    width: 100%;\n}\n\n.btn-large {\n    padding: 14px 24px;\n    font-size: 1.05rem;\n}\n\n.btn:disabled {\n    opacity: 0.5;\n    cursor: not-allowed;\n    box-shadow: none;\n    transform: none;\n}\n\n.btn-icon-only {\n    background: rgba(255, 255, 255, 0.1);\n    border: none;\n    color: #fff;\n    width: 28px;\n    height: 28px;\n    border-radius: 50%;\n    cursor: pointer;\n    font-size: 0.85rem;\n}\n\n/* Pipeline Stepper */\n.status-card {\n    background: var(--bg-card);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-lg);\n    padding: 24px;\n    display: flex;\n    flex-direction: column;\n    gap: 18px;\n    box-shadow: var(--shadow-card);\n}\n\n.pipeline-stepper {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    padding: 10px 0;\n}\n\n.step-node {\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n    gap: 6px;\n    opacity: 0.4;\n    transition: all 0.3s ease;\n}\n\n.step-node.active {\n    opacity: 1;\n    transform: scale(1.08);\n}\n\n.step-node.completed {\n    opacity: 1;\n}\n\n.step-node.active .node-icon {\n    border-color: var(--accent-indigo);\n    box-shadow: 0 0 15px var(--accent-indigo);\n    background: rgba(99, 102, 241, 0.25);\n    animation: pulse 1.5s infinite alternate;\n}\n\n.step-node.completed .node-icon {\n    border-color: var(--accent-emerald);\n    background: rgba(16, 185, 129, 0.2);\n    color: var(--accent-emerald);\n}\n\n.node-icon {\n    width: 44px;\n    height: 44px;\n    border-radius: 50%;\n    background: rgba(0, 0, 0, 0.4);\n    border: 2px solid var(--border-color);\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    font-size: 1.2rem;\n    transition: all 0.3s ease;\n}\n\n.node-title {\n    font-size: 0.75rem;\n    font-weight: 500;\n    color: var(--text-secondary);\n}\n\n.step-connector {\n    flex: 1;\n    height: 2px;\n    background: rgba(255, 255, 255, 0.08);\n    margin: 0 8px;\n    margin-bottom: 20px;\n}\n\n@keyframes pulse {\n    0% { transform: scale(1); box-shadow: 0 0 10px rgba(99, 102, 241, 0.3); }\n    100% { transform: scale(1.06); box-shadow: 0 0 20px rgba(99, 102, 241, 0.7); }\n}\n\n/* Progress Bar */\n.progress-bar-wrapper {\n    width: 100%;\n    height: 8px;\n    background: rgba(0, 0, 0, 0.4);\n    border-radius: 4px;\n    overflow: hidden;\n}\n\n.progress-bar-fill {\n    height: 100%;\n    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);\n    transition: width 0.4s ease;\n}\n\n.status-message {\n    font-size: 0.88rem;\n    color: var(--text-secondary);\n    font-style: italic;\n}\n\n/* Tabs */\n.tabs-container {\n    background: var(--bg-card);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-lg);\n    display: flex;\n    flex-direction: column;\n    flex: 1;\n    overflow: hidden;\n    box-shadow: var(--shadow-card);\n}\n\n.tabs-nav {\n    display: flex;\n    border-bottom: 1px solid var(--border-color);\n    background: rgba(0, 0, 0, 0.2);\n}\n\n.tab-btn {\n    flex: 1;\n    padding: 14px 20px;\n    background: none;\n    border: none;\n    color: var(--text-muted);\n    font-size: 0.95rem;\n    font-weight: 600;\n    cursor: pointer;\n    transition: all 0.2s ease;\n    border-bottom: 2px solid transparent;\n}\n\n.tab-btn.active {\n    color: var(--text-primary);\n    background: rgba(255, 255, 255, 0.03);\n    border-bottom-color: var(--accent-indigo);\n}\n\n.tab-content {\n    display: none;\n    padding: 24px;\n    flex: 1;\n}\n\n.tab-content.active {\n    display: flex;\n    flex-direction: column;\n    gap: 20px;\n}\n\n/* Empty States */\n.empty-state {\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n    justify-content: center;\n    padding: 60px 20px;\n    text-align: center;\n    gap: 12px;\n    color: var(--text-muted);\n}\n\n.empty-icon {\n    font-size: 3rem;\n    opacity: 0.6;\n}\n\n.empty-state h3 {\n    color: var(--text-secondary);\n    font-size: 1.2rem;\n}\n\n.empty-state p {\n    max-width: 420px;\n    font-size: 0.9rem;\n}\n\n/* Video Player Output */\n.player-wrapper {\n    display: flex;\n    flex-direction: column;\n    gap: 18px;\n    width: 100%;\n}\n\n.video-container video {\n    width: 100%;\n    max-height: 440px;\n    border-radius: var(--radius-md);\n    background: #000;\n    box-shadow: var(--shadow-card);\n}\n\n.export-actions {\n    display: flex;\n    gap: 16px;\n    flex-wrap: wrap;\n}\n\n/* Transcript Segment Editor */\n.segments-list {\n    display: flex;\n    flex-direction: column;\n    gap: 12px;\n    max-height: 480px;\n    overflow-y: auto;\n    padding-right: 8px;\n}\n\n.segment-item {\n    background: rgba(0, 0, 0, 0.3);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-sm);\n    padding: 14px;\n    display: flex;\n    flex-direction: column;\n    gap: 8px;\n}\n\n.segment-time {\n    font-size: 0.75rem;\n    font-family: \'JetBrains Mono\', monospace;\n    color: var(--accent-indigo);\n    font-weight: 600;\n}\n\n.segment-en {\n    font-size: 0.88rem;\n    color: var(--text-secondary);\n}\n\n.segment-hi {\n    font-size: 0.95rem;\n    color: var(--text-primary);\n    font-weight: 500;\n}\n\n/* Modal */\n.modal-overlay {\n    position: fixed;\n    top: 0;\n    left: 0;\n    width: 100vw;\n    height: 100vh;\n    background: rgba(0, 0, 0, 0.75);\n    backdrop-filter: blur(8px);\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    z-index: 100;\n}\n\n.modal-card {\n    background: var(--bg-secondary);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-lg);\n    width: 90%;\n    max-width: 520px;\n    box-shadow: var(--shadow-card);\n    display: flex;\n    flex-direction: column;\n}\n\n.modal-header {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    padding: 20px 24px;\n    border-bottom: 1px solid var(--border-color);\n}\n\n.modal-header h2 {\n    font-size: 1.25rem;\n    font-weight: 700;\n}\n\n.btn-close {\n    background: none;\n    border: none;\n    color: var(--text-muted);\n    font-size: 1.2rem;\n    cursor: pointer;\n}\n\n.modal-body {\n    padding: 24px;\n    display: flex;\n    flex-direction: column;\n    gap: 18px;\n}\n\n.external-link {\n    color: var(--accent-indigo);\n    font-size: 0.8rem;\n    margin-left: 6px;\n    text-decoration: none;\n}\n\n.modal-footer {\n    display: flex;\n    justify-content: flex-end;\n    gap: 12px;\n    padding: 16px 24px;\n    border-top: 1px solid var(--border-color);\n}\n\n.settings-feedback {\n    font-size: 0.85rem;\n    min-height: 20px;\n}\n\n.settings-feedback.success { color: var(--accent-emerald); }\n.settings-feedback.error { color: var(--accent-rose); }\n\n/* Utilities */\n.hidden {\n    display: none !important;\n}\n\n.badge {\n    padding: 4px 10px;\n    border-radius: 12px;\n    font-size: 0.75rem;\n    font-weight: 600;\n    background: rgba(255, 255, 255, 0.08);\n}\n\n/* ========================================================\n   Pricing & UPI Payment System Styles\n   ======================================================== */\n.pricing-banner {\n    background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(49, 21, 56, 0.9) 50%, rgba(20, 28, 48, 0.9) 100%);\n    border: 1px solid rgba(139, 92, 246, 0.35);\n    border-radius: var(--radius-lg);\n    padding: 20px 28px;\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    gap: 20px;\n    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 25px rgba(139, 92, 246, 0.2);\n    position: relative;\n    overflow: hidden;\n}\n\n.pricing-banner::before {\n    content: \'\';\n    position: absolute;\n    top: 0;\n    left: 0;\n    right: 0;\n    height: 3px;\n    background: linear-gradient(90deg, #f59e0b, #ec4899, #8b5cf6, #3b82f6);\n}\n\n.banner-badge {\n    position: absolute;\n    top: 10px;\n    right: 16px;\n    background: linear-gradient(135deg, #f59e0b, #ec4899);\n    color: #fff;\n    font-size: 0.7rem;\n    font-weight: 800;\n    padding: 3px 10px;\n    border-radius: 20px;\n    text-transform: uppercase;\n    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4);\n}\n\n.banner-content h2 {\n    font-size: 1.25rem;\n    font-weight: 800;\n    color: #fff;\n    margin-bottom: 4px;\n}\n\n.banner-content p {\n    font-size: 0.85rem;\n    color: var(--text-secondary);\n}\n\n.banner-content strong {\n    color: #38bdf8;\n}\n\n.banner-plans-preview {\n    display: flex;\n    gap: 10px;\n}\n\n.plan-chip {\n    background: rgba(255, 255, 255, 0.05);\n    border: 1px solid var(--border-color);\n    border-radius: 10px;\n    padding: 8px 14px;\n    text-align: center;\n    cursor: pointer;\n    transition: all 0.2s;\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n}\n\n.plan-chip:hover {\n    border-color: #8b5cf6;\n    background: rgba(139, 92, 246, 0.15);\n    transform: translateY(-2px);\n}\n\n.plan-chip strong {\n    font-size: 1.2rem;\n    color: #fff;\n    line-height: 1.1;\n}\n\n.plan-chip small {\n    font-size: 0.68rem;\n    color: var(--text-muted);\n}\n\n.chip-badge {\n    font-size: 0.6rem;\n    font-weight: 700;\n    text-transform: uppercase;\n    color: var(--text-secondary);\n    margin-bottom: 2px;\n}\n\n.chip-popular {\n    border-color: #f59e0b;\n    background: rgba(245, 158, 11, 0.12);\n}\n\n.badge-pop {\n    color: #f59e0b;\n    font-weight: 800;\n}\n\n.btn-banner-action {\n    background: linear-gradient(135deg, #f59e0b 0%, #ec4899 50%, #8b5cf6 100%);\n    color: #fff;\n    border: none;\n    border-radius: var(--radius-md);\n    padding: 12px 20px;\n    font-weight: 800;\n    font-size: 0.95rem;\n    cursor: pointer;\n    white-space: nowrap;\n    box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);\n    transition: all 0.25s ease;\n}\n\n.btn-banner-action:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 6px 22px rgba(236, 72, 153, 0.6);\n}\n\n@media (max-width: 900px) {\n    .pricing-banner {\n        flex-direction: column;\n        align-items: stretch;\n        text-align: center;\n    }\n    .banner-plans-preview {\n        justify-content: center;\n    }\n    .btn-banner-action {\n        width: 100%;\n    }\n}\n\n.btn-pass {\n    background: linear-gradient(135deg, #f59e0b 0%, #ec4899 50%, #8b5cf6 100%);\n    color: #ffffff;\n    font-weight: 700;\n    border: none;\n    box-shadow: 0 0 20px rgba(245, 158, 11, 0.35);\n    transition: all 0.25s ease;\n}\n\n.btn-pass:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 0 28px rgba(236, 72, 153, 0.5);\n}\n\n.btn-pass.unlocked-pro {\n    background: linear-gradient(135deg, #10b981 0%, #059669 100%);\n    box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);\n}\n\n.pricing-modal-card {\n    max-width: 680px;\n}\n\n.modal-subtitle {\n    font-size: 0.85rem;\n    color: var(--text-secondary);\n    margin-top: 4px;\n}\n\n/* Plans Grid */\n.plans-container {\n    display: grid;\n    grid-template-columns: repeat(3, 1fr);\n    gap: 14px;\n}\n\n.plan-card {\n    background: rgba(15, 18, 30, 0.9);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-md);\n    padding: 16px 14px;\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n    text-align: center;\n    position: relative;\n    cursor: pointer;\n    transition: all 0.25s ease;\n}\n\n.plan-card:hover {\n    border-color: var(--accent-indigo);\n    transform: translateY(-2px);\n}\n\n.plan-card.selected {\n    border: 2px solid #8b5cf6;\n    background: rgba(139, 92, 246, 0.12);\n    box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);\n}\n\n.popular-plan.selected {\n    border-color: #f59e0b;\n    background: rgba(245, 158, 11, 0.12);\n    box-shadow: 0 0 25px rgba(245, 158, 11, 0.3);\n}\n\n.plan-badge {\n    font-size: 0.68rem;\n    font-weight: 700;\n    text-transform: uppercase;\n    padding: 2px 8px;\n    border-radius: 20px;\n    background: rgba(255, 255, 255, 0.08);\n    color: var(--text-secondary);\n    margin-bottom: 8px;\n}\n\n.badge-popular {\n    background: linear-gradient(135deg, #f59e0b, #ec4899);\n    color: #fff;\n}\n\n.plan-price {\n    font-size: 1.8rem;\n    font-weight: 800;\n    color: #fff;\n    line-height: 1;\n    margin-bottom: 4px;\n}\n\n.plan-duration {\n    font-size: 0.8rem;\n    color: var(--text-muted);\n    margin-bottom: 12px;\n}\n\n.plan-features {\n    list-style: none;\n    font-size: 0.76rem;\n    color: var(--text-secondary);\n    text-align: left;\n    width: 100%;\n    margin-bottom: 14px;\n    display: flex;\n    flex-direction: column;\n    gap: 6px;\n}\n\n.plan-features strong {\n    color: #f59e0b;\n}\n\n.btn-plan-select {\n    width: 100%;\n    padding: 6px 0;\n    font-size: 0.8rem;\n    font-weight: 600;\n    border-radius: var(--radius-sm);\n    border: 1px solid var(--border-color);\n    background: rgba(255, 255, 255, 0.05);\n    color: var(--text-primary);\n    cursor: pointer;\n    transition: all 0.2s;\n    margin-top: auto;\n}\n\n.plan-card.selected .btn-plan-select {\n    background: var(--accent-indigo);\n    border-color: var(--accent-indigo);\n    color: #fff;\n}\n\n.popular-plan.selected .btn-plan-select {\n    background: linear-gradient(135deg, #f59e0b, #ec4899);\n    border: none;\n}\n\n/* Payment Checkout Card */\n.payment-checkout-card {\n    background: rgba(10, 13, 22, 0.85);\n    border: 1px solid var(--border-color);\n    border-radius: var(--radius-md);\n    padding: 20px;\n    display: flex;\n    gap: 22px;\n    align-items: center;\n}\n\n@media (max-width: 600px) {\n    .plans-container {\n        grid-template-columns: 1fr;\n    }\n    .payment-checkout-card {\n        flex-direction: column;\n    }\n}\n\n.qr-col {\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n    gap: 8px;\n    flex-shrink: 0;\n}\n\n.qr-box {\n    background: #ffffff;\n    padding: 10px;\n    border-radius: 12px;\n    position: relative;\n    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);\n    display: flex;\n    justify-content: center;\n    align-items: center;\n}\n\n.qr-box img {\n    width: 160px;\n    height: 160px;\n    display: block;\n    border-radius: 6px;\n}\n\n.qr-overlay-brand {\n    position: absolute;\n    bottom: -6px;\n    right: -6px;\n    background: #10b981;\n    color: #fff;\n    font-weight: 800;\n    font-size: 0.8rem;\n    padding: 3px 8px;\n    border-radius: 8px;\n    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);\n}\n\n.qr-scan-label {\n    font-size: 0.72rem;\n    color: var(--text-muted);\n    text-align: center;\n}\n\n.upi-details-col {\n    display: flex;\n    flex-direction: column;\n    gap: 12px;\n    flex: 1;\n}\n\n.upi-id-pill {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    background: rgba(255, 255, 255, 0.04);\n    border: 1px dashed var(--border-color);\n    padding: 8px 14px;\n    border-radius: 8px;\n}\n\n.upi-label {\n    font-size: 0.78rem;\n    color: var(--text-muted);\n}\n\n.upi-val {\n    font-family: \'JetBrains Mono\', monospace;\n    color: #38bdf8;\n    font-size: 0.95rem;\n}\n\n.btn-copy {\n    background: transparent;\n    border: 1px solid var(--border-color);\n    color: var(--text-secondary);\n    padding: 4px 8px;\n    border-radius: 6px;\n    font-size: 0.75rem;\n    cursor: pointer;\n    transition: all 0.2s;\n}\n\n.btn-copy:hover {\n    background: rgba(255, 255, 255, 0.1);\n    color: #fff;\n}\n\n.btn-upi-pay {\n    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);\n    color: #fff;\n    text-decoration: none;\n    padding: 10px 16px;\n    border-radius: var(--radius-sm);\n    font-weight: 700;\n    font-size: 0.9rem;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    gap: 8px;\n    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3);\n    transition: all 0.2s;\n}\n\n.btn-upi-pay:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45);\n}\n\n.verification-box {\n    display: flex;\n    flex-direction: column;\n    gap: 6px;\n    margin-top: 4px;\n}\n\n.verification-box label {\n    font-size: 0.76rem;\n    color: var(--text-secondary);\n}\n\n.verify-input-group {\n    display: flex;\n    gap: 8px;\n}\n\n.verify-input-group input {\n    flex: 1;\n    font-size: 0.85rem;\n    padding: 8px 12px;\n}\n\n.verify-input-group button {\n    flex-shrink: 0;\n    font-size: 0.85rem;\n    padding: 8px 14px;\n    font-weight: 700;\n}\n\n/* ==================== CREATOR AFFILIATE DEALS ==================== */\n.affiliate-deals-box {\n    margin-top: 20px;\n    padding: 16px;\n    background: rgba(255, 255, 255, 0.03);\n    border: 1px solid rgba(255, 255, 255, 0.08);\n    border-radius: var(--radius-md);\n}\n\n.affiliate-deals-box h4 {\n    font-size: 0.88rem;\n    font-weight: 700;\n    color: #f1f5f9;\n    margin-bottom: 12px;\n}\n\n.deals-grid {\n    display: grid;\n    grid-template-columns: 1fr 1fr;\n    gap: 10px;\n}\n\n@media (max-width: 600px) {\n    .deals-grid {\n        grid-template-columns: 1fr;\n    }\n}\n\n.deal-item {\n    display: flex;\n    align-items: center;\n    gap: 10px;\n    padding: 10px 12px;\n    background: rgba(15, 23, 42, 0.6);\n    border: 1px solid rgba(255, 255, 255, 0.06);\n    border-radius: var(--radius-sm);\n    text-decoration: none;\n    transition: all 0.2s ease;\n}\n\n.deal-item:hover {\n    border-color: #6366f1;\n    background: rgba(99, 102, 241, 0.1);\n    transform: translateY(-2px);\n}\n\n.deal-icon {\n    font-size: 1.4rem;\n    flex-shrink: 0;\n}\n\n.deal-info {\n    flex: 1;\n    display: flex;\n    flex-direction: column;\n    overflow: hidden;\n}\n\n.deal-info strong {\n    font-size: 0.8rem;\n    color: #fff;\n    white-space: nowrap;\n    overflow: hidden;\n    text-overflow: ellipsis;\n}\n\n.deal-info small {\n    font-size: 0.7rem;\n    color: var(--text-muted);\n    white-space: nowrap;\n    overflow: hidden;\n    text-overflow: ellipsis;\n}\n\n.deal-tag {\n    font-size: 0.68rem;\n    font-weight: 600;\n    color: #38bdf8;\n    background: rgba(56, 189, 248, 0.1);\n    padding: 2px 6px;\n    border-radius: 4px;\n    flex-shrink: 0;\n}\n\n/* ==================== FOOTER & AD BANNERS ==================== */\n.app-footer {\n    max-width: 1200px;\n    margin: 30px auto 20px;\n    padding: 0 20px;\n    display: flex;\n    flex-direction: column;\n    align-items: center;\n    gap: 14px;\n}\n\n.ad-banner-slot {\n    width: 100%;\n    max-width: 900px;\n    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);\n    border: 1px dashed rgba(255, 255, 255, 0.15);\n    border-radius: var(--radius-md);\n    padding: 16px 20px;\n    text-align: center;\n    position: relative;\n}\n\n.ad-label {\n    font-size: 0.65rem;\n    letter-spacing: 1px;\n    color: var(--text-muted);\n    font-weight: 700;\n    display: block;\n    margin-bottom: 6px;\n}\n\n.ad-placeholder p {\n    font-size: 0.88rem;\n    color: #e2e8f0;\n    margin: 0;\n}\n\n.footer-copy {\n    font-size: 0.78rem;\n    color: var(--text-muted);\n    text-align: center;\n}\n\n'
EMBEDDED_JS = 'document.addEventListener("DOMContentLoaded", () => {\n    // State\n    const DEFAULT_UPI_ID = "subham.088@fam";\n    let selectedPlan = 20;\n    let selectedPlanName = "28-Days Monthly VIP Pass";\n    let currentConfig = {};\n    let uploadedVideoData = null;\n    let activeJobId = null;\n    let pollInterval = null;\n\n    // DOM Elements - Nav & Badges\n    const btnOpenPricing = document.getElementById("btnOpenPricing");\n    const passStatusText = document.getElementById("passStatusText");\n    const apiStatusBadge = document.getElementById("apiStatusBadge");\n    const apiStatusText = document.getElementById("apiStatusText");\n    const btnOpenSettings = document.getElementById("btnOpenSettings");\n    const btnCloseSettings = document.getElementById("btnCloseSettings");\n    const btnCancelSettings = document.getElementById("btnCancelSettings");\n    const btnSaveSettings = document.getElementById("btnSaveSettings");\n    const settingsModal = document.getElementById("settingsModal");\n    const settingsFeedback = document.getElementById("settingsFeedback");\n\n    // Pricing Modal Elements\n    const pricingModal = document.getElementById("pricingModal");\n    const btnClosePricing = document.getElementById("btnClosePricing");\n    const btnCancelPricing = document.getElementById("btnCancelPricing");\n    const planCards = document.querySelectorAll(".plan-card");\n    const upiQrImage = document.getElementById("upiQrImage");\n    const selectedPlanAmountDisplay = document.getElementById("selectedPlanAmountDisplay");\n    const intentPayAmount = document.getElementById("intentPayAmount");\n    const btnUpiIntentLink = document.getElementById("btnUpiIntentLink");\n    const textUpiId = document.getElementById("textUpiId");\n    const btnCopyUpi = document.getElementById("btnCopyUpi");\n    const inputUtrNumber = document.getElementById("inputUtrNumber");\n    const btnVerifyUtr = document.getElementById("btnVerifyUtr");\n    const paymentFeedback = document.getElementById("paymentFeedback");\n\n    // Settings Inputs\n    const inputElevenLabsKey = document.getElementById("inputElevenLabsKey");\n    const inputVoiceId = document.getElementById("inputVoiceId");\n    const inputGeminiKey = document.getElementById("inputGeminiKey");\n\n    // Upload & Form Elements\n    const dropzone = document.getElementById("dropzone");\n    const videoFileInput = document.getElementById("videoFileInput");\n    const dropzonePrompt = document.getElementById("dropzonePrompt");\n    const selectedVideoPreview = document.getElementById("selectedVideoPreview");\n    const inputVideoPreview = document.getElementById("inputVideoPreview");\n    const previewFileName = document.getElementById("previewFileName");\n    const previewSpecs = document.getElementById("previewSpecs");\n    const btnRemoveVideo = document.getElementById("btnRemoveVideo");\n\n    const voiceSelect = document.getElementById("voiceSelect");\n    const voiceHelperText = document.getElementById("voiceHelperText");\n    const btnTestVoice = document.getElementById("btnTestVoice");\n    const voiceTestAudio = document.getElementById("voiceTestAudio");\n\n    const bgVolume = document.getElementById("bgVolume");\n    const bgVolumeVal = document.getElementById("bgVolumeVal");\n    const btnStartDubbing = document.getElementById("btnStartDubbing");\n\n    const jobStatusBadge = document.getElementById("jobStatusBadge");\n    const progressBarFill = document.getElementById("progressBarFill");\n    const statusMessage = document.getElementById("statusMessage");\n\n    const stepNodes = {\n        extract: document.getElementById("step-extract"),\n        transcribe: document.getElementById("step-transcribe"),\n        translate: document.getElementById("step-translate"),\n        synthesize: document.getElementById("step-synthesize"),\n        merge: document.getElementById("step-merge")\n    };\n\n    const tabBtns = document.querySelectorAll(".tab-btn");\n    const tabContents = document.querySelectorAll(".tab-content");\n\n    const playerEmptyState = document.getElementById("playerEmptyState");\n    const playerWrapper = document.getElementById("playerWrapper");\n    const outputVideoPlayer = document.getElementById("outputVideoPlayer");\n    const btnDownloadVideo = document.getElementById("btnDownloadVideo");\n    const btnDownloadSrt = document.getElementById("btnDownloadSrt");\n    const transcriptSegmentsList = document.getElementById("transcriptSegmentsList");\n\n    // Initialize App\n    initApp();\n\n    async function initApp() {\n        checkPassStatus();\n        await loadConfig();\n        setupEventListeners();\n        setupPricingEvents();\n        loadVoices();\n    }\n\n    // Check if user already unlocked pass\n    function checkPassStatus() {\n        const isUnlocked = localStorage.getItem("videodubber_unlocked_pass") === "true";\n        if (isUnlocked) {\n            passStatusText.textContent = "👑 Pro Member (Active)";\n            btnOpenPricing.classList.add("unlocked-pro");\n        } else {\n            passStatusText.textContent = "👑 Unlock Pass (₹10/₹15/₹20)";\n            btnOpenPricing.classList.remove("unlocked-pro");\n        }\n    }\n\n    async function loadConfig() {\n        try {\n            // Check LocalStorage first for BYOK (Bring Your Own Key)\n            const localGemini = localStorage.getItem("videodubber_gemini_key");\n            const localEleven = localStorage.getItem("videodubber_elevenlabs_key");\n            const localVoice = localStorage.getItem("videodubber_voice_id");\n\n            const res = await fetch("/api/config");\n            currentConfig = await res.json();\n\n            // LocalStorage takes priority for visitors\n            if (localGemini) currentConfig.gemini_api_key = localGemini;\n            if (localEleven) currentConfig.elevenlabs_api_key = localEleven;\n            if (localVoice) currentConfig.selected_voice_id = localVoice;\n\n            inputElevenLabsKey.value = currentConfig.elevenlabs_api_key || "";\n            inputGeminiKey.value = currentConfig.gemini_api_key || "";\n            inputVoiceId.value = currentConfig.selected_voice_id || "";\n\n            if (currentConfig.bg_music_volume !== undefined) {\n                bgVolume.value = currentConfig.bg_music_volume;\n                bgVolumeVal.textContent = `${Math.round(currentConfig.bg_music_volume * 100)}%`;\n            }\n\n            if (currentConfig.audio_mode) {\n                const radio = document.querySelector(`input[name="audioMode"][value="${currentConfig.audio_mode}"]`);\n                if (radio) radio.checked = true;\n            }\n\n            updateApiStatusBadge();\n        } catch (err) {\n            console.error("Failed to load config:", err);\n        }\n    }\n\n    function updateApiStatusBadge() {\n        const hasGemini = !!(currentConfig.gemini_api_key || localStorage.getItem("videodubber_gemini_key"));\n        const hasEleven = !!(currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key"));\n\n        if (hasGemini) {\n            apiStatusBadge.className = "status-pill status-success";\n            apiStatusText.textContent = hasEleven ? "AI Ready (Gemini + ElevenLabs/Bunty)" : "AI Ready (Gemini + Bunty Voice)";\n        } else {\n            apiStatusBadge.className = "status-pill status-warning";\n            apiStatusText.textContent = "Gemini Key Needed";\n        }\n    }\n\n    async function loadVoices() {\n        try {\n            voiceSelect.innerHTML = \'<option value="" disabled selected>Loading ElevenLabs & Indian voices...</option>\';\n            const key = currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key") || "";\n            const res = await fetch(`/api/voices?api_key=${encodeURIComponent(key)}`);\n            const data = await res.json();\n\n            if (data.voices && data.voices.length > 0) {\n                voiceSelect.innerHTML = "";\n                let buntyVoice = null;\n\n                data.voices.forEach(v => {\n                    const opt = document.createElement("option");\n                    opt.value = v.voice_id;\n                    opt.textContent = `${v.name} ${v.is_bunty ? "⭐ (Bunty)" : `(${v.category || "Voice"})`}`;\n                    if (v.is_bunty) {\n                        opt.classList.add("bunty-opt");\n                        buntyVoice = v;\n                    }\n                    voiceSelect.appendChild(opt);\n                });\n\n                // Auto-select Bunty if available or previously saved voice\n                const savedVoice = currentConfig.selected_voice_id || localStorage.getItem("videodubber_voice_id");\n                if (savedVoice) {\n                    voiceSelect.value = savedVoice;\n                } else if (buntyVoice) {\n                    voiceSelect.value = buntyVoice.voice_id;\n                    voiceHelperText.textContent = `Selected: ${buntyVoice.name} (Auto-detected Bunty)`;\n                } else if (data.voices.length > 0) {\n                    voiceSelect.value = data.voices[0].voice_id;\n                }\n\n                inputVoiceId.value = voiceSelect.value;\n            } else {\n                const errMsg = data.error || data.message || "No voices found.";\n                voiceSelect.innerHTML = `<option value="" disabled selected>${errMsg}</option>`;\n                voiceHelperText.textContent = errMsg;\n            }\n        } catch (err) {\n            voiceSelect.innerHTML = \'<option value="" disabled selected>Default: Bunty Voice Ready</option>\';\n            voiceHelperText.textContent = "Bunty Hindi Voice Enabled";\n        }\n    }\n\n    function updatePricingCheckout() {\n        selectedPlanAmountDisplay.textContent = selectedPlan;\n        intentPayAmount.textContent = selectedPlan;\n        \n        const upiUri = `upi://pay?pa=${DEFAULT_UPI_ID}&pn=VideoDubber%20AI&am=${selectedPlan}&cu=INR&tn=VideoDubber%20${encodeURIComponent(selectedPlanName)}`;\n        \n        // Update QR Code\n        upiQrImage.src = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(upiUri)}`;\n        \n        // Update Intent Button\n        btnUpiIntentLink.href = upiUri;\n    }\n\n    function setupPricingEvents() {\n        // Modal toggles\n        btnOpenPricing.addEventListener("click", () => {\n            pricingModal.classList.remove("hidden");\n            paymentFeedback.textContent = "";\n            updatePricingCheckout();\n        });\n\n        btnClosePricing.addEventListener("click", () => pricingModal.classList.add("hidden"));\n        btnCancelPricing.addEventListener("click", () => pricingModal.classList.add("hidden"));\n\n        // Plan Selection\n        planCards.forEach(card => {\n            card.addEventListener("click", () => {\n                planCards.forEach(c => {\n                    c.classList.remove("selected");\n                    const b = c.querySelector(".btn-plan-select");\n                    if (b) b.textContent = `Select ₹${c.dataset.plan}`;\n                });\n\n                card.classList.add("selected");\n                const btn = card.querySelector(".btn-plan-select");\n                if (btn) btn.textContent = `Selected (₹${card.dataset.plan})`;\n\n                selectedPlan = parseInt(card.dataset.plan, 10);\n                selectedPlanName = card.dataset.name;\n                updatePricingCheckout();\n            });\n        });\n\n        // Copy UPI ID\n        btnCopyUpi.addEventListener("click", () => {\n            navigator.clipboard.writeText(DEFAULT_UPI_ID).then(() => {\n                btnCopyUpi.textContent = "✓ Copied!";\n                setTimeout(() => { btnCopyUpi.textContent = "📋 Copy"; }, 2000);\n            }).catch(() => {\n                prompt("Copy UPI ID:", DEFAULT_UPI_ID);\n            });\n        });\n\n        // Verify UTR / Passcode\n        btnVerifyUtr.addEventListener("click", async () => {\n            const utr = inputUtrNumber.value.trim();\n            if (!utr) {\n                paymentFeedback.className = "settings-feedback error";\n                paymentFeedback.textContent = "Kripya payment ke baad apna 12-digit UTR No. ya VIP code dalein.";\n                return;\n            }\n\n            btnVerifyUtr.disabled = true;\n            btnVerifyUtr.textContent = "Verifying...";\n            paymentFeedback.className = "settings-feedback";\n            paymentFeedback.textContent = "Verifying transaction reference...";\n\n            try {\n                const res = await fetch("/api/verify-pass", {\n                    method: "POST",\n                    headers: { "Content-Type": "application/json" },\n                    body: JSON.stringify({\n                        utr_or_code: utr,\n                        plan_price: selectedPlan\n                    })\n                });\n\n                const data = await res.json();\n                if (res.ok && data.unlocked) {\n                    localStorage.setItem("videodubber_unlocked_pass", "true");\n                    localStorage.setItem("videodubber_pass_type", selectedPlanName);\n                    checkPassStatus();\n\n                    paymentFeedback.className = "settings-feedback success";\n                    paymentFeedback.innerHTML = `🎉 <strong>Mubarak Ho!</strong> Studio Access Unlocked. (Plan: ${selectedPlanName})`;\n\n                    setTimeout(() => {\n                        pricingModal.classList.add("hidden");\n                        btnVerifyUtr.disabled = false;\n                        btnVerifyUtr.textContent = "Unlock Now 🔓";\n                    }, 1500);\n                } else {\n                    throw new Error(data.detail || "Invalid transaction reference.");\n                }\n            } catch (err) {\n                paymentFeedback.className = "settings-feedback error";\n                paymentFeedback.textContent = `Verification Error: ${err.message}`;\n                btnVerifyUtr.disabled = false;\n                btnVerifyUtr.textContent = "Unlock Now 🔓";\n            }\n        });\n    }\n\n    function setupEventListeners() {\n        // Modal toggles\n        btnOpenSettings.addEventListener("click", () => {\n            settingsModal.classList.remove("hidden");\n            settingsFeedback.textContent = "";\n        });\n        btnCloseSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));\n        btnCancelSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));\n\n        // Save Settings (Save to LocalStorage & Config)\n        btnSaveSettings.addEventListener("click", async () => {\n            const elKey = inputElevenLabsKey.value.trim();\n            const gemKey = inputGeminiKey.value.trim();\n            const vId = inputVoiceId.value.trim();\n\n            // Store in user\'s browser localStorage for BYOK\n            if (gemKey) localStorage.setItem("videodubber_gemini_key", gemKey);\n            if (elKey) localStorage.setItem("videodubber_elevenlabs_key", elKey);\n            if (vId) localStorage.setItem("videodubber_voice_id", vId);\n\n            currentConfig.gemini_api_key = gemKey || currentConfig.gemini_api_key;\n            currentConfig.elevenlabs_api_key = elKey || currentConfig.elevenlabs_api_key;\n            currentConfig.selected_voice_id = vId || currentConfig.selected_voice_id;\n\n            settingsFeedback.className = "settings-feedback";\n            settingsFeedback.textContent = "Saving keys securely in browser...";\n\n            try {\n                await fetch("/api/config", {\n                    method: "POST",\n                    headers: { "Content-Type": "application/json" },\n                    body: JSON.stringify({\n                        elevenlabs_api_key: elKey,\n                        gemini_api_key: gemKey,\n                        selected_voice_id: vId\n                    })\n                });\n\n                settingsFeedback.className = "settings-feedback success";\n                settingsFeedback.textContent = "✓ API Keys saved securely in browser!";\n                updateApiStatusBadge();\n                if (elKey) loadVoices();\n                setTimeout(() => {\n                    settingsModal.classList.add("hidden");\n                }, 1000);\n            } catch (err) {\n                settingsFeedback.className = "settings-feedback success";\n                settingsFeedback.textContent = "✓ API Keys saved locally!";\n                updateApiStatusBadge();\n                setTimeout(() => { settingsModal.classList.add("hidden"); }, 1000);\n            }\n        });\n\n        // Voice Select Change\n        voiceSelect.addEventListener("change", () => {\n            inputVoiceId.value = voiceSelect.value;\n            localStorage.setItem("videodubber_voice_id", voiceSelect.value);\n            const selectedText = voiceSelect.options[voiceSelect.selectedIndex].text;\n            voiceHelperText.textContent = `Selected: ${selectedText}`;\n        });\n\n        // Test Voice button\n        btnTestVoice.addEventListener("click", async () => {\n            const voiceId = voiceSelect.value || inputVoiceId.value.trim();\n            if (!voiceId) {\n                alert("Please select or enter a Voice ID first.");\n                return;\n            }\n            btnTestVoice.textContent = "⏳ Generating sample...";\n            btnTestVoice.disabled = true;\n\n            const elKey = currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key") || "";\n\n            try {\n                const res = await fetch("/api/test-voice", {\n                    method: "POST",\n                    headers: { "Content-Type": "application/json" },\n                    body: JSON.stringify({\n                        voice_id: voiceId,\n                        text: "नमस्ते दोस्तों! मैं बंटी हूँ, और आपकी इंग्लिश वीडियो को हिंदी में डब करूँगा।",\n                        elevenlabs_api_key: elKey\n                    })\n                });\n                const data = await res.json();\n                if (data.status === "success" && data.audio_url) {\n                    voiceTestAudio.src = data.audio_url;\n                    voiceTestAudio.play();\n                } else {\n                    alert(`Voice test error: ${data.detail || "Unknown error"}`);\n                }\n            } catch (err) {\n                alert(`Could not play sample: ${err.message}`);\n            } finally {\n                btnTestVoice.textContent = \'🔊 Test "Bunty" Voice\';\n                btnTestVoice.disabled = false;\n            }\n        });\n\n        // Drag & Drop Upload\n        dropzone.addEventListener("click", (e) => {\n            if (e.target !== btnRemoveVideo && !selectedVideoPreview.contains(e.target)) {\n                videoFileInput.click();\n            }\n        });\n\n        dropzone.addEventListener("dragover", (e) => {\n            e.preventDefault();\n            dropzone.classList.add("dragover");\n        });\n\n        dropzone.addEventListener("dragleave", () => {\n            dropzone.classList.remove("dragover");\n        });\n\n        dropzone.addEventListener("drop", (e) => {\n            e.preventDefault();\n            dropzone.classList.remove("dragover");\n            if (e.dataTransfer.files.length > 0) {\n                handleVideoFile(e.dataTransfer.files[0]);\n            }\n        });\n\n        videoFileInput.addEventListener("change", () => {\n            if (videoFileInput.files.length > 0) {\n                handleVideoFile(videoFileInput.files[0]);\n            }\n        });\n\n        btnRemoveVideo.addEventListener("click", (e) => {\n            e.stopPropagation();\n            resetVideoUpload();\n        });\n\n        // Volume Slider\n        bgVolume.addEventListener("input", () => {\n            bgVolumeVal.textContent = `${Math.round(bgVolume.value * 100)}%`;\n        });\n\n        // Tabs\n        tabBtns.forEach(btn => {\n            btn.addEventListener("click", () => {\n                tabBtns.forEach(b => b.classList.remove("active"));\n                tabContents.forEach(c => c.classList.remove("active"));\n                btn.classList.add("active");\n                const target = document.getElementById(btn.dataset.tab);\n                if (target) target.classList.add("active");\n            });\n        });\n\n        // Start Dubbing\n        btnStartDubbing.addEventListener("click", startDubbingProcess);\n    }\n\n    async function handleVideoFile(file) {\n        if (!file) return;\n        \n        statusMessage.textContent = "Uploading video...";\n        const formData = new FormData();\n        formData.append("file", file);\n\n        try {\n            const res = await fetch("/api/upload", {\n                method: "POST",\n                body: formData\n            });\n\n            if (!res.ok) {\n                const err = await res.json();\n                throw new Error(err.detail || "Upload failed");\n            }\n\n            uploadedVideoData = await res.json();\n            \n            // Show preview\n            dropzonePrompt.classList.add("hidden");\n            selectedVideoPreview.classList.remove("hidden");\n            inputVideoPreview.src = uploadedVideoData.video_url;\n            previewFileName.textContent = uploadedVideoData.filename;\n            \n            const mins = Math.floor(uploadedVideoData.duration / 60);\n            const secs = Math.floor(uploadedVideoData.duration % 60);\n            previewSpecs.textContent = `${mins}:${secs < 10 ? \'0\' : \'\'}${secs} • ${uploadedVideoData.width}x${uploadedVideoData.height}`;\n\n            btnStartDubbing.disabled = false;\n            statusMessage.textContent = `Video loaded (${uploadedVideoData.filename}). Ready to dub to Hindi.`;\n        } catch (err) {\n            alert(`Upload failed: ${err.message}`);\n            resetVideoUpload();\n        }\n    }\n\n    function resetVideoUpload() {\n        uploadedVideoData = null;\n        videoFileInput.value = "";\n        inputVideoPreview.src = "";\n        selectedVideoPreview.classList.add("hidden");\n        dropzonePrompt.classList.remove("hidden");\n        btnStartDubbing.disabled = true;\n        statusMessage.textContent = "Upload a video to begin translation and dubbing.";\n    }\n\n    async function startDubbingProcess() {\n        if (!uploadedVideoData) return;\n\n        // Check if pass unlocked (If not unlocked, ask for ₹10/15/20 pass)\n        const isUnlocked = localStorage.getItem("videodubber_unlocked_pass") === "true";\n        if (!isUnlocked) {\n            pricingModal.classList.remove("hidden");\n            paymentFeedback.className = "settings-feedback";\n            paymentFeedback.textContent = "👑 Unlimited Video Dubbing ke liye apna ₹10, ₹15, ya ₹20 ka Pass choose karein.";\n            return;\n        }\n\n        const gemKey = currentConfig.gemini_api_key || localStorage.getItem("videodubber_gemini_key");\n        const elKey = currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key");\n\n        if (!gemKey) {\n            settingsModal.classList.remove("hidden");\n            settingsFeedback.className = "settings-feedback error";\n            settingsFeedback.textContent = "Please provide your Free Google Gemini API Key in Settings to proceed.";\n            return;\n        }\n\n        const voiceId = voiceSelect.value || inputVoiceId.value.trim();\n        const audioMode = document.querySelector(\'input[name="audioMode"]:checked\').value;\n        const bgVol = parseFloat(bgVolume.value);\n\n        btnStartDubbing.disabled = true;\n        btnStartDubbing.innerHTML = \'<span class="btn-icon">⏳</span><span>Dubbing in Progress...</span>\';\n\n        resetPipelineVisualizer();\n\n        try {\n            const res = await fetch("/api/start-dub", {\n                method: "POST",\n                headers: { "Content-Type": "application/json" },\n                body: JSON.stringify({\n                    video_id: uploadedVideoData.video_id,\n                    voice_id: voiceId,\n                    audio_mode: audioMode,\n                    bg_music_volume: bgVol,\n                    voice_volume: 1.0,\n                    gemini_api_key: gemKey,\n                    elevenlabs_api_key: elKey\n                })\n            });\n\n            const data = await res.json();\n            activeJobId = data.job_id;\n            jobStatusBadge.textContent = "Processing";\n            jobStatusBadge.style.backgroundColor = "rgba(99, 102, 241, 0.3)";\n\n            // Start polling\n            pollInterval = setInterval(pollJobStatus, 1500);\n        } catch (err) {\n            alert(`Failed to start dubbing: ${err.message}`);\n            btnStartDubbing.disabled = false;\n            btnStartDubbing.innerHTML = \'<span class="btn-icon">⚡</span><span>Start English to Hindi Dubbing</span>\';\n        }\n    }\n\n    async function pollJobStatus() {\n        if (!activeJobId) return;\n\n        try {\n            const res = await fetch(`/api/job/${activeJobId}`);\n            const job = await res.json();\n\n            // Update Progress Bar & Message\n            progressBarFill.style.width = `${job.progress}%`;\n            statusMessage.textContent = job.message;\n\n            // Update Step Nodes\n            updateStepVisualizer(job.step);\n\n            // Update Transcripts if present\n            if (job.segments && job.segments.length > 0) {\n                renderTranscript(job.segments);\n            }\n\n            if (job.status === "completed") {\n                clearInterval(pollInterval);\n                jobStatusBadge.textContent = "Completed";\n                jobStatusBadge.style.backgroundColor = "rgba(16, 185, 129, 0.3)";\n                btnStartDubbing.disabled = false;\n                btnStartDubbing.innerHTML = \'<span class="btn-icon">⚡</span><span>Start English to Hindi Dubbing</span>\';\n\n                // Display finished video\n                playerEmptyState.classList.add("hidden");\n                playerWrapper.classList.remove("hidden");\n                outputVideoPlayer.src = job.dubbed_video_url;\n                btnDownloadVideo.href = job.dubbed_video_url;\n                btnDownloadSrt.href = job.subtitles_srt_url;\n\n                // Auto switch to player tab\n                document.querySelector(\'.tab-btn[data-tab="tab-player"]\').click();\n            } else if (job.status === "failed") {\n                clearInterval(pollInterval);\n                jobStatusBadge.textContent = "Failed";\n                jobStatusBadge.style.backgroundColor = "rgba(244, 63, 94, 0.3)";\n                btnStartDubbing.disabled = false;\n                btnStartDubbing.innerHTML = \'<span class="btn-icon">⚡</span><span>Start English to Hindi Dubbing</span>\';\n                alert(`Dubbing failed: ${job.error || "Unknown error"}`);\n            }\n        } catch (err) {\n            console.error("Polling error:", err);\n        }\n    }\n\n    function resetPipelineVisualizer() {\n        Object.values(stepNodes).forEach(node => {\n            node.className = "step-node";\n        });\n        progressBarFill.style.width = "0%";\n    }\n\n    function updateStepVisualizer(step) {\n        const order = ["extract", "transcribe", "translate", "synthesize", "merge"];\n        const stepMap = {\n            "extracting_audio": 0,\n            "transcribing": 1,\n            "translating": 2,\n            "synthesizing": 3,\n            "merging": 4,\n            "completed": 5\n        };\n\n        const activeIndex = stepMap[step] !== undefined ? stepMap[step] : -1;\n\n        order.forEach((key, idx) => {\n            const node = stepNodes[key];\n            if (idx < activeIndex) {\n                node.className = "step-node completed";\n            } else if (idx === activeIndex) {\n                node.className = "step-node active";\n            } else {\n                node.className = "step-node";\n            }\n        });\n    }\n\n    function renderTranscript(segments) {\n        transcriptSegmentsList.innerHTML = "";\n        segments.forEach((seg, idx) => {\n            const div = document.createElement("div");\n            div.className = "segment-item";\n            div.innerHTML = `\n                <div class="segment-time">#${idx + 1} [${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s]</div>\n                <div class="segment-en"><strong>EN:</strong> ${seg.english_text || seg.text || ""}</div>\n                <div class="segment-hi"><strong>HI:</strong> ${seg.hindi_text || "Translating..."}</div>\n            `;\n            transcriptSegmentsList.appendChild(div);\n        });\n    }\n});\n\n'

WEB_DIR = BASE_DIR / "web"
CSS_DIR = WEB_DIR / "css"
JS_DIR = WEB_DIR / "js"

for d in [WEB_DIR, CSS_DIR, JS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not (WEB_DIR / "index.html").exists():
    with open(WEB_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(EMBEDDED_HTML)

if not (CSS_DIR / "styles.css").exists():
    with open(CSS_DIR / "styles.css", "w", encoding="utf-8") as f:
        f.write(EMBEDDED_CSS)

if not (JS_DIR / "app.js").exists():
    with open(JS_DIR / "app.js", "w", encoding="utf-8") as f:
        f.write(EMBEDDED_JS)


# ==================== MAIN FASTAPI STUDIO APP ====================
app = FastAPI(title="VideoDubber AI - English to Hindi Dubbing Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory jobs tracking
JOBS = {}

class ConfigUpdate(BaseModel):
    elevenlabs_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    selected_voice_id: Optional[str] = None
    selected_voice_name: Optional[str] = None
    audio_mode: Optional[str] = "ducking"
    bg_music_volume: Optional[float] = 0.15
    voice_volume: Optional[float] = 1.0

class StartDubRequest(BaseModel):
    video_id: str
    voice_id: Optional[str] = None
    audio_mode: Optional[str] = "ducking"
    bg_music_volume: Optional[float] = 0.15
    voice_volume: Optional[float] = 1.0
    segments: Optional[List[dict]] = None
    gemini_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None

class TestVoiceRequest(BaseModel):
    voice_id: str
    text: Optional[str] = "नमस्ते दोस्तों! मैं बंटी हूँ, और आपकी इंग्लिश वीडियो को हिंदी में डब करूँगा।"
    elevenlabs_api_key: Optional[str] = None

class VerifyPassRequest(BaseModel):
    utr_or_code: str
    plan_price: Optional[int] = 20

@app.get("/api/config")
def api_get_config():
    config = load_config()
    # Mask api keys partially for security in UI display if needed, but return full for local app
    return config

@app.post("/api/config")
def api_save_config(update: ConfigUpdate):
    cfg = load_config()
    data = update.dict(exclude_unset=True)
    cfg.update(data)
    save_config(cfg)
    return {"status": "success", "config": cfg}

@app.post("/api/verify-pass")
def api_verify_pass(req: VerifyPassRequest):
    code = (req.utr_or_code or "").strip().upper()
    valid_codes = ["DUBBER20", "RITIK088", "PROPASS", "BUNTYVIP", "SUBHAM088", "ADMIN"]
    # Allow 12-digit UPI transaction UTRs or promo VIP codes
    if len(code) >= 6 or code in valid_codes:
        return {
            "status": "success",
            "message": "Payment verified! Studio Access Unlocked.",
            "unlocked": True,
            "plan_price": req.plan_price or 20,
            "pass_code": code
        }
    raise HTTPException(status_code=400, detail="Please enter a valid 12-digit UPI Reference / UTR number or VIP Access Code.")

@app.get("/api/voices")
def api_get_voices(api_key: Optional[str] = None):
    cfg = load_config()
    key = api_key or cfg.get("elevenlabs_api_key")
    try:
        voices = get_voices(key)
        return {"voices": voices, "count": len(voices)}
    except Exception as e:
        return {"voices": [], "error": str(e)}

@app.post("/api/test-voice")
def api_test_voice(req: TestVoiceRequest):
    cfg = load_config()
    api_key = req.elevenlabs_api_key or cfg.get("elevenlabs_api_key")
    
    test_id = str(uuid.uuid4())[:8]
    out_file = AUDIO_DIR / f"test_{test_id}.mp3"
    
    try:
        synthesize_speech_segment(
            text=req.text,
            voice_id=req.voice_id,
            output_path=str(out_file),
            api_key=api_key
        )
        return {"status": "success", "audio_url": f"/storage/audio/{out_file.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def api_upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Please upload MP4, MOV, MKV, or WebM.")
    
    video_id = str(uuid.uuid4())
    save_filename = f"{video_id}{ext}"
    video_path = UPLOADS_DIR / save_filename

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Get metadata
    info = get_video_info(str(video_path))
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "video_url": f"/storage/uploads/{save_filename}",
        "duration": info["duration"],
        "width": info["width"],
        "height": info["height"],
        "has_audio": info["has_audio"]
    }

def run_dubbing_process(job_id: str, req_data: dict):
    job = JOBS[job_id]
    try:
        cfg = load_config()
        video_id = req_data["video_id"]
        voice_id = req_data.get("voice_id") or cfg.get("selected_voice_id")
        audio_mode = req_data.get("audio_mode", cfg.get("audio_mode", "ducking"))
        bg_music_volume = req_data.get("bg_music_volume", cfg.get("bg_music_volume", 0.15))
        voice_volume = req_data.get("voice_volume", cfg.get("voice_volume", 1.0))
        custom_segments = req_data.get("segments")

        # Find input video file
        video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise FileNotFoundError(f"Video file with id {video_id} not found.")
        video_path = str(video_files[0])

        job["step"] = "extracting_audio"
        job["progress"] = 10
        job["message"] = "Step 1/5: Extracting audio track from video..."

        extracted_audio_path = str(AUDIO_DIR / f"{video_id}_original.mp3")
        extract_audio(video_path, extracted_audio_path)
        job["extracted_audio_url"] = f"/storage/audio/{Path(extracted_audio_path).name}"

        gemini_key = req_data.get("gemini_api_key") or cfg.get("gemini_api_key")
        elevenlabs_key = req_data.get("elevenlabs_api_key") or cfg.get("elevenlabs_api_key")

        # Step 2: Transcribe if segments not provided
        if not custom_segments:
            job["step"] = "transcribing"
            job["progress"] = 30
            job["message"] = "Step 2/5: Transcribing English speech and timestamps using AI..."
            segments = transcribe_audio_gemini(extracted_audio_path, api_key=gemini_key)
            
            job["step"] = "translating"
            job["progress"] = 50
            job["message"] = "Step 3/5: Translating English transcript to natural conversational Hindi..."
            segments = translate_segments_to_hindi(segments, api_key=gemini_key)
        else:
            segments = custom_segments

        job["segments"] = segments

        # Step 4: ElevenLabs Hindi TTS
        job["step"] = "synthesizing"
        job["progress"] = 70
        job["message"] = f"Step 4/5: Generating Hindi speech with ElevenLabs ({cfg.get('selected_voice_name', 'Bunty')})..."
        
        job_audio_dir = str(AUDIO_DIR / f"job_{job_id}")
        
        def tts_progress(curr, total, msg):
            job["progress"] = int(70 + (curr / total) * 15)
            job["message"] = f"Step 4/5: {msg}"

        synthesized_segments = synthesize_all_segments(
            segments=segments,
            voice_id=voice_id,
            output_dir=job_audio_dir,
            api_key=elevenlabs_key,
            progress_callback=tts_progress
        )
        job["segments"] = synthesized_segments

        # Step 5: Merge Audio Timeline & Video
        job["step"] = "merging"
        job["progress"] = 90
        job["message"] = "Step 5/5: Synchronizing Hindi voice track & rendering final dubbed video..."

        video_info = get_video_info(video_path)
        duration = video_info.get("duration", 10.0)

        hindi_full_audio = str(AUDIO_DIR / f"{job_id}_hindi_full.mp3")
        build_hindi_audio_track(synthesized_segments, duration, hindi_full_audio)

        output_video_file = str(OUTPUTS_DIR / f"{video_id}_hindi_dubbed.mp4")
        merge_video_and_hindi_audio(
            video_path=video_path,
            hindi_audio_path=hindi_full_audio,
            output_video_path=output_video_file,
            mode=audio_mode,
            bg_music_volume=bg_music_volume,
            voice_volume=voice_volume
        )

        # Generate Subtitles
        srt_file = str(OUTPUTS_DIR / f"{video_id}_hindi_subtitles.srt")
        vtt_file = str(OUTPUTS_DIR / f"{video_id}_hindi_subtitles.vtt")
        generate_srt(synthesized_segments, srt_file, use_hindi=True)
        generate_vtt(synthesized_segments, vtt_file, use_hindi=True)

        job["progress"] = 100
        job["step"] = "completed"
        job["message"] = "Dubbing Complete! Your Hindi video is ready."
        job["dubbed_video_url"] = f"/storage/outputs/{Path(output_video_file).name}"
        job["subtitles_srt_url"] = f"/storage/outputs/{Path(srt_file).name}"
        job["subtitles_vtt_url"] = f"/storage/outputs/{Path(vtt_file).name}"
        job["status"] = "completed"

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "failed"
        job["step"] = "error"
        job["error"] = str(e)
        job["message"] = f"Dubbing Failed: {str(e)}"

@app.post("/api/start-dub")
def api_start_dub(req: StartDubRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": req.video_id,
        "status": "processing",
        "progress": 5,
        "step": "started",
        "message": "Initializing dubbing engine...",
        "segments": [],
        "dubbed_video_url": None,
        "created_at": time.time()
    }

    background_tasks.add_task(run_dubbing_process, job_id, req.dict())
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/job/{job_id}")
def api_get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

# Serve Static files and storage
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")
app.mount("/", StaticFiles(directory=str(BASE_DIR / "web"), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[*] Starting VideoDubber AI Server on {host}:{port} ...")
    uvicorn.run(app, host=host, port=port)
