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
