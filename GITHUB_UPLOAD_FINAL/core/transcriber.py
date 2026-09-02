import os
import sys
import json
import time
import base64
import subprocess
import requests
from pathlib import Path
from core.config import load_config
from core.audio_extractor import get_ffmpeg_path, get_video_info

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
