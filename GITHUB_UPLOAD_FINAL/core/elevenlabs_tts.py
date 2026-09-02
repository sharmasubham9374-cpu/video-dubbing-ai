import os
import json
import asyncio
import requests
from pathlib import Path
import edge_tts
from core.config import load_config

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
