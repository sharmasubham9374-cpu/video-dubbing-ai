import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"

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

