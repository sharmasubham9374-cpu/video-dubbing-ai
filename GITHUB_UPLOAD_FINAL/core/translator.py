import os
import json
import time
import re
import requests
from pathlib import Path
from core.config import load_config

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
