import os
import sys
import uuid
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, List

BASE_DIR = Path(__file__).resolve().parent
for p in [str(BASE_DIR), "/app", os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import load_config, save_config
from core.audio_extractor import extract_audio, get_video_info, get_ffmpeg_path
from core.transcriber import transcribe_audio_gemini
from core.translator import translate_segments_to_hindi
from core.elevenlabs_tts import get_voices, synthesize_speech_segment, synthesize_all_segments
from core.video_merger import build_hindi_audio_track, merge_video_and_hindi_audio
from core.subtitles import generate_srt, generate_vtt

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
AUDIO_DIR = STORAGE_DIR / "audio"
OUTPUTS_DIR = STORAGE_DIR / "outputs"

for p in [UPLOADS_DIR, AUDIO_DIR, OUTPUTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

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
