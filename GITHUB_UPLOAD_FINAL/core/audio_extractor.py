import os
import subprocess
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_BIN = BASE_DIR / "bin"

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

