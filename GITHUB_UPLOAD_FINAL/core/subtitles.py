from pathlib import Path

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

