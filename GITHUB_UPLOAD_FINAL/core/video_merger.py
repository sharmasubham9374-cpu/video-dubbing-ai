import os
import subprocess
import shutil
from pathlib import Path
from core.audio_extractor import get_ffmpeg_path, get_video_info

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
