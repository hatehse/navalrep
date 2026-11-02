import json
import math
from pathlib import Path
import subprocess

from .models import MediaInfo


def probe_media(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path.resolve())],
        check=True, capture_output=True, text=True, timeout=60,
    )
    data = json.loads(result.stdout)
    videos = [s for s in data.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError(f"No video stream in {path.name}")
    video = videos[0]
    # Long audio tails must not permit clipping beyond the last video frame.
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration: {path.name}")
    return MediaInfo(duration, int(video["width"]), int(video["height"]), any(s.get("codec_type") == "audio" for s in data.get("streams", [])))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
