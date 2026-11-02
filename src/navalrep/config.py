"""Validated, editable JSON configuration; all times are seconds."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    temp_dir: Path = Path("temp")
    model_dir: Path | None = None
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "es"
    offline: bool = True
    beam_size: int = 5
    cpu_threads: int = 4
    threads: int = 4
    timeout: int = 14400
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    scene_threshold: float = 27.0
    scene_min_frames: int = 15
    scene_downscale: int = 2
    audio_hop: int = 512
    audio_peak_db: float = 7.0
    min_peak_gap: float = 2.0
    trigger_phrases: tuple[str, ...] = (
        "quién es", "no puede ser", "qué hiciste", "no me digas",
        "te voy a matar", "es tu hijo", "estás vivo", "la verdad", "ayúdame",
    )
    min_duration: float = 35.0
    max_duration: float = 55.0
    target_duration: float = 45.0
    speed: float = 1.04
    max_clips: int = 5
    min_score: float = 2.0
    peak_lookback: float = 5.0
    max_silence_after_peak: float = 2.0
    max_overlap: float = 0.35
    width: int = 1080
    height: int = 1920
    fps: int = 30
    mirror: bool = True
    zoom: float = 1.08
    nvenc_preset: str = "p5"
    cq: int = 21
    font_name: str = "DejaVu Sans"
    font_size: int = 80
    subtitle_words: int = 6
    telegram_url: str = ""
    hashtags: tuple[str, ...] = ("#Novelas", "#Drama", "#QuéPasó", "#ParaTi", "#Series")

    def __post_init__(self):
        # Reject wrong JSON types early, before subprocesses or expensive analysis.
        for key in ("input_dir", "output_dir", "temp_dir"):
            if not isinstance(getattr(self, key), Path):
                raise ValueError(f"{key} must be a filesystem path")
        if self.model_dir is not None and not isinstance(self.model_dir, Path):
            raise ValueError("model_dir must be a filesystem path or null")
        defaults = {f.name: f.default for f in fields(self)}
        for key, default in defaults.items():
            value = getattr(self, key)
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean")
            if isinstance(default, int) and not isinstance(default, bool):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{key} must be a positive integer")
            if isinstance(default, float):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"{key} must be a finite number")
            if isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
        for key in ("ffmpeg", "ffprobe"):
            executable = getattr(self, key)
            if not executable:
                raise ValueError(f"{key} cannot be empty")
            if "/" in executable or "\\" in executable:
                # Rendering changes cwd; preserve custom relative executable paths.
                object.__setattr__(self, key, str(Path(executable).expanduser().resolve()))
        if not 35 <= self.min_duration <= self.target_duration <= self.max_duration <= 55:
            raise ValueError("Require 35 <= min_duration <= target_duration <= max_duration <= 55")
        if not 0.5 <= self.speed <= 2 or not 1 <= self.zoom <= 2:
            raise ValueError("speed must be 0.5..2, zoom 1..2")
        if self.width * 16 != self.height * 9 or self.width % 2 or self.height % 2:
            raise ValueError("Require even 9:16 output dimensions")
        if self.device != "cuda" or self.compute_type != "float16" or self.language != "es":
            raise ValueError("This workflow requires cuda / float16 / es")
        if self.nvenc_preset not in {f"p{i}" for i in range(1, 8)} or not 1 <= self.cq <= 51:
            raise ValueError("Invalid NVENC preset or CQ")
        if not 0 <= self.max_overlap < 1 or min(self.scene_threshold, self.audio_peak_db, self.peak_lookback, self.min_peak_gap) <= 0:
            raise ValueError("Invalid selection/audio/scene thresholds")
        if self.min_score < 0 or self.max_silence_after_peak < 0:
            raise ValueError("Scores and silence allowance cannot be negative")
        if self.audio_hop > 2048:
            raise ValueError("audio_hop cannot exceed the RMS frame length (2048)")
        if len(self.hashtags) != 5 or any(not isinstance(x, str) or not x.startswith("#") or any(c.isspace() for c in x) for x in self.hashtags):
            raise ValueError("Exactly five single-token hashtags are required")
        if not all(isinstance(x, str) and x.strip() for x in self.trigger_phrases):
            raise ValueError("trigger_phrases must contain nonempty strings")
        if self.telegram_url and not self.telegram_url.startswith("https://t.me/"):
            raise ValueError("telegram_url must be empty or start with https://t.me/")
        if any(c in self.font_name for c in "'\"\\,:;[]\n\r"):
            raise ValueError("font_name contains unsupported FFmpeg characters")


def load_config(path: Path | None) -> PipelineConfig:
    data = {} if path is None else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(data) - {f.name for f in fields(PipelineConfig)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for name in ("input_dir", "output_dir", "temp_dir", "model_dir"):
        if name in data and data[name] is not None:
            data[name] = Path(data[name]).expanduser()
    for name in ("trigger_phrases", "hashtags"):
        if name in data:
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be a JSON array")
            data[name] = tuple(data[name])
    return PipelineConfig(**data)
