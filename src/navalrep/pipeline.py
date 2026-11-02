"""One episode, one stage, one GPU consumer at a time."""
from dataclasses import asdict
import hashlib
import logging
from pathlib import Path
import re
import tempfile
import uuid

from .analysis import analyze_audio, detect_scenes, extract_audio, transcribe
from .config import PipelineConfig
from .render import render_clip
from .selection import select_clips
from .util import atomic_json, probe_media

LOG = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}


def discover_inputs(config: PipelineConfig, files: list[Path]) -> list[Path]:
    if files:
        # Explicit requests must never silently disappear from the run report.
        return list(dict.fromkeys(p.resolve() for p in files))
    candidates = sorted(config.input_dir.glob("*"))
    # Resolve aliases so a repeated input cannot accidentally process twice.
    return list(dict.fromkeys(p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))


def caption_for(stem: str, config: PipelineConfig) -> str:
    choices = (
        "¡No manches! 😳 ¿Neta terminó así? Esto se puso bien intenso…",
        "¡Qué fuerte, banda! 🔥 ¿Tú qué habrías hecho en su lugar?",
        "¡Se armó el drama! 👀 Ese final sí me dejó con el pendiente…",
    )
    index = int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16) % len(choices)
    cta = f"Más clips en Telegram 👉 {config.telegram_url}" if config.telegram_url else "Más clips en nuestro Telegram: enlace en la bio."
    return f"{choices[index]}\n\n{cta}\n\n{' '.join(config.hashtags)}\n"


def process_episode(source: Path, config: PipelineConfig) -> dict:
    report = {"source": str(source), "status": "failed", "stage": "probe", "clips": [], "errors": []}
    info = probe_media(source, config.ffprobe)
    if not info.has_audio:
        raise ValueError("Input video has no audio stream")
    if info.duration < config.min_duration * config.speed:
        raise ValueError("Input is too short for a 35-second clip after speed change")
    safe_stem = re.sub(r"[^\w-]+", "_", source.stem, flags=re.UNICODE).strip("_")[:60] or "episode"
    # A unique subdirectory prevents output collisions across retries and same-name inputs.
    episode_dir = config.output_dir / f"{safe_stem}_{uuid.uuid4().hex[:12]}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix="episode-", dir=config.temp_dir) as work:
            audio = Path(work) / "audio.wav"
            report["stage"] = "scene detection"
            LOG.info("%s: scene detection (CPU)", source.name)
            boundaries = detect_scenes(source, config)
            report["stage"] = "audio extraction"
            extract_audio(source, audio, config)
            report["stage"] = "transcription"
            LOG.info("%s: Whisper %s / %s / %s (beam=%s, CPU retry on CUDA OOM=%s)",
                     source.name, config.whisper_model, config.device, config.compute_type,
                     config.beam_size, config.device == "cuda" and config.cpu_fallback)
            transcript = transcribe(audio, config)
            report["transcription"] = {
                "model": config.whisper_model,
                "requested_device": config.device,
                "requested_compute_type": config.compute_type,
                "device": transcript.device,
                "compute_type": transcript.compute_type,
                "cpu_fallback_used": config.device == "cuda" and transcript.device == "cpu",
            }
            LOG.info("%s: transcription finished on %s / %s", source.name,
                     transcript.device or "unspecified test backend",
                     transcript.compute_type or "unspecified test backend")
            # transcribe returns only AFTER the isolated model process has exited.
            report["stage"] = "audio analysis"
            LOG.info("%s: loudness analysis (CPU); Whisper process has exited", source.name)
            peaks = analyze_audio(audio, config)
            report["stage"] = "selection"
            clips = select_clips(boundaries, transcript, peaks, info.duration, config)
            atomic_json(episode_dir / "analysis.json", {
                "source": str(source), "media": asdict(info), "scene_boundaries": boundaries,
                "transcript": asdict(transcript), "audio_peaks": [asdict(p) for p in peaks],
                "candidates": [asdict(c) for c in clips], "config": _config_dict(config),
            })
            if not clips:
                report["status"] = "no_candidates"
                LOG.warning("%s: no cliffhangers meet score, scene-end and duration constraints", source.name)
            else:
                for number, clip in enumerate(clips, 1):
                    stem = f"clip_{number:02d}"
                    report["stage"] = f"render {stem}"
                    try:
                        LOG.info("%s: render %s %.3f..%.3f (score %.2f)", source.name, stem, clip.start, clip.end, clip.score)
                        artifacts = render_clip(source, clip, transcript, episode_dir, stem, config)
                        caption_path = episode_dir / f"{stem}.txt"
                        caption_path.write_text(caption_for(stem, config), encoding="utf-8")
                        report["clips"].append({**asdict(clip), **artifacts, "caption": str(caption_path)})
                    except Exception as exc:
                        LOG.exception("%s: clip %s failed; continuing", source.name, stem)
                        report["errors"].append({"stage": report["stage"], "error": str(exc)})
                report["status"] = "partial" if report["clips"] and report["errors"] else "ok" if report["clips"] else "failed"
            report["stage"] = "finished"
    except Exception as exc:
        LOG.exception("%s: stage %s failed", source.name, report["stage"])
        report["errors"].append({"stage": report["stage"], "error": str(exc)})
    finally:
        # TemporaryDirectory removes only this episode's owned files, even on failure.
        atomic_json(episode_dir / "report.json", report)
    return report


def _config_dict(config):
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def run_pipeline(config: PipelineConfig, files: list[Path]) -> list[dict]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    inputs = discover_inputs(config, files)
    if not inputs:
        raise ValueError("No input videos found. Put episode.mp4 in input/ or pass --input FILE")
    reports = []
    for source in inputs:
        try:
            if not source.is_file():
                raise FileNotFoundError(f"Input is not a file: {source}")
            if source.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported video extension: {source.suffix}")
            reports.append(process_episode(source, config))
        except Exception as exc:
            LOG.exception("Cannot process %s; continuing to next file", source)
            reports.append({"source": str(source), "status": "failed", "errors": [{"stage": "prepare", "error": str(exc)}], "clips": []})
    atomic_json(config.output_dir / f"run_{uuid.uuid4().hex[:12]}.json", reports)
    return reports
