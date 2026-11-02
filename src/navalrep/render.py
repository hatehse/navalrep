"""Speed-adjusted subtitles and sequential NVENC rendering.

The SRT remains a useful output artifact. A temporary ASS conversion gives libass
an explicit canvas size, so ``font_size`` means pixels on the final video.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .config import PipelineConfig
from .models import Clip, Transcript


def _stamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _join_words(words: list[str]) -> str:
    text = " ".join(word.strip() for word in words if word.strip())
    # Whisper generally attaches punctuation to words; also handle separate
    # punctuation tokens without introducing spaces before them.
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    return re.sub(r"([¿¡])\s+", r"\1", text).strip()


def write_srt(
    transcript: Transcript, clip: Clip, path: Path, config: PipelineConfig
) -> int:
    """Write clipped word captions on the *output* timeline; return cue count.

    Word timestamps are preferred. Segment timestamps are a conservative
    fallback when the transcriber could not align any words.
    """
    if config.speed <= 0 or clip.end <= clip.start:
        raise ValueError("Invalid playback speed or empty clip")
    duration = (clip.end - clip.start) / config.speed
    candidates: list[tuple[float, float, str]] = []
    word_group: list[str] = []
    group_start = group_end = 0.0

    def flush() -> None:
        if word_group:
            candidates.append((group_start, group_end, _join_words(word_group)))
            word_group.clear()

    for word in transcript.words:
        if word.end <= clip.start or word.start >= clip.end:
            continue
        start = max(0.0, (word.start - clip.start) / config.speed)
        end = min(duration, (word.end - clip.start) / config.speed)
        if end <= start or not word.text.strip():
            continue
        if word_group and (
            len(word_group) >= config.subtitle_words
            or start - group_end > 0.65
            or end - group_start > 3.5
        ):
            flush()
        if not word_group:
            group_start = start
        word_group.append(word.text)
        group_end = max(group_end, end)
        if re.search(r"[.!?…][\"'»)]?$", word.text.strip()):
            flush()
    flush()

    if not candidates:
        for segment in transcript.segments:
            if segment.end <= clip.start or segment.start >= clip.end:
                continue
            start = max(0.0, (segment.start - clip.start) / config.speed)
            end = min(duration, (segment.end - clip.start) / config.speed)
            if end > start and segment.text.strip():
                candidates.append((start, end, segment.text.strip()))

    cues: list[str] = []
    previous_end = 0
    duration_ms = math.floor(duration * 1_000 + 1e-6)
    for start, end, text in sorted(candidates):
        start_ms = max(previous_end, round(start * 1_000))
        end_ms = min(duration_ms, round(end * 1_000))
        if end_ms <= start_ms:
            continue
        # A subtitle is text, not an opportunity to inject extra SRT blocks.
        text = " ".join(text.split())
        cues.append(
            f"{len(cues) + 1}\n{_stamp(start_ms)} --> {_stamp(end_ms)}\n{text}\n"
        )
        previous_end = end_ms
    Path(path).write_text("\n".join(cues), encoding="utf-8")
    return len(cues)


def _run(command: list[str], cwd: Path, timeout: float) -> None:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{Path(command[0]).name} exited {result.returncode}: "
            f"{result.stderr[-6000:].strip()}"
        )


def _duration(path: Path, config: PipelineConfig) -> float:
    result = subprocess.run(
        [config.ffprobe, "-v", "error", "-select_streams", "V:0", "-show_entries",
         "format=duration:stream=width,height,duration,sample_aspect_ratio",
         "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True, timeout=min(config.timeout, 120),
    )
    media = json.loads(result.stdout)
    streams = media.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("Encoded MP4 has no primary video stream")
    stream = streams[0]
    if (stream.get("width"), stream.get("height")) != (config.width, config.height):
        raise RuntimeError("Encoded video dimensions do not match the requested 9:16 canvas")
    if stream.get("sample_aspect_ratio") != "1:1":
        raise RuntimeError("Encoded video must use square pixels to preserve its 9:16 aspect")
    try:
        video_duration = float(stream["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Cannot verify the encoded video stream duration") from exc
    if (not math.isfinite(video_duration)
            or not config.min_duration <= video_duration <= config.max_duration):
        raise RuntimeError(
            f"Encoded video stream duration {video_duration:.6f}s is outside "
            f"[{config.min_duration}, {config.max_duration}]s"
        )
    # Audio can outlive a truncated video stream. Check both independently;
    # a valid container duration alone must never certify a complete clip.
    duration = float(media["format"]["duration"])
    if not math.isfinite(duration):
        raise RuntimeError("FFprobe returned a non-finite duration")
    return duration


def _prepare_ass(work: Path, config: PipelineConfig) -> None:
    _run(
        [config.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
         "-i", "subtitles.srt", "subtitles.ass"],
        work, min(config.timeout, 120),
    )
    ass = (work / "subtitles.ass").read_text(encoding="utf-8-sig")
    ass = re.sub(r"(?m)^PlayResX:.*$", f"PlayResX: {config.width}", ass)
    ass = re.sub(r"(?m)^PlayResY:.*$", f"PlayResY: {config.height}", ass)
    (work / "subtitles.ass").write_text(ass, encoding="utf-8")


def render_clip(
    source: Path,
    clip: Clip,
    transcript: Transcript,
    output_dir: Path,
    stem: str,
    config: PipelineConfig,
) -> dict[str, str | float]:
    """Render one clip; never fall back silently from NVENC to CPU encoding."""
    source = Path(source).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not stem or Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError("Output stem must be a plain file name")
    if not re.fullmatch(r"[\w .-]+", config.font_name, flags=re.UNICODE):
        raise ValueError("Font name must contain only letters, digits, spaces, '.', '_' or '-'")
    if config.zoom < 1 or config.width % 2 or config.height % 2:
        raise ValueError("Zoom must be >= 1 and output dimensions must be even")
    requested = (clip.end - clip.start) / config.speed
    if not config.min_duration - 1e-7 <= requested <= config.max_duration + 1e-7:
        raise ValueError("Selected clip is outside final duration limits")
    # Whole frame counts avoid extending the last video frame beyond 55 s.
    target_duration = math.floor(requested * config.fps + 1e-6) / config.fps
    if target_duration < config.min_duration:
        raise ValueError("Selected clip is too short after frame rounding")
    video_path = output_dir / f"{stem}.mp4"
    srt_path = output_dir / f"{stem}.srt"
    if video_path.exists() or srt_path.exists():
        raise FileExistsError(f"Refusing to overwrite {stem}")

    published: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix=".render-", dir=output_dir) as directory:
            work = Path(directory)
            # Frame rounding may shorten the output by less than one frame.
            # Keep even the standalone SRT inside that final duration.
            subtitle_clip = replace(clip, end=clip.start + target_duration * config.speed)
            cue_count = write_srt(transcript, subtitle_clip, work / "subtitles.srt", config)
            filters = [f"setpts=(PTS-STARTPTS)/{config.speed:.12g}"]
            if config.mirror:
                filters.append("hflip")
            zoom_width = math.ceil(config.width * config.zoom / 2) * 2
            zoom_height = math.ceil(config.height * config.zoom / 2) * 2
            filters.extend([
                f"scale={zoom_width}:{zoom_height}:force_original_aspect_ratio=increase",
                f"crop={config.width}:{config.height}:(iw-ow)/2:(ih-oh)/2",
                "setsar=1", f"fps={config.fps}",
            ])
            if cue_count:
                _prepare_ass(work, config)
                style = (
                    f"FontName={config.font_name},FontSize={config.font_size},Bold=-1,"
                    "PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,"
                    "BorderStyle=1,Outline=5,Shadow=1,Alignment=5,"
                    "MarginL=65,MarginR=65,MarginV=65,WrapStyle=0"
                )
                filters.append(f"subtitles=filename=subtitles.ass:force_style='{style}'")
            command = [
                config.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-threads", str(config.threads), "-filter_threads", str(config.threads),
                "-ss", f"{clip.start:.9f}", "-t", f"{clip.end - clip.start:.9f}",
                "-i", str(source), "-map", "0:V:0", "-map", "0:a:0",
                "-vf", ",".join(filters),
                "-af", f"asetpts=PTS-STARTPTS,atempo={config.speed:.12g},"
                       f"atrim=duration={target_duration:.9f},asetpts=PTS-STARTPTS",
                "-c:v", "h264_nvenc", "-preset", config.nvenc_preset,
                "-rc", "vbr", "-cq", str(config.cq), "-b:v", "0",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-t", f"{target_duration:.9f}", "-map_metadata", "-1",
                "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1",
                "-map_chapters", "-1", "-metadata", "title=", "-metadata", "comment=",
                "-metadata", "encoder=", "-metadata:s:v", "encoder=",
                "-metadata:s:a", "encoder=", "-metadata:s:v", "handler_name=Video",
                "-metadata:s:a", "handler_name=Audio", "-fflags", "+bitexact",
                "-movflags", "+faststart", "render.mp4",
            ]
            _run(command, work, config.timeout)
            actual_duration = _duration(work / "render.mp4", config)
            if not config.min_duration <= actual_duration <= config.max_duration:
                raise RuntimeError(
                    f"Encoded duration {actual_duration:.6f}s is outside "
                    f"[{config.min_duration}, {config.max_duration}]s"
                )
            # Remember ownership *before* copying: a failed copy can leave a
            # partial file that must also be cleaned up.
            for staged, destination in ((work / "render.mp4", video_path),
                                         (work / "subtitles.srt", srt_path)):
                try:
                    with destination.open("xb") as target, staged.open("rb") as src:
                        published.append(destination)
                        shutil.copyfileobj(src, target)
                except FileExistsError:
                    raise FileExistsError(f"Refusing to overwrite {destination}") from None
        return {"video": str(video_path), "subtitles": str(srt_path),
                "duration": actual_duration}
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
