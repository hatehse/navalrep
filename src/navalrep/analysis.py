"""Sequential scene, speech and loudness analysis.

Only the short-lived Whisper child process owns a CUDA model. Audio analysis
keeps small overlapping blocks in RAM, even for a multi-hour source episode.
"""

from __future__ import annotations

from dataclasses import replace
import gc
import importlib.util
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import TYPE_CHECKING

from .models import AudioPeak, Segment, Transcript, Word

if TYPE_CHECKING:
    from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


class WhisperOutOfMemoryError(RuntimeError):
    """A CUDA allocation failure explicitly reported by the inference worker."""


def _is_cuda_oom(error: BaseException) -> bool:
    """Recognize CUDA OOM without mistaking driver/model errors for memory pressure.

    Host MemoryError and generic cuBLAS initialization/allocation failures are
    intentionally excluded: moving those failures to CPU can hide a broken
    installation or consume still more exhausted system RAM.
    """
    message = str(error).casefold()
    return (
        ("cuda" in message and "out of memory" in message)
        or "cuda_error_out_of_memory" in message
        or "cuda_error_memory_allocation" in message
    )


def _cuda_library_paths() -> list[str]:
    """Locate pip-provided CUDA DLLs without importing torch or creating CUDA."""
    candidates: list[Path] = []
    nvidia = importlib.util.find_spec("nvidia")
    if nvidia is not None:
        for root in nvidia.submodule_search_locations or ():
            for package in ("cublas", "cudnn", "cuda_runtime", "nvjitlink"):
                candidates.append(Path(root) / package / "lib")
                candidates.append(Path(root) / package / "bin")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.origin is not None:
        candidates.append(Path(torch_spec.origin).parent / "lib")
    return list(dict.fromkeys(str(path) for path in candidates if path.is_dir()))


def detect_scenes(source: Path, config: PipelineConfig) -> list[float]:
    """Return sorted shot boundaries, including the beginning and final frame.

    PySceneDetect decodes on the CPU; no GPU decoder or inference model is
    retained while later stages run. Boundaries are seconds in source time.
    """
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(source))
    try:
        manager = SceneManager()
        manager.auto_downscale = False
        manager.downscale = config.scene_downscale
        manager.add_detector(
            ContentDetector(
                threshold=config.scene_threshold,
                min_scene_len=config.scene_min_frames,
            )
        )
        manager.detect_scenes(video=video, show_progress=False)
        scenes = manager.get_scene_list(start_in_scene=True)
        boundaries = {0.0}
        for start, end in scenes:
            boundaries.add(float(start.get_seconds()))
            boundaries.add(float(end.get_seconds()))
        duration = float(video.duration.get_seconds())
        if math.isfinite(duration) and duration > 0:
            boundaries.add(duration)
        if len(boundaries) < 2:
            raise RuntimeError(f"Scene detection produced no usable duration: {source}")
        return sorted(boundaries)
    finally:
        # VideoStream owns its decoder. Release it before starting the next stage.
        del video
        gc.collect()


def extract_audio(source: Path, dest: Path, config: PipelineConfig) -> None:
    """Extract a temporary, uncompressed mono 16 kHz track without using CUDA."""
    command = [
        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", str(config.threads), "-i", str(source), "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-map_metadata", "-1", str(dest),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=config.timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[-6000:].strip()}")
    if not dest.is_file() or dest.stat().st_size <= 44:
        raise RuntimeError("Audio extraction returned an empty WAV file")


def _release_whisper(model: object | None, segments: object | None) -> None:
    """Release both the generator's model reference and CTranslate2 allocations."""
    if segments is not None:
        close = getattr(segments, "close", None)
        if close is not None:
            close()
    if model is not None:
        backend = getattr(model, "model", None)
        unload = getattr(backend, "unload_model", None)
        if unload is not None:
            unload()
    # The caller drops its final references before emptying the PyTorch cache.


def _whisper_worker(audio: str, config: PipelineConfig, result_path: str) -> None:
    """Spawn target; write JSON rather than filling an IPC pipe with an episode."""
    model = None
    segments = None
    dll_handles = []
    payload: dict = {}
    try:
        if sys.platform == "win32":
            for directory in _cuda_library_paths():
                dll_handles.append(os.add_dll_directory(directory))
        from faster_whisper import WhisperModel

        model_name = str(config.model_dir) if config.model_dir is not None else config.whisper_model
        model = WhisperModel(
            model_name,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=1,
            local_files_only=config.offline,
        )
        segments, info = model.transcribe(
            audio,
            language=config.language,
            beam_size=config.beam_size,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        words: list[dict] = []
        texts: list[dict] = []
        # faster-whisper returns a lazy generator: iteration performs inference.
        for segment in segments:
            texts.append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
            for word in segment.words or ():
                if word.end > word.start and word.word.strip():
                    words.append({"start": word.start, "end": word.end, "text": word.word.strip()})
        payload = {
            "words": words, "segments": texts, "language": info.language,
            # CT2 may resolve generic int8 to int8_float32 on CPU. Report the
            # effective backend, while the saved config retains the request.
            "device": getattr(getattr(model, "model", None), "device", config.device),
            "compute_type": getattr(getattr(model, "model", None), "compute_type", config.compute_type),
        }
    except Exception as error:
        payload = {
            "error": f"{type(error).__name__}: {error}",
            "error_type": type(error).__name__,
            "failure_kind": "cuda_oom" if config.device == "cuda" and _is_cuda_oom(error) else "inference_error",
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            _release_whisper(model, segments)
        except Exception:
            LOGGER.warning("Explicit CTranslate2 cleanup failed; child process will exit", exc_info=True)
        segments = None
        model = None
        gc.collect()
        if config.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                # CTranslate2 has its own allocator. Process exit below also
                # releases it in environments without PyTorch.
                pass
            except Exception:
                LOGGER.warning("PyTorch CUDA cache cleanup failed; child process will exit", exc_info=True)
        for handle in dll_handles:
            handle.close()
    Path(result_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _transcribe_once(audio: Path, config: PipelineConfig) -> Transcript:
    """Finish and join one Whisper child before returning or raising an error.

    ``empty_cache`` alone cannot release CTranslate2's CUDA context. A spawned
    process provides a hard memory boundary before NVENC is allowed to start.
    Offline mode uses a previously downloaded Hugging Face cache or model_dir.
    """
    if config.model_dir is not None and not Path(config.model_dir).is_dir():
        raise FileNotFoundError(f"Local Whisper model directory does not exist: {config.model_dir}")
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="whisper-", dir=audio.parent) as scratch:
        result_path = Path(scratch) / "transcript.json"
        process = context.Process(
            target=_whisper_worker,
            args=(str(audio), config, str(result_path)),
            name="whisper-sequential",
        )
        # Linux's dynamic loader reads LD_LIBRARY_PATH at Python startup, so it
        # must be inherited by the spawned child rather than set inside it.
        # Only this process launch sees additions; existing paths are preserved.
        loader_variable = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
        previous_loader_path = os.environ.get(loader_variable)
        library_paths = _cuda_library_paths()
        try:
            if library_paths:
                os.environ[loader_variable] = os.pathsep.join(
                    library_paths + ([previous_loader_path] if previous_loader_path else [])
                )
            process.start()
        finally:
            if previous_loader_path is None:
                os.environ.pop(loader_variable, None)
            else:
                os.environ[loader_variable] = previous_loader_path
        try:
            process.join(timeout=config.timeout)
            if process.is_alive():
                raise TimeoutError(f"Whisper exceeded {config.timeout} seconds")
            if process.exitcode != 0:
                raise RuntimeError(f"Whisper child failed with exit code {process.exitcode}")
            if not result_path.is_file():
                raise RuntimeError("Whisper child exited without producing a transcript")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if "error" in payload:
                LOGGER.debug("Whisper worker traceback: %s", payload.get("traceback", ""))
                error_class = WhisperOutOfMemoryError if payload.get("failure_kind") == "cuda_oom" else RuntimeError
                raise error_class(f"Whisper transcription failed: {payload['error']}")
            return Transcript(
                words=tuple(Word(**word) for word in payload["words"]),
                segments=tuple(Segment(**segment) for segment in payload["segments"]),
                language=payload["language"],
                device=payload["device"],
                compute_type=payload["compute_type"],
            )
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()


def transcribe(audio: Path, config: PipelineConfig) -> Transcript:
    """Run sequential ASR, optionally retrying a CUDA OOM using CPU/system RAM.

    CTranslate2 does not provide transparent layer offload between GPU VRAM and
    system RAM. The CPU retry loads the same model afresh as INT8, only after the
    failed GPU process has fully exited. It is slower and starts the episode's
    transcription from the beginning. Other errors are never retried as OOM.
    """
    LOGGER.info(
        "Whisper attempt: model=%s device=%s compute_type=%s beam_size=%s",
        config.whisper_model, config.device, config.compute_type, config.beam_size,
    )
    try:
        return _transcribe_once(audio, config)
    except WhisperOutOfMemoryError:
        if config.device != "cuda" or not config.cpu_fallback:
            raise
        LOGGER.warning(
            "Whisper CUDA out of memory; the GPU worker has exited. Retrying the "
            "same model on CPU/int8 using system RAM. This can be much slower."
        )
        cpu_config = replace(config, device="cpu", compute_type="int8", cpu_fallback=False)
        return _transcribe_once(audio, cpu_config)


def analyze_audio(audio: Path, config: PipelineConfig) -> list[AudioPeak]:
    """Find sudden loudness changes using librosa RMS, returning dB strength.

    Only a 30-second waveform block is decoded at once. The retained RMS series
    is tiny compared with decoded audio; its baseline spans block boundaries.
    """
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter
    import soundfile as sf

    frame_length = 2048
    hop = config.audio_hop
    if not 0 < hop <= frame_length:
        raise ValueError(f"audio_hop must be in [1, {frame_length}]")
    rms_parts = []
    with sf.SoundFile(audio) as stream:
        sample_rate = stream.samplerate
        carry = np.empty(0, dtype=np.float32)
        while True:
            block = stream.read(sample_rate * 30, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = np.mean(block, axis=1)
            samples = np.concatenate((carry, mono))
            if len(samples) < frame_length:
                carry = samples
                continue
            rms = librosa.feature.rms(
                y=samples, frame_length=frame_length, hop_length=hop, center=False,
            )[0]
            rms_parts.append(rms)
            # Retain the unprocessed tail; each frame is evaluated exactly once.
            carry = samples[len(rms) * hop :].copy()
    if not rms_parts:
        return []
    rms_values = np.concatenate(rms_parts)
    db = librosa.amplitude_to_db(np.maximum(rms_values, 1e-7), ref=1.0, top_db=None)
    baseline_frames = max(3, int(round(3 * sample_rate / hop)) | 1)
    baseline = median_filter(db, size=baseline_frames, mode="nearest")
    rise = np.maximum(0.0, np.diff(db, prepend=db[0]))
    strength = np.maximum(db - baseline, rise)
    # Suppress quiet-room/quantization transitions that are large only in dB.
    strength[db < -45.0] = 0
    # A short loud burst produces a wide plateau in dB. Selecting its tallest
    # frame would let tiny floating-point/RMS differences move the timestamp
    # deep into that plateau. Timestamp the onset of each above-threshold
    # region, while keeping its maximum dB strength for candidate ranking.
    active = strength >= config.audio_peak_db
    starts = np.flatnonzero(active & np.concatenate(([True], ~active[:-1])))
    ends = np.flatnonzero(active & np.concatenate((~active[1:], [True]))) + 1
    candidates = [
        (int(start), float(np.max(strength[start:end])))
        for start, end in zip(starts, ends)
    ]
    minimum_gap = max(1, int(config.min_peak_gap * sample_rate / hop))
    selected: list[tuple[int, float]] = []
    for index, height in sorted(candidates, key=lambda item: (-item[1], item[0])):
        if all(abs(index - other) >= minimum_gap for other, _ in selected):
            selected.append((index, height))
    return [
        AudioPeak(
            time=float((index * hop + frame_length / 2) / sample_rate),
            strength=float(height),
        )
        for index, height in sorted(selected)
    ]
