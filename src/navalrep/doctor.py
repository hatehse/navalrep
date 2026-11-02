"""Fail early on missing executables, filters, model files or actual NVENC support."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

from .config import PipelineConfig


def check_environment(config: PipelineConfig, *, encode_test: bool = True) -> list[dict]:
    checks = []

    def record(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for module in ("scenedetect", "cv2", "faster_whisper", "librosa", "soundfile", "numpy", "scipy", "torch"):
        record(module, importlib.util.find_spec(module) is not None, "Python dependency")
    for exe in (config.ffmpeg, config.ffprobe):
        record(exe, shutil.which(exe) is not None, "Executable on PATH")
    if config.model_dir is not None:
        model_path = Path(config.model_dir)
        required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
        missing = [name for name in required if not (model_path / name).is_file()]
        record("Whisper model", not missing, "Missing: " + ", ".join(missing) if missing else str(model_path))
    else:
        # This resolves cached files only: no HTTP and no model/GPU initialization.
        try:
            from faster_whisper.utils import download_model
            model_path = download_model(config.whisper_model, local_files_only=True)
            record("Whisper cache", True, str(model_path))
        except Exception as exc:
            record("Whisper cache", not config.offline, f"Not cached ({type(exc).__name__}); use download-model before offline run")
    if shutil.which(config.ffmpeg):
        try:
            filters = subprocess.run([config.ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, check=True, timeout=30)
            record("libass", any(" subtitles " in line for line in filters.stdout.splitlines()), "FFmpeg subtitles filter")
            encoders = subprocess.run([config.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True, timeout=30)
            record("h264_nvenc", "h264_nvenc" in encoders.stdout, "Encoder compiled into FFmpeg")
            if encode_test:
                with tempfile.TemporaryDirectory(prefix="navalrep-doctor-") as tmp:
                    result = subprocess.run([
                        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30:d=0.2",
                        "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-frames:v", "3",
                        str(Path(tmp) / "nvenc.mp4"),
                    ], capture_output=True, text=True, timeout=45)
                    record("NVENC hardware", result.returncode == 0, result.stderr[-2000:].strip() or "Three frames encoded on GPU")
        except Exception as exc:
            record("FFmpeg checks", False, str(exc))
    return checks
