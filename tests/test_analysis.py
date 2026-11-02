"""CPU integration checks for real media analysis; no model downloads or CUDA."""

from dataclasses import replace
import importlib.util
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from navalrep.analysis import analyze_audio, detect_scenes, extract_audio, transcribe
from navalrep.config import PipelineConfig


def available(*modules):
    return all(importlib.util.find_spec(name) is not None for name in modules)


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
AUDIO_AVAILABLE = available("numpy", "librosa", "soundfile", "scipy")
SCENES_AVAILABLE = available("scenedetect", "cv2")


class MediaAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="navalrep-analysis-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = PipelineConfig()

    def ffmpeg(self, *arguments):
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(AUDIO_AVAILABLE, "Requires numpy, librosa, soundfile and scipy")
    def test_detects_real_loudness_spike_crossing_audio_block_boundary(self):
        import numpy as np
        import soundfile as sf

        sample_rate = 16000
        time = np.arange(33 * sample_rate, dtype=np.float32) / sample_rate
        amplitude = np.full(time.shape, 0.005, dtype=np.float32)
        # Identical bursts appear within a block and across its 30-second
        # boundary. Their onset estimates should agree independently of where
        # the decoder split its input buffer.
        for onset_time in (9.98, 29.98):
            amplitude[(time >= onset_time) & (time < onset_time + 0.3)] = 0.6
        waveform = amplitude * np.sin(2 * np.pi * 220 * time)
        audio = self.root / "spike.wav"
        sf.write(audio, waveform, sample_rate, subtype="PCM_16")

        peaks = analyze_audio(audio, self.config)

        self.assertEqual(len(peaks), 2, peaks)
        for peak, expected in zip(peaks, (9.98, 29.98)):
            self.assertAlmostEqual(peak.time, expected, delta=0.12)
            self.assertGreaterEqual(peak.strength, self.config.audio_peak_db)
        self.assertAlmostEqual(peaks[1].time - peaks[0].time, 20, delta=0.04)

    @unittest.skipUnless(AUDIO_AVAILABLE, "Requires numpy, librosa, soundfile and scipy")
    def test_constant_tone_does_not_create_peaks_at_block_boundaries(self):
        import numpy as np
        import soundfile as sf

        sample_rate = 16000
        time = np.arange(62 * sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.08 * np.sin(2 * np.pi * 220 * time)
        audio = self.root / "constant.wav"
        sf.write(audio, waveform, sample_rate, subtype="PCM_16")

        self.assertEqual(analyze_audio(audio, self.config), [])

    @unittest.skipUnless(FFMPEG_AVAILABLE, "Requires ffmpeg")
    def test_ffmpeg_extracts_nonempty_mono_16khz_pcm_audio(self):
        source = self.root / "stereo48k.wav"
        destination = self.root / "mono16k.wav"
        self.ffmpeg(
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-ac", "2", "-c:a", "pcm_s16le", str(source),
        )

        extract_audio(source, destination, self.config)

        with wave.open(str(destination), "rb") as output:
            self.assertEqual(output.getframerate(), 16000)
            self.assertEqual(output.getnchannels(), 1)
            self.assertEqual(output.getsampwidth(), 2)
            self.assertAlmostEqual(output.getnframes() / output.getframerate(), 2, places=2)
            self.assertTrue(any(output.readframes(output.getnframes())))

    @unittest.skipUnless(FFMPEG_AVAILABLE and SCENES_AVAILABLE, "Requires ffmpeg, scenedetect and OpenCV")
    def test_three_color_shots_return_actual_edit_boundaries(self):
        source = self.root / "three-shots.mp4"
        self.ffmpeg(
            "-filter_complex_threads", "1",
            "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=1",
            "-f", "lavfi", "-i", "color=c=white:s=160x90:r=25:d=1",
            "-f", "lavfi", "-i", "color=c=red:s=160x90:r=25:d=1",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "mpeg4", "-q:v", "2", "-threads", "1", str(source),
        )
        config = replace(self.config, scene_threshold=15.0, scene_min_frames=5, scene_downscale=1)

        boundaries = detect_scenes(source, config)

        self.assertEqual(len(boundaries), 4, boundaries)
        for actual, expected in zip(boundaries, (0.0, 1.0, 2.0, 3.0)):
            self.assertAlmostEqual(actual, expected, delta=1 / 25)

    def test_whisper_timeout_terminates_and_joins_child_before_return(self):
        class StalledProcess:
            exitcode = None
            alive = False
            terminated = False
            closed = False

            def __init__(self):
                self.joins = []

            def start(self):
                self.alive = True

            def join(self, timeout=None):
                self.joins.append(timeout)

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def close(self):
                self.closed = True

        child = StalledProcess()
        context = SimpleNamespace(Process=lambda **kwargs: child)
        with patch("navalrep.analysis.multiprocessing.get_context", return_value=context), patch(
            "navalrep.analysis._cuda_library_paths", return_value=[]
        ):
            with self.assertRaises(TimeoutError):
                transcribe(self.root / "audio.wav", replace(self.config, timeout=1))

        self.assertTrue(child.terminated)
        self.assertFalse(child.alive)
        self.assertTrue(child.closed)
        self.assertEqual(child.joins, [1, 10])
        self.assertEqual(list(self.root.glob("whisper-*")), [])

    @unittest.skipUnless(available("faster_whisper", "torch"), "Requires faster-whisper and torch")
    def test_real_whisper_worker_reports_model_failure_and_cleans_up(self):
        """Exercise actual spawn/IPC/error cleanup without a model or GPU."""
        audio = self.root / "preserved-audio.wav"
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 4000)
        original_audio = audio.read_bytes()
        model = self.root / "empty-local-model"
        model.mkdir()
        config = replace(self.config, model_dir=model, offline=True, timeout=45)

        # An existing local directory takes the loader's local path. Explicit
        # offline flags also prohibit an accidental fallback to a download.
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            with self.assertRaisesRegex(RuntimeError, r"^Whisper transcription failed:"):
                transcribe(audio, config)

        self.assertEqual(audio.read_bytes(), original_audio)
        self.assertTrue(model.is_dir())
        self.assertEqual(list(model.iterdir()), [])
        self.assertEqual(list(self.root.glob("whisper-*")), [])
        self.assertFalse([
            child for child in multiprocessing.active_children()
            if child.name == "whisper-sequential"
        ])


if __name__ == "__main__":
    unittest.main()
