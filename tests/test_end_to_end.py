"""Real CPU orchestration smoke test with explicit ASR/encoder test adapters.

The synthetic transcript is known input, not a speech recognition result. Only
the requested h264_nvenc encoder is replaced for this test; production keeps
the configured Whisper memory policy and NVENC requirements. This cannot
certify either GPU component.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from navalrep import pipeline, render
from navalrep.config import PipelineConfig
from navalrep.models import Segment, Transcript, Word


class CPUOrchestrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = ("numpy", "soundfile", "librosa", "scipy", "scenedetect", "cv2")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("CPU smoke requires FFmpeg/FFprobe and all analysis dependencies")
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
            text=True, check=True, timeout=15,
        ).stdout
        filters = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
            text=True, check=True, timeout=15,
        ).stdout
        if "libx264" not in encoders or "subtitles" not in filters:
            raise unittest.SkipTest("CPU smoke requires libx264 and libass subtitles")

    def test_real_analysis_selection_render_and_reports_with_known_transcript(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="navalrep-cpu-orchestration-") as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            temp_dir = root / "temp"
            temp_dir.mkdir()
            sentinel = temp_dir / "user-owned.txt"
            sentinel.write_text("preserve this file", encoding="utf-8")
            source = input_dir / "episodio con espacios y apóstrofo's.mp4"
            sample_rate = 16000
            times = np.arange(80 * sample_rate, dtype=np.float64) / sample_rate
            amplitudes = np.full(times.shape, 0.01)
            amplitudes[(times >= 48.7) & (times < 49.1)] = 0.65
            audio = root / "synthetic-audio.wav"
            sf.write(audio, amplitudes * np.sin(2 * np.pi * 220 * times), sample_rate)
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-filter_complex_threads", "1",
                "-f", "lavfi", "-i", "color=c=black:s=160x90:r=5:d=30",
                "-f", "lavfi", "-i", "color=c=white:s=160x90:r=5:d=20",
                "-f", "lavfi", "-i", "color=c=red:s=160x90:r=5:d=30",
                "-i", str(audio),
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map", "[v]", "-map", "3:a:0", "-c:v", "mpeg4", "-q:v", "2",
                "-threads", "1", "-c:a", "aac", "-t", "80",
                "-metadata", "title=ORIGINAL_EPISODE", str(source),
            ], capture_output=True, check=True, timeout=60)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            config = PipelineConfig(
                input_dir=input_dir, output_dir=root / "output", temp_dir=temp_dir,
                width=180, height=320, fps=5, font_size=12, threads=1,
                max_clips=1, scene_threshold=15.0, scene_min_frames=5,
                scene_downscale=1, telegram_url="https://t.me/local_test",
            )
            transcript = Transcript(
                words=(Word(1, 2, "Hola."), Word(48.7, 48.9, "¡No"),
                       Word(48.9, 49.1, "puede"), Word(49.1, 49.5, "ser!")),
                segments=(Segment(1, 2, "Hola."), Segment(48.7, 49.5, "¡No puede ser!")),
            )
            transcription_completed = False
            nvenc_requests = []

            def known_transcript(extracted_audio, supplied_config):
                nonlocal transcription_completed
                self.assertEqual(supplied_config, config)
                with wave.open(str(extracted_audio), "rb") as stream:
                    self.assertEqual((stream.getframerate(), stream.getnchannels()), (16000, 1))
                    self.assertAlmostEqual(stream.getnframes() / 16000, 80, delta=0.1)
                transcription_completed = True
                return transcript

            real_render_run = render._run

            def cpu_encoder_adapter(command, cwd, timeout):
                self.assertTrue(transcription_completed)
                adapted = list(command)
                if "-c:v" in adapted and adapted[adapted.index("-c:v") + 1] == "h264_nvenc":
                    nvenc_requests.append(list(adapted))
                    adapted[adapted.index("-c:v") + 1] = "libx264"
                    adapted[adapted.index("-preset") + 1] = "ultrafast"
                    for option in ("-rc", "-cq"):
                        index = adapted.index(option)
                        del adapted[index:index + 2]
                real_render_run(adapted, cwd, timeout)

            with patch.object(pipeline, "transcribe", side_effect=known_transcript) as asr, \
                 patch.object(render, "_run", side_effect=cpu_encoder_adapter):
                reports = pipeline.run_pipeline(config, [])

            asr.assert_called_once()
            self.assertEqual(len(reports), 1)
            report = reports[0]
            self.assertEqual(report["status"], "ok", report)
            self.assertFalse(report["errors"])
            self.assertEqual(len(report["clips"]), 1)
            self.assertEqual(len(nvenc_requests), 1)
            clip = report["clips"][0]
            self.assertAlmostEqual(clip["end"], 50, delta=0.2)
            self.assertTrue(35 <= (clip["end"] - clip["start"]) / config.speed <= 55)
            self.assertIn("volume_jump", clip["reasons"])
            self.assertIn("trigger:no puede ser", clip["reasons"])
            video = Path(clip["video"])
            subtitles = Path(clip["subtitles"])
            caption = Path(clip["caption"])
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in (video, subtitles, caption)))
            self.assertAlmostEqual(clip["duration"], 48, places=3)
            self.assertIn("¡No puede ser!", subtitles.read_text(encoding="utf-8"))
            self.assertIn("00:00:46,827 --> 00:00:47,596", subtitles.read_text(encoding="utf-8"))
            text = caption.read_text(encoding="utf-8")
            self.assertIn("https://t.me/local_test", text)
            self.assertEqual(sum(word.startswith("#") for word in text.split()), 5)

            analysis = json.loads((video.parent / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(len(analysis["scene_boundaries"]), 4)
            for actual, expected in zip(analysis["scene_boundaries"], (0, 30, 50, 80)):
                self.assertAlmostEqual(actual, expected, delta=0.2)
            self.assertTrue(any(abs(p["time"] - 48.7) < 0.15 for p in analysis["audio_peaks"]))
            # JSON naturally serializes Clip.reasons tuples as arrays.
            self.assertEqual(json.loads((video.parent / "report.json").read_text(encoding="utf-8")),
                             json.loads(json.dumps(report)))
            batch_reports = list(config.output_dir.glob("run_*.json"))
            self.assertEqual(len(batch_reports), 1)
            self.assertEqual(json.loads(batch_reports[0].read_text(encoding="utf-8")),
                             json.loads(json.dumps(reports)))

            info = json.loads(subprocess.run([
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video),
            ], capture_output=True, text=True, check=True, timeout=20).stdout)
            stream = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual((stream["width"], stream["height"]), (180, 320))
            self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
            self.assertNotIn("ORIGINAL_EPISODE", json.dumps(info))
            frame = subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", "1.5",
                "-i", str(video), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ], capture_output=True, check=True, timeout=20).stdout
            yellow_pixels = sum(frame[i] > 170 and frame[i + 1] > 140 and frame[i + 2] < 120
                                for i in range(0, len(frame), 3))
            self.assertGreater(yellow_pixels, 5)
            self.assertEqual(list(temp_dir.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(), "preserve this file")
            self.assertFalse(list(video.parent.glob(".render-*")))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)


if __name__ == "__main__":
    unittest.main()
