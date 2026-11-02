"""Subtitle tests and an explicitly CPU-only FFmpeg filter integration test.

Production always requests h264_nvenc. The integration adapter below intercepts
that exact encoder option to exercise filters on CI machines without NVIDIA;
these tests do not establish CUDA or NVENC readiness.
"""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from navalrep.config import PipelineConfig
from navalrep.models import Clip, Segment, Transcript, Word
from navalrep import render


def parse_srt(path):
    def milliseconds(stamp):
        hours, minutes, rest = stamp.split(":")
        seconds, millis = rest.split(",")
        return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    result = []
    for block in text.split("\n\n"):
        identifier, timing, content = block.split("\n", 2)
        start, end = timing.split(" --> ")
        result.append((int(identifier), milliseconds(start), milliseconds(end), content))
    return result


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "subtitles.srt"
        self.config = PipelineConfig()

    def test_spanish_words_are_grouped_and_times_are_speed_adjusted(self):
        transcript = Transcript((
            Word(10.0, 10.26, "¿Quién"),
            Word(10.26, 11.04, "es?"),
            Word(11.04, 11.3, "¡No"),
            Word(11.3, 11.6, "puede"),
            Word(11.6, 12.08, "ser!"),
        ), ())
        count = render.write_srt(transcript, Clip(10, 15.2, 3, ()), self.path, self.config)
        self.assertEqual(count, 2)
        self.assertEqual(parse_srt(self.path), [
            (1, 0, 1000, "¿Quién es?"),
            (2, 1000, 2000, "¡No puede ser!"),
        ])

    def test_word_count_limits_grouping_without_losing_unicode(self):
        words = tuple(Word(10 + i * .26, 10 + (i + 1) * .26, text)
                      for i, text in enumerate(("¡", "Qué", "pasó", "aquí", "!")))
        count = render.write_srt(
            Transcript(words, ()), Clip(10, 15.2, 3, ()), self.path,
            replace(self.config, subtitle_words=2),
        )
        cues = parse_srt(self.path)
        self.assertEqual(count, 3)
        self.assertEqual([cue[3] for cue in cues], ["¡Qué", "pasó aquí", "!"])
        self.assertTrue(all(a[2] <= b[1] for a, b in zip(cues, cues[1:])))

    def test_segment_fallback_clamps_boundaries_and_removes_overlap(self):
        transcript = Transcript((), (
            Segment(1, 2, "outside"),
            Segment(9, 12.08, "¡No!"),
            Segment(11.04, 15.2, "¿Quién es?"),
            Segment(14, 17, "fully overlapped"),
            Segment(16, 17, "outside again"),
        ))
        count = render.write_srt(transcript, Clip(10, 15.2, 3, ()), self.path, self.config)
        self.assertEqual(count, 2)
        self.assertEqual(parse_srt(self.path), [
            (1, 0, 2000, "¡No!"),
            (2, 2000, 5000, "¿Quién es?"),
        ])

    def test_clipped_words_stay_inside_output_and_do_not_overlap(self):
        transcript = Transcript((
            Word(9, 11.04, "¡No!"),
            Word(10.52, 12.08, "¿Qué?"),
            Word(14.16, 16, "¡Mira!"),
        ), ())
        render.write_srt(transcript, Clip(10, 15.2, 3, ()), self.path, self.config)
        self.assertEqual(parse_srt(self.path), [
            (1, 0, 1000, "¡No!"),
            (2, 1000, 2000, "¿Qué?"),
            (3, 4000, 5000, "¡Mira!"),
        ])

    def test_empty_transcript_produces_empty_file(self):
        self.assertEqual(render.write_srt(
            Transcript((), ()), Clip(0, 40, 0, ()), self.path, self.config,
        ), 0)
        self.assertEqual(self.path.read_bytes(), b"")


class RenderFailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.source = self.work / "source.mp4"
        self.source.write_bytes(b"source placeholder; subprocess is mocked")
        self.output = self.work / "output"
        self.clip = Clip(0, 36.4, 3, ())
        self.config = PipelineConfig()
        self.transcript = Transcript((), ())

    def test_encode_failure_cleans_staging_files(self):
        def fail_after_partial_encode(command, cwd, timeout):
            (cwd / "render.mp4").write_bytes(b"partial encoded data")
            raise RuntimeError("injected encoder failure")

        with patch.object(render, "_run", side_effect=fail_after_partial_encode):
            with self.assertRaisesRegex(RuntimeError, "injected encoder failure"):
                render.render_clip(self.source, self.clip, self.transcript,
                                   self.output, "clip", self.config)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_failed_publication_removes_partial_new_file(self):
        def fake_encode(command, cwd, timeout):
            (cwd / "render.mp4").write_bytes(b"encoded placeholder")

        def fail_copy(source, destination):
            destination.write(b"partial publication")
            raise OSError("injected disk failure")

        with patch.object(render, "_run", side_effect=fake_encode), \
             patch.object(render, "_duration", return_value=35.0), \
             patch.object(render.shutil, "copyfileobj", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "injected disk failure"):
                render.render_clip(self.source, self.clip, self.transcript,
                                   self.output, "clip", self.config)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_artifacts_are_not_overwritten(self):
        self.output.mkdir()
        for name in ("clip.mp4", "clip.srt"):
            with self.subTest(name=name):
                existing = self.output / name
                existing.write_bytes(b"keep original")
                with patch.object(render, "_run") as run:
                    with self.assertRaises(FileExistsError):
                        render.render_clip(self.source, self.clip, self.transcript,
                                           self.output, "clip", self.config)
                    run.assert_not_called()
                self.assertEqual(existing.read_bytes(), b"keep original")
                self.assertEqual(list(self.output.iterdir()), [existing])
                existing.unlink()

    def test_encoded_duration_outside_limits_is_rejected_and_cleaned(self):
        def fake_encode(command, cwd, timeout):
            (cwd / "render.mp4").write_bytes(b"encoded placeholder")

        for duration in (34.999, 55.001):
            with self.subTest(duration=duration):
                with patch.object(render, "_run", side_effect=fake_encode), \
                     patch.object(render, "_duration", return_value=duration):
                    with self.assertRaisesRegex(RuntimeError, "Encoded duration"):
                        render.render_clip(self.source, self.clip, self.transcript,
                                           self.output, "clip", self.config)
                self.assertEqual(list(self.output.iterdir()), [])


class EncodedVideoValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig()
        self.media = {
            "format": {"duration": "40.000000"},
            "streams": [{"width": 1080, "height": 1920,
                         "duration": "40.000000", "sample_aspect_ratio": "1:1"}],
        }

    def validate(self):
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(self.media), stderr="")
        with patch.object(render.subprocess, "run", return_value=result) as run:
            value = render._duration(Path("encoded.mp4"), self.config)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-select_streams") + 1], "V:0")
            return value

    def test_valid_container_cannot_hide_a_short_video_stream(self):
        self.media["streams"][0]["duration"] = "10.000000"
        with self.assertRaisesRegex(RuntimeError, "video stream duration"):
            self.validate()

    def test_video_dimensions_and_square_pixels_are_required(self):
        for field, value in (("width", 1920), ("height", 1080),
                             ("sample_aspect_ratio", "2:1")):
            with self.subTest(field=field):
                original = self.media["streams"][0][field]
                self.media["streams"][0][field] = value
                with self.assertRaises(RuntimeError):
                    self.validate()
                self.media["streams"][0][field] = original

    def test_missing_video_duration_is_not_treated_as_ready(self):
        del self.media["streams"][0]["duration"]
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            self.validate()

    def test_primary_video_stream_and_container_are_accepted(self):
        self.assertEqual(self.validate(), 40.0)


class CPUFFmpegFilterIntegrationTests(unittest.TestCase):
    """Real filter/MP4 test with an explicit, test-only CPU encoder adapter."""

    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("FFmpeg and FFprobe are required for CPU integration")
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
            text=True, check=True, timeout=15,
        ).stdout
        filters = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
            text=True, check=True, timeout=15,
        ).stdout
        if "libx264" not in encoders or "subtitles" not in filters:
            raise unittest.SkipTest("CPU integration requires libx264 and libass subtitles")

    def test_cpu_adapter_renders_subtitles_and_exact_duration_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "episode with an apostrophe's.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-f", "lavfi", "-i", "color=c=blue:size=64x64:rate=2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
                "-t", "60", "-c:v", "libx264", "-preset", "ultrafast",
                "-threads", "1", "-c:a", "aac", "-metadata", "title=ORIGINAL_SOURCE",
                str(source),
            ], capture_output=True, check=True, timeout=45)
            config = PipelineConfig(width=180, height=320, fps=2, font_size=12, threads=1)
            transcript = Transcript((Word(1, 3, "¡No!"),), ())
            production_commands = []
            real_run = render._run

            def cpu_encoder_adapter(command, cwd, timeout):
                command = list(command)
                # Intercept only the requested video encoder, never a filename
                # or another occurrence of the string. No production fallback.
                if "-c:v" in command and command[command.index("-c:v") + 1] == "h264_nvenc":
                    production_commands.append(list(command))
                    command[command.index("-c:v") + 1] = "libx264"
                    command[command.index("-preset") + 1] = "ultrafast"
                    for option in ("-rc", "-cq"):
                        index = command.index(option)
                        del command[index:index + 2]
                real_run(command, cwd, timeout)

            for seconds in (35.0, 55.0):
                with self.subTest(seconds=seconds):
                    clip = Clip(1, 1 + seconds * config.speed, 3, ())
                    with patch.object(render, "_run", side_effect=cpu_encoder_adapter):
                        artifacts = render.render_clip(
                            source, clip, transcript, work / "output with spaces",
                            f"clip_{int(seconds)}", config,
                        )
                    self.assertEqual(artifacts["duration"], seconds)
                    video = Path(artifacts["video"])
                    srt = Path(artifacts["subtitles"])
                    self.assertEqual(parse_srt(srt), [(1, 0, 1923, "¡No!")])
                    probe = subprocess.run([
                        "ffprobe", "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", str(video),
                    ], capture_output=True, check=True, text=True, timeout=15)
                    info = json.loads(probe.stdout)
                    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
                    self.assertEqual((video_stream["width"], video_stream["height"]), (180, 320))
                    self.assertEqual(video_stream["codec_name"], "h264")
                    self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
                    self.assertNotIn("ORIGINAL_SOURCE", probe.stdout)
                    # Solid blue input has no yellow: yellow pixels demonstrate
                    # that libass actually burned the caption into the frame.
                    frame = subprocess.run([
                        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-ss", "0.5", "-i", str(video), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                    ], capture_output=True, check=True, timeout=15).stdout
                    yellow = [(i // 3 % config.width, i // 3 // config.width)
                              for i in range(0, len(frame), 3)
                              if frame[i] > 170 and frame[i + 1] > 140 and frame[i + 2] < 120]
                    self.assertGreater(len(yellow), 5)
                    center_x = (min(x for x, _ in yellow) + max(x for x, _ in yellow)) / 2
                    center_y = (min(y for _, y in yellow) + max(y for _, y in yellow)) / 2
                    self.assertAlmostEqual(center_x, config.width / 2, delta=config.width * .05)
                    self.assertAlmostEqual(center_y, config.height / 2, delta=config.height * .05)
                    self.assertFalse(any(p.name.startswith(".render-") for p in video.parent.iterdir()))
            self.assertEqual(len(production_commands), 2)
            for command in production_commands:
                self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc")

    def test_subtitles_are_centered_on_full_hd_vertical_canvas(self):
        """Check actual glyph position, not merely the presence of yellow text."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "blue-source.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=1:d=40",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000:duration=40",
                "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
                "-c:a", "aac", str(source),
            ], capture_output=True, check=True, timeout=45)
            config = PipelineConfig(fps=1, threads=1)
            real_run = render._run
            requested_encoders = []

            def cpu_encoder_adapter(command, cwd, timeout):
                adapted = list(command)
                if "-c:v" in adapted:
                    requested_encoders.append(adapted[adapted.index("-c:v") + 1])
                    self.assertEqual(requested_encoders[-1], "h264_nvenc")
                    adapted[adapted.index("-c:v") + 1] = "libx264"
                    adapted[adapted.index("-preset") + 1] = "ultrafast"
                    for option in ("-rc", "-cq"):
                        index = adapted.index(option)
                        del adapted[index:index + 2]
                real_run(adapted, cwd, timeout)

            with patch.object(render, "_run", side_effect=cpu_encoder_adapter):
                artifacts = render.render_clip(
                    source, Clip(0, 36.4, 3, ()), Transcript((Word(0, 2, "¡No!"),), ()),
                    work / "output", "centered", config,
                )
            self.assertEqual(requested_encoders, ["h264_nvenc"])
            frame = subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", "0.5", "-i", artifacts["video"], "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ], capture_output=True, check=True, timeout=20).stdout
            self.assertEqual(len(frame), 1080 * 1920 * 3)
            yellow = [(i // 3 % 1080, i // 3 // 1080)
                      for i in range(0, len(frame), 3)
                      if frame[i] > 170 and frame[i + 1] > 140 and frame[i + 2] < 120]
            self.assertGreater(len(yellow), 100)
            center_x = (min(x for x, _ in yellow) + max(x for x, _ in yellow)) / 2
            center_y = (min(y for _, y in yellow) + max(y for _, y in yellow)) / 2
            self.assertAlmostEqual(center_x, 540, delta=20)
            self.assertAlmostEqual(center_y, 960, delta=20)


if __name__ == "__main__":
    unittest.main()
