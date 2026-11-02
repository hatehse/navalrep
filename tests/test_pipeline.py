import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from navalrep.config import PipelineConfig, load_config
from navalrep.models import Clip, MediaInfo, Segment, Transcript
from navalrep.pipeline import caption_for, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_custom_executables_are_resolved_before_render_changes_directory(self):
        config = PipelineConfig(ffmpeg="tools/ffmpeg", ffprobe="tools/ffprobe")
        self.assertTrue(Path(config.ffmpeg).is_absolute())
        self.assertTrue(Path(config.ffprobe).is_absolute())

    def test_explicit_missing_input_is_reported_and_next_file_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.mp4"
            valid.touch()
            missing = root / "missing.mp4"
            config = PipelineConfig(output_dir=root / "output", temp_dir=root / "temp")
            with patch("navalrep.pipeline.process_episode", return_value={"status": "ok", "clips": []}) as process, self.assertLogs("navalrep.pipeline"):
                reports = run_pipeline(config, [missing, valid])
            self.assertEqual([r["status"] for r in reports], ["failed", "ok"])
            self.assertIn("not a file", reports[0]["errors"][0]["error"])
            process.assert_called_once_with(valid, config)

    def test_validation_rejects_invalid_limits_and_unknown_keys(self):
        for kwargs in ({"min_duration": 34.0}, {"max_duration": 56.0}, {"speed": 0.0},
                       {"width": 1920}, {"device": "cpu"}, {"threads": 0}, {"mirror": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"typo": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path)

    def test_spanish_caption_with_five_tags_and_telegram(self):
        config = PipelineConfig(telegram_url="https://t.me/example")
        text = caption_for("clip_01", config)
        self.assertIn("https://t.me/example", text)
        self.assertEqual(sum(word.startswith("#") for word in text.split()), 5)
        self.assertEqual(text, caption_for("clip_01", config))

    def test_file_failure_continues_and_cleans_only_owned_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            for name in ("bad.mp4", "good.mp4"):
                (source / name).touch()
            temp = root / "temp"
            temp.mkdir()
            preserved = temp / "user-file.txt"
            preserved.write_text("keep", encoding="utf-8")
            config = PipelineConfig(input_dir=source, output_dir=root / "output", temp_dir=temp)

            def scenes(path, config):
                if path.name == "bad.mp4":
                    raise RuntimeError("injected detector failure")
                return [0, 60]

            with patch("navalrep.pipeline.probe_media", return_value=MediaInfo(60, 1280, 720, True)), \
                 patch("navalrep.pipeline.detect_scenes", side_effect=scenes), \
                 patch("navalrep.pipeline.extract_audio"), \
                 patch("navalrep.pipeline.transcribe", return_value=Transcript((), ())), \
                 patch("navalrep.pipeline.analyze_audio", return_value=[]), \
                 patch("navalrep.pipeline.select_clips", return_value=[]), \
                 self.assertLogs("navalrep.pipeline"):
                reports = run_pipeline(config, [])
            self.assertEqual([r["status"] for r in reports], ["failed", "no_candidates"])
            self.assertEqual(list(temp.iterdir()), [preserved])
            self.assertEqual(preserved.read_text(), "keep")
            self.assertEqual(len(list(config.output_dir.glob("*/report.json"))), 2)
            self.assertEqual(len(list(config.output_dir.glob("run_*.json"))), 1)

    def test_clip_failure_continues_to_next_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "episode.mp4"
            episode.touch()
            config = PipelineConfig(output_dir=root / "output", temp_dir=root / "temp")
            clips = [Clip(0, 46.8, 4, ()), Clip(60, 106.8, 5, ())]
            with patch("navalrep.pipeline.probe_media", return_value=MediaInfo(120, 1280, 720, True)), \
                 patch("navalrep.pipeline.detect_scenes", return_value=[0, 60, 120]), \
                 patch("navalrep.pipeline.extract_audio"), \
                 patch("navalrep.pipeline.transcribe", return_value=Transcript((), (Segment(0, 120, "Hola"),))), \
                 patch("navalrep.pipeline.analyze_audio", return_value=[]), \
                 patch("navalrep.pipeline.select_clips", return_value=clips), \
                 patch("navalrep.pipeline.render_clip", side_effect=[RuntimeError("NVENC error"), {"duration": 45.0}]) as render, \
                 self.assertLogs("navalrep.pipeline"):
                reports = run_pipeline(config, [episode])
            self.assertEqual(render.call_count, 2)
            self.assertEqual(reports[0]["status"], "partial")
            self.assertEqual(len(reports[0]["clips"]), 1)
            self.assertFalse(list(config.temp_dir.iterdir()))
            saved = next(config.output_dir.glob("*/report.json"))
            self.assertEqual(json.loads(saved.read_text())["status"], "partial")


if __name__ == "__main__":
    unittest.main()
