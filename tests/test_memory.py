"""Verify memory policy and sequential retry; no GPU or model is simulated as ready."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from navalrep import analysis
from navalrep.config import PipelineConfig, load_config
from navalrep.models import Transcript


class MemoryPolicyTests(unittest.TestCase):
    def test_defaults_and_example_select_memory_saving_gpu_mode(self):
        config = PipelineConfig()
        self.assertEqual((config.whisper_model, config.device, config.compute_type, config.beam_size),
                         ("large-v3", "cuda", "int8_float16", 1))
        self.assertTrue(config.cpu_fallback)
        example = load_config(Path(__file__).resolve().parents[1] / "config.example.json")
        self.assertEqual((example.device, example.compute_type, example.beam_size),
                         (config.device, config.compute_type, config.beam_size))

    def test_manual_cpu_and_strict_fp16_modes_remain_available(self):
        for compute in ("int8", "int8_float32", "float32"):
            self.assertEqual(PipelineConfig(device="cpu", compute_type=compute).compute_type, compute)
        config = PipelineConfig(compute_type="float16", beam_size=5, cpu_fallback=False)
        self.assertEqual((config.device, config.compute_type, config.beam_size, config.cpu_fallback),
                         ("cuda", "float16", 5, False))
        for kwargs in ({"device": "cpu", "compute_type": "float16"},
                       {"device": "cuda", "compute_type": "int8"},
                       {"device": "auto"}, {"cpu_fallback": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)

    def test_only_explicit_cuda_oom_is_classified_for_retry(self):
        for message in ("CUDA failed with error out of memory", "CUDA out of memory",
                        "CUDA_ERROR_OUT_OF_MEMORY", "CUDA_ERROR_MEMORY_ALLOCATION"):
            with self.subTest(message=message):
                self.assertTrue(analysis._is_cuda_oom(RuntimeError(message)))
        for error in (MemoryError("out of memory"), RuntimeError("No CUDA-capable device"),
                      RuntimeError("CUDA driver version is insufficient for CUDA runtime version"),
                      RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED"),
                      RuntimeError("Unable to open model.bin"), RuntimeError("libcudnn.so.9 missing")):
            with self.subTest(error=str(error)):
                self.assertFalse(analysis._is_cuda_oom(error))

    def test_successful_gpu_attempt_does_not_start_cpu(self):
        expected = Transcript((), (), device="cuda", compute_type="int8_float16")
        config = PipelineConfig()
        with patch.object(analysis, "_transcribe_once", return_value=expected) as once:
            self.assertIs(analysis.transcribe(Path("audio.wav"), config), expected)
        once.assert_called_once_with(Path("audio.wav"), config)

    def test_cuda_oom_retries_same_model_once_on_cpu_and_logs_change(self):
        expected = Transcript((), (), device="cpu", compute_type="int8_float32")
        config = PipelineConfig(model_dir=Path("models/large-v3"))
        with patch.object(analysis, "_transcribe_once", side_effect=[
            analysis.WhisperOutOfMemoryError("CUDA out of memory"), expected,
        ]) as once, self.assertLogs("navalrep.analysis", level="WARNING") as logs:
            self.assertIs(analysis.transcribe(Path("audio.wav"), config), expected)
        self.assertIn("system RAM", " ".join(logs.output))
        first = once.call_args_list[0].args[1]
        second = once.call_args_list[1].args[1]
        self.assertEqual(first, config)
        self.assertEqual(second, replace(config, device="cpu", compute_type="int8", cpu_fallback=False))
        self.assertEqual(second.whisper_model, "large-v3")
        self.assertEqual(second.model_dir, config.model_dir)
        self.assertEqual((config.device, config.compute_type), ("cuda", "int8_float16"))

    def test_non_oom_errors_and_disabled_or_cpu_modes_do_not_retry(self):
        cases = [(PipelineConfig(), RuntimeError("CUDA driver version is insufficient")),
                 (PipelineConfig(), FileNotFoundError("missing model")),
                 (PipelineConfig(), TimeoutError("Whisper timed out")),
                 (PipelineConfig(cpu_fallback=False), analysis.WhisperOutOfMemoryError("CUDA out of memory")),
                 (PipelineConfig(device="cpu", compute_type="int8"), analysis.WhisperOutOfMemoryError("CUDA out of memory"))]
        for config, error in cases:
            with self.subTest(error=error, device=config.device), \
                 patch.object(analysis, "_transcribe_once", side_effect=error) as once:
                with self.assertRaises(type(error)):
                    analysis.transcribe(Path("audio.wav"), config)
                self.assertEqual(once.call_count, 1)

    def test_cpu_failure_propagates_without_a_retry_loop(self):
        with patch.object(analysis, "_transcribe_once", side_effect=[
            analysis.WhisperOutOfMemoryError("CUDA out of memory"), MemoryError("RAM exhausted"),
        ]) as once, self.assertLogs("navalrep.analysis", level="WARNING"):
            with self.assertRaisesRegex(MemoryError, "RAM exhausted"):
                analysis.transcribe(Path("audio.wav"), PipelineConfig())
        self.assertEqual(once.call_count, 2)

    def test_failed_gpu_worker_is_closed_and_scratch_removed_before_cpu_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workers = []
            payloads = [
                {"error": "CUDA failed with error out of memory", "failure_kind": "cuda_oom"},
                {"words": [], "segments": [], "language": "es", "device": "cpu", "compute_type": "int8_float32"},
            ]

            class Worker:
                exitcode = 0

                def __init__(self, arguments, payload):
                    self.config = arguments[1]
                    self.result = Path(arguments[2])
                    self.payload = payload
                    self.closed = False
                    self.alive = False

                def start(self):
                    self.alive = True

                def join(self, timeout=None):
                    self.result.write_text(json.dumps(self.payload), encoding="utf-8")
                    self.alive = False

                def is_alive(self):
                    return self.alive

                def close(self):
                    self.closed = True

            def create_worker(**kwargs):
                if workers:
                    self.assertTrue(workers[0].closed)
                    self.assertFalse(workers[0].alive)
                    self.assertFalse(workers[0].result.parent.exists())
                    self.assertEqual(kwargs["args"][1].device, "cpu")
                worker = Worker(kwargs["args"], payloads[len(workers)])
                workers.append(worker)
                return worker

            context = SimpleNamespace(Process=create_worker)
            with patch.object(analysis.multiprocessing, "get_context", return_value=context), \
                 patch.object(analysis, "_cuda_library_paths", return_value=[]), \
                 self.assertLogs("navalrep.analysis", level="WARNING"):
                transcript = analysis.transcribe(root / "audio.wav", PipelineConfig())
            self.assertEqual((transcript.device, transcript.compute_type), ("cpu", "int8_float32"))
            self.assertEqual(len(workers), 2)
            self.assertTrue(all(worker.closed and not worker.alive for worker in workers))
            self.assertFalse(list(root.glob("whisper-*")))


if __name__ == "__main__":
    unittest.main()
