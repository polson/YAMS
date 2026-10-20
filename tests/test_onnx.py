import io
import math
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch

from yams import audio
from yams.folders import digest
from yams.onnx_audio import (
    ONNX_STEM_MAPPING,
    _is_out_of_memory,
    _resolve_providers,
    run_onnx_6stem_pcm,
    separate_onnx,
)


class OnnxAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.models = self.root / "models"
        self.models.mkdir()

        # Create dummy reference file
        self.rate = 44100
        self.duration_seconds = 4.5
        samples = int(self.rate * self.duration_seconds)
        t = np.linspace(0, self.duration_seconds, samples, endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)
        self.reference = np.column_stack([tone, tone])
        sf.write(self.stage / "original", self.reference, self.rate, format="WAV", subtype="FLOAT")

        # Create dummy model file
        (self.models / "bs_roformer_sw_6stem_fp16.onnx").write_bytes(b"dummy onnx weights")

    def test_provider_resolution(self):
        with patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            providers = _resolve_providers("dml:0")
            self.assertEqual(providers[0], ("DmlExecutionProvider", {"device_id": 0}))
            self.assertEqual(providers[1], "CPUExecutionProvider")

        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            providers = _resolve_providers("dml:0")
            self.assertEqual(providers, ["CPUExecutionProvider"])

        with patch("onnxruntime.get_available_providers", return_value=["ROCMExecutionProvider", "CPUExecutionProvider"]):
            with patch("sys.platform", "linux"):
                providers = _resolve_providers("cuda:0")
                self.assertIn("ROCMExecutionProvider", providers)

    def test_is_out_of_memory_detection(self):
        self.assertTrue(_is_out_of_memory(RuntimeError("DML allocator out of memory")))
        self.assertTrue(_is_out_of_memory(RuntimeError("DXGI_ERROR_DEVICE_RESET")))
        self.assertTrue(_is_out_of_memory(RuntimeError("E_OUTOFMEMORY: failed to allocate")))
        self.assertTrue(_is_out_of_memory(RuntimeError("hipErrorMemoryAllocation: out of memory")))
        self.assertTrue(_is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory")))
        self.assertFalse(_is_out_of_memory(ValueError("Invalid shape")))

    def test_stft_istft_numerical_fidelity(self):
        chunk_len = 176400
        audio_chunk = torch.randn(2, chunk_len)
        window = torch.hann_window(2048)
        spec = torch.stft(
            audio_chunk,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=window,
            center=True,
            return_complex=True,
        )
        recovered = torch.istft(
            spec,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=window,
            center=True,
            length=chunk_len,
        )
        max_error = (audio_chunk - recovered).abs().max().item()
        self.assertLess(max_error, 1e-5)

    def test_separate_onnx_full_pipeline(self):
        events = []

        def emit(event, **kwargs):
            events.append((event, kwargs))

        # Mock ONNX InferenceSession so test runs without needing GPU or large model download
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["DmlExecutionProvider"]

        def fake_run(output_names, input_feed):
            spec_real = input_feed["spec_real"]  # [1, 2, 1025, 345]
            # Output [1, 6, 2, 1025, 345] - divide input energy across 6 stems
            out_real = np.repeat(spec_real[:, None, ...] / 6.0, 6, axis=1)
            out_imag = np.zeros_like(out_real)
            return [out_real, out_imag]

        mock_session.run.side_effect = fake_run

        with patch("onnxruntime.InferenceSession", return_value=mock_session), \
             patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            result = separate_onnx(
                stage=self.stage,
                models=self.models,
                emit=emit,
                device_name="dml:0",
            )

        self.assertEqual(result["model_profile"], "onnx-6stem")
        self.assertEqual(result["model_type"], "bs_roformer_onnx")
        self.assertEqual(result["device"], "DmlExecutionProvider")
        self.assertTrue((self.stage / "pcm/song.wav").is_file())
        self.assertTrue((self.stage / "pcm/drums.wav").is_file())
        self.assertTrue((self.stage / "pcm/bass.wav").is_file())
        self.assertTrue((self.stage / "pcm/guitar.wav").is_file())
        self.assertTrue((self.stage / "pcm/keys.wav").is_file())
        self.assertTrue((self.stage / "pcm/vocals.wav").is_file())

        event_names = [e[0] for e in events]
        self.assertIn("decoded", event_names)
        self.assertIn("model_started", event_names)
        self.assertIn("model_progress", event_names)
        self.assertIn("model_finished", event_names)
        self.assertIn("validated", event_names)

    def test_onnx_directml_oom_fallback_to_cpu(self):
        events = []

        def emit(event, **kwargs):
            events.append((event, kwargs))

        gpu_session = MagicMock()
        gpu_session.get_providers.return_value = ["DmlExecutionProvider"]
        gpu_session.run.side_effect = RuntimeError("DML allocator out of memory")

        cpu_session = MagicMock()
        cpu_session.get_providers.return_value = ["CPUExecutionProvider"]

        def fake_cpu_run(output_names, input_feed):
            spec_real = input_feed["spec_real"]
            out_real = np.repeat(spec_real[:, None, ...] / 6.0, 6, axis=1)
            return [out_real, np.zeros_like(out_real)]

        cpu_session.run.side_effect = fake_cpu_run

        sessions = [gpu_session, cpu_session]

        def get_session(*args, **kwargs):
            return sessions.pop(0)

        with patch("onnxruntime.InferenceSession", side_effect=get_session), \
             patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            assets, elapsed, n_hops, active_device = run_onnx_6stem_pcm(
                stage=self.stage,
                models=self.models,
                reference=self.reference,
                rate=self.rate,
                emit=emit,
                device="dml:0",
            )

        retry_events = [e for e in events if e[0] == "model_retry"]
        self.assertEqual(len(retry_events), 1)
        self.assertIn("falling back gracefully to CPU", retry_events[0][1]["detail"])
        self.assertIn("CPUExecutionProvider", active_device)

    def test_onnx_cancellation(self):
        cancelled = threading.Event()
        cancelled.set()

        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaises(KeyboardInterrupt):
                run_onnx_6stem_pcm(
                    stage=self.stage,
                    models=self.models,
                    reference=self.reference,
                    rate=self.rate,
                    cancelled=cancelled,
                )
