import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from yams.audio import budget_gpu, initial_chunk_size
from yams.__main__ import main


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GpuBudgetTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(4, 4)
        self.device = torch.device("cuda:0")

    def test_caps_allocations_and_leaves_desktop_headroom(self):
        for free, total, reserved in ((8, 10, 2), (3, 4, 1)):
            with self.subTest(total=total), patch("yams.audio.torch.cuda.get_allocator_backend", return_value="native"), \
                    patch.dict("os.environ", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "0"}), \
                    patch("yams.audio.torch.cuda.empty_cache"), \
                    patch("yams.audio.torch.cuda.mem_get_info", return_value=(free * 1024**3, total * 1024**3)), \
                    patch("yams.audio.torch.cuda.set_per_process_memory_fraction") as limit:
                budget_gpu(model=self.model, device=self.device)
                limit.assert_called_once_with(fraction=(free - reserved) / total, device=self.device)

    def test_rejects_unsafe_budget_before_setting_allocator_limit(self):
        with patch("yams.audio.torch.cuda.get_allocator_backend", return_value="native"), \
                patch.dict("os.environ", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "0"}), \
                patch("yams.audio.torch.cuda.empty_cache"), \
                patch("yams.audio.torch.cuda.mem_get_info", return_value=(1024**3, 8 * 1024**3)), \
                patch("yams.audio.torch.cuda.set_per_process_memory_fraction") as limit:
            with self.assertRaisesRegex(ValueError, "Insufficient safe GPU memory"):
                budget_gpu(model=self.model, device=self.device)
            limit.assert_not_called()

    def test_rejects_model_larger_than_budget(self):
        model = torch.nn.Module()
        model.register_parameter("weights", torch.nn.Parameter(torch.empty(1024**3, device="meta")))
        with patch("yams.audio.torch.cuda.get_allocator_backend", return_value="native"), \
                patch.dict("os.environ", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "0"}), \
                patch("yams.audio.torch.cuda.empty_cache"), \
                patch("yams.audio.torch.cuda.mem_get_info", return_value=(3 * 1024**3, 4 * 1024**3)), \
                patch("yams.audio.torch.cuda.set_per_process_memory_fraction") as limit:
            with self.assertRaisesRegex(ValueError, "Insufficient safe GPU memory"):
                budget_gpu(model=model, device=self.device)
            limit.assert_not_called()

    def test_out_of_memory_rolls_back_original_and_exits_with_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "song.ini").write_text("[song]\n")
            (folder / "notes.mid").write_bytes(b"chart")
            (folder / "song.ogg").write_bytes(b"original")
            output = io.StringIO()
            arguments = ["yams", str(folder), "--models", temporary, "--inference-source", temporary]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(output), \
                    patch("yams.audio.separate", side_effect=torch.cuda.OutOfMemoryError("safe memory limit")):
                status = main()
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["event"], "error")
            self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
            self.assertFalse((folder / ".yams-work").exists())
            self.assertFalse((folder / "backup/song.ogg").exists())

    def test_rejects_allocator_that_cannot_enforce_protection(self):
        with patch("yams.audio.torch.cuda.get_allocator_backend", return_value="cudaMallocAsync"), \
                patch("yams.audio.torch.cuda.mem_get_info") as memory:
            with self.assertRaisesRegex(ValueError, "native caching allocator"):
                budget_gpu(model=self.model, device=self.device)
            memory.assert_not_called()

    def test_initial_chunk_size_selection(self):
        with patch("yams.audio.torch.cuda.empty_cache"), \
                patch("yams.audio.torch.cuda.mem_get_info", return_value=(7 * 1024**3, 8 * 1024**3)):
            self.assertEqual(initial_chunk_size(torch.device("cuda:0")), 882000)
        with patch("yams.audio.torch.cuda.empty_cache"), \
                patch("yams.audio.torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
            self.assertEqual(initial_chunk_size(torch.device("cuda:0")), 441000)
        with patch("yams.audio.torch.cuda.empty_cache"), \
                patch("yams.audio.torch.cuda.mem_get_info", return_value=(2 * 1024**3, 8 * 1024**3)):
            self.assertEqual(initial_chunk_size(torch.device("cuda:0")), 220500)
        self.assertEqual(initial_chunk_size(torch.device("mps")), 441000)
        self.assertEqual(initial_chunk_size(torch.device("cpu")), 441000)

    def test_oom_retries_with_halved_chunk_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            stage = temp / "stage"
            stage.mkdir()
            models = temp / "models"
            models.mkdir()
            (models / "BS-Rofo-SW-Fixed.yaml").write_text("config")
            (models / "BS-Rofo-SW-Fixed.ckpt").write_text("weights")
            import numpy as np
            from types import SimpleNamespace
            from yams.audio import STEM_MAPPING_6STEM, run_torch_6stem_pcm
            reference = np.zeros((1000, 2), dtype=np.float32)

            observed_chunks = []

            def fake_demix(config, model, mix, device, model_type):
                observed_chunks.append(config.inference.chunk_size)
                if len(observed_chunks) == 1:
                    raise torch.cuda.OutOfMemoryError("out of memory")
                return {name: np.zeros((2, 1000), dtype=np.float32) for name in STEM_MAPPING_6STEM}

            fake_model = unittest.mock.MagicMock()
            fake_model.register_forward_hook.return_value = unittest.mock.MagicMock()
            fake_model.to.return_value.eval.return_value = fake_model

            config = SimpleNamespace(
                model=SimpleNamespace(num_stems=6),
                audio=SimpleNamespace(sample_rate=44100),
                training=SimpleNamespace(use_amp=False),
                inference=SimpleNamespace(normalize=False, num_overlap=2, batch_size=1),
            )
            settings = unittest.mock.MagicMock()
            settings.get_model_from_config.return_value = (fake_model, config)
            model_utils = unittest.mock.MagicMock()
            model_utils.demix.side_effect = fake_demix

            def copy_audio(source, destination, rate):
                import shutil
                shutil.copyfile(source, destination)

            with patch.object(torch, "load", return_value={}), \
                    patch("yams.audio.initial_chunk_size", return_value=882000), \
                    patch("yams.audio.convert", side_effect=copy_audio):
                run_torch_6stem_pcm(stage, models, reference, 44100, settings, model_utils,
                                    torch.device("cpu"), lambda *a, **kw: None, None)

            self.assertEqual(observed_chunks[0], 882000)
            self.assertEqual(observed_chunks[1], 441000)

    def test_oom_exhaustion_raises_safe_memory_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            stage = temp / "stage"
            stage.mkdir()
            models = temp / "models"
            models.mkdir()
            (models / "BS-Rofo-SW-Fixed.yaml").write_text("config")
            (models / "BS-Rofo-SW-Fixed.ckpt").write_text("weights")
            import numpy as np
            from types import SimpleNamespace
            from yams.audio import run_torch_6stem_pcm
            reference = np.zeros((1000, 2), dtype=np.float32)

            def always_oom(config, model, mix, device, model_type):
                raise torch.cuda.OutOfMemoryError("out of memory")

            fake_model = unittest.mock.MagicMock()
            fake_model.register_forward_hook.return_value = unittest.mock.MagicMock()
            fake_model.to.return_value.eval.return_value = fake_model

            config = SimpleNamespace(
                model=SimpleNamespace(num_stems=6),
                audio=SimpleNamespace(sample_rate=44100),
                training=SimpleNamespace(use_amp=False),
                inference=SimpleNamespace(normalize=False, num_overlap=2, batch_size=1),
            )
            settings = unittest.mock.MagicMock()
            settings.get_model_from_config.return_value = (fake_model, config)
            model_utils = unittest.mock.MagicMock()
            model_utils.demix.side_effect = always_oom

            def copy_audio(source, destination, rate):
                import shutil
                shutil.copyfile(source, destination)

            with patch.object(torch, "load", return_value={}), \
                    patch("yams.audio.initial_chunk_size", return_value=220500), \
                    patch("yams.audio.convert", side_effect=copy_audio):
                with self.assertRaisesRegex(ValueError, "safe GPU memory limit"):
                    run_torch_6stem_pcm(stage, models, reference, 44100, settings, model_utils,
                                        torch.device("cpu"), lambda *a, **kw: None, None)

    def test_adapts_headroom_when_memory_constrained(self):
        with patch("yams.audio.torch.cuda.get_allocator_backend", return_value="native"), \
                patch.dict("os.environ", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "0"}), \
                patch("yams.audio.torch.cuda.empty_cache"), \
                patch("yams.audio.torch.cuda.mem_get_info", return_value=(int(2.5 * 1024**3), 10 * 1024**3)), \
                patch("yams.audio.torch.cuda.set_per_process_memory_fraction") as limit:
            budget_gpu(model=self.model, device=self.device)
            fraction = limit.call_args.kwargs["fraction"]
            self.assertGreater(fraction * 10, 1.4)
            self.assertLessEqual(fraction * 10, 1.6)

    def test_accumulation_device_selection(self):
        from yams.inference.utils.model_utils import _get_accumulation_device
        self.assertEqual(_get_accumulation_device("cpu", 100), torch.device("cpu"))
        with patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 10 * 1024**3)), \
                patch("torch.cuda.memory_reserved", return_value=0), \
                patch("torch.cuda.memory_allocated", return_value=0), \
                patch("torch.cuda.get_per_process_memory_fraction", return_value=1.0):
            self.assertEqual(_get_accumulation_device("cuda:0", 500 * 1024**2), torch.device("cuda:0"))
            self.assertEqual(_get_accumulation_device("cuda:0", 3 * 1024**3), torch.device("cpu"))
        with patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 10 * 1024**3)), \
                patch("torch.cuda.memory_reserved", return_value=0), \
                patch("torch.cuda.memory_allocated", return_value=500 * 1024**2), \
                patch("torch.cuda.get_per_process_memory_fraction", return_value=0.25):
            self.assertEqual(_get_accumulation_device("cuda:0", 600 * 1024**2), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
