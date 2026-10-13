import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf

from yams import models, folders
from yams.__main__ import main
from yams.folders import OUTPUTS_6STEM, digest, begin, prepare, install, recover


class MLXAssetTests(unittest.TestCase):
    def test_registry_contains_pinned_mlx_assets(self):
        self.assertIn("mlx_files", models.ASSETS)
        mlx_files = models.ASSETS["mlx_files"]
        self.assertIn("bs_roformer_sw_config.json", mlx_files)
        self.assertIn("bs_roformer_sw.safetensors", mlx_files)
        for name, asset in mlx_files.items():
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertGreater(asset["bytes"], 0)
            self.assertTrue(asset["url"].startswith("https://huggingface.co/"))

    def test_default_models_returns_profile_cache_path(self):
        path = models.default_models(profile="mlx-6stem")
        path_str = str(path)
        self.assertTrue(path_str.endswith("mlx-6stem"))

    def test_load_mlx_model_strictly_loads_weights(self):
        if sys.platform != "darwin":
            return
        cache = models.default_models(profile="mlx-6stem")
        config_path = cache / "bs_roformer_sw_config.json"
        weights_path = cache / "bs_roformer_sw.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            return
        from yams.mlx_audio import load_mlx_model
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        model = load_mlx_model(weights_path, cfg)
        self.assertIsNotNone(model)


class MLXFolderLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_prepare_and_install_6stem_outputs(self):
        (self.folder / "song.ini").write_text("[song]\n")
        (self.folder / "notes.mid").write_bytes(b"chart")
        source = self.folder / "song.wav"
        sr = 44100
        audio = np.zeros((sr, 2), dtype=np.float32)
        sf.write(source, audio, sr, format="WAV", subtype="FLOAT")

        stage, record = begin(self.folder, source)
        files_report = {}
        for name in OUTPUTS_6STEM:
            path = stage / "outputs" / name
            path.write_bytes(name.encode())
            files_report[name] = {"sha256": digest(path), "peak": 0.0, "samples_above_full_scale": 0}

        audio_report = {
            "model_profile": "mlx-6stem",
            "validation": {"peak_error": 0.0, "rms_error": 0.0, "files": files_report},
        }
        prepare(stage, record, audio_report)
        install(self.folder, stage, record)

        manifest = json.loads((self.folder / "backup" / "separation-manifest.json").read_text())
        self.assertEqual(manifest["model_profile"], "mlx-6stem")
        self.assertEqual(set(manifest["outputs"].keys()), set(OUTPUTS_6STEM))
        self.assertEqual({p.name for p in self.folder.glob("*.opus")}, set(OUTPUTS_6STEM))

    def test_recover_rolls_back_6stem_staging(self):
        stage = self.folder / ".yams-work"
        stage.mkdir()
        (stage / "removed-original").write_bytes(b"original audio")
        (self.folder / "backup").mkdir()
        (stage / "original").write_bytes(b"original audio")
        os.link(stage / "original", self.folder / "backup" / "song.wav")

        outputs = {}
        (stage / "outputs").mkdir()
        for name in OUTPUTS_6STEM:
            out = stage / "outputs" / name
            out.write_bytes(f"{name} content".encode())
            outputs[name] = digest(out)
            os.link(out, self.folder / name)

        (stage / "manifest.json").write_text("manifest")
        os.link(stage / "manifest.json", self.folder / "backup" / "separation-manifest.json")

        record = {
            "version": 1,
            "state": "ready",
            "source": "song.wav",
            "source_sha256": digest(stage / "removed-original"),
            "backup_existed": False,
            "outputs": outputs,
            "manifest_sha256": digest(stage / "manifest.json"),
        }
        (stage / "job.json").write_text(json.dumps(record))

        res = recover(self.folder)
        self.assertEqual(res, "rolled_back")
        self.assertEqual((self.folder / "song.wav").read_bytes(), b"original audio")
        for name in OUTPUTS_6STEM[:-1]:
            self.assertFalse((self.folder / name).exists())


class MLXCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_mlx_profile_rejected_on_unsupported_platform(self):
        (self.folder / "song.ini").write_text("[song]\n")
        (self.folder / "notes.mid").write_bytes(b"chart")
        (self.folder / "song.ogg").write_bytes(b"audio")
        output = io.StringIO()
        args = ["yams", str(self.folder), "--model-profile", "mlx-6stem"]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(output), \
                patch.object(sys, "platform", "win32"):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_mlx_profile_accepted_on_con_package(self):
        try:
            from test_confile import create_synthetic_con
        except ImportError:
            from tests.test_confile import create_synthetic_con
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp)
            sibling = Path(tmp) / "testsong"
            output = io.StringIO()
            captured_profile = None

            def fake_separate(stage, models, model_profile="mega-8stem", **kwargs):
                nonlocal captured_profile
                captured_profile = model_profile
                outputs_dir = stage / "pcm"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                for name in ("drums.wav", "bass.wav", "guitar.wav", "vocals.wav", "keys.wav", "song.wav"):
                    stem = outputs_dir / name
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-ac", "2", str(stem)
                    ], check=True, capture_output=True)
                return {"pcm_validation": {"peak_error": 0.0}, "validation": {"codec": "opus"}}

            args = ["yams", str(con_path), "--models", str(con_path), "--model-profile", "mlx-6stem"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(output), \
                    patch.object(sys, "platform", "darwin"), patch("platform.machine", return_value="arm64"), \
                    patch("yams.audio.separate", side_effect=fake_separate):
                status = main()
            self.assertEqual(status, 0)
            self.assertEqual(captured_profile, "mlx-6stem")
            dta_text = (sibling / "songs/songs.dta").read_text()
            self.assertIn("drum (0 1)", dta_text)
            self.assertIn("bass (2 3)", dta_text)
