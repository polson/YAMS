import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yams import models
from yams.__main__ import main


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.data = b"verified model"
        self.asset = {"url": "https://example.test/model", "bytes": len(self.data),
                      "sha256": hashlib.sha256(self.data).hexdigest()}
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def test_download_verifies_then_publishes_and_reuses_offline(self):
        registry = {"torch_6stem_files": {"BS-Rofo-SW-Fixed.ckpt": self.asset}}
        with patch.object(models, "ASSETS", registry), patch("yams.models.urllib.request.urlopen", return_value=io.BytesIO(self.data)) as network:
            models.ensure_models(folder=self.folder, emit=self.emit, profile="torch-6stem")
            network.assert_called_once()
        self.assertEqual((self.folder / "BS-Rofo-SW-Fixed.ckpt").read_bytes(), self.data)
        self.assertEqual(list(self.folder.glob("*.part")), [])
        with patch.object(models, "ASSETS", registry), patch("yams.models.urllib.request.urlopen") as network:
            models.ensure_models(folder=self.folder, emit=self.emit, profile="torch-6stem")
            network.assert_not_called()

    def test_invalid_download_never_replaces_existing_file(self):
        destination = self.folder / "BS-Rofo-SW-Fixed.ckpt"
        destination.write_bytes(b"previous cache")
        for data in (b"wrong checksum", b"short", self.data + b"extra"):
            with self.subTest(data=data), patch("yams.models.urllib.request.urlopen", return_value=io.BytesIO(data)):
                with self.assertRaises(ValueError):
                    models.download(folder=self.folder, name=destination.name, asset=self.asset, emit=self.emit)
                self.assertEqual(destination.read_bytes(), b"previous cache")
                self.assertEqual(list(self.folder.glob("*.part")), [])

    def test_corrupt_cached_file_is_repaired(self):
        (self.folder / "BS-Rofo-SW-Fixed.ckpt").write_bytes(b"x" * len(self.data))
        with patch.object(models, "ASSETS", {"torch_6stem_files": {"BS-Rofo-SW-Fixed.ckpt": self.asset}}), \
                patch("yams.models.urllib.request.urlopen", return_value=io.BytesIO(self.data)):
            models.ensure_models(folder=self.folder, emit=self.emit, profile="torch-6stem")
        self.assertEqual((self.folder / "BS-Rofo-SW-Fixed.ckpt").read_bytes(), self.data)

    def test_cancel_and_network_failure_clean_partial_files(self):
        def cancel(event, **fields):
            if fields["received"] > 0:
                raise KeyboardInterrupt

        with patch("yams.models.urllib.request.urlopen", return_value=io.BytesIO(self.data)):
            with self.assertRaises(KeyboardInterrupt):
                models.download(folder=self.folder, name="BS-Rofo-SW-Fixed.ckpt", asset=self.asset, emit=cancel)
        with patch("yams.models.urllib.request.urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                models.download(folder=self.folder, name="BS-Rofo-SW-Fixed.ckpt", asset=self.asset, emit=self.emit)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_download_failure_does_not_start_song_transaction(self):
        (self.folder / "song.ini").write_text("[song]\n")
        (self.folder / "notes.mid").write_bytes(b"chart")
        (self.folder / "song.ogg").write_bytes(b"original")
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        output = io.StringIO()
        arguments = ["yams", str(self.folder), "--download-models", "--models", str(self.folder / "cache"),
                     "--inference-source", str(self.folder)]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(output), \
                patch("yams.models.ensure_models", side_effect=ValueError("download failed")):
            self.assertEqual(main(), 1)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})
        self.assertEqual([json.loads(line)["event"] for line in output.getvalue().splitlines()], ["models_error", "error"])

    def test_registry_has_no_legacy_mega_assets(self):
        self.assertNotIn("files", models.ASSETS)
        self.assertNotIn("revision", models.ASSETS)

    def test_unknown_model_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model profile"):
            models.default_models(profile="unknown")

    def test_default_models_returns_torch_6stem_path(self):
        path = models.default_models(profile="torch-6stem")
        self.assertTrue(str(path).replace("\\", "/").endswith("models/torch-6stem"))

    def test_registry_contains_pinned_torch_6stem_assets(self):
        self.assertIn("torch_6stem_files", models.ASSETS)
        self.assertEqual(len(models.ASSETS["torch_6stem_files"]), 2)
        for name in ("BS-Rofo-SW-Fixed.ckpt", "BS-Rofo-SW-Fixed.yaml"):
            asset = models.ASSETS["torch_6stem_files"][name]
            self.assertIn(models.ASSETS["torch_6stem_revision"], asset["url"])
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertGreater(asset["bytes"], 0)

    def test_registry_contains_pinned_drumsep_assets(self):
        self.assertIn("drumsep_files", models.ASSETS)
        self.assertEqual(len(models.ASSETS["drumsep_files"]), 2)
        for name in ("aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt", "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"):
            asset = models.ASSETS["drumsep_files"][name]
            self.assertIn(models.ASSETS["drumsep_revision"], asset["url"])
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertGreater(asset["bytes"], 0)

    def test_default_models_returns_8stem_path(self):
        path = models.default_models(profile="8stem")
        self.assertTrue(str(path).replace("\\", "/").endswith("models/torch-8stem") or str(path).replace("\\", "/").endswith("models/mlx-8stem"))


class BatchPreviewTests(unittest.TestCase):
    def test_one_process_previews_mixed_folders_without_ml_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name in ("ready", "skip", "pending", "invalid"):
                folder = root / name
                folder.mkdir()
                (folder / "song.ini").write_text("[song]\n")
                (folder / "notes.mid").write_bytes(b"chart")
                (folder / ("vocals.ogg" if name == "skip" else "song.ogg")).write_bytes(b"original")
                if name == "pending":
                    (folder / ".yams-work").mkdir()
                if name == "invalid":
                    (folder / "notes.mid").unlink()
                paths.append(str(folder))
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            output = io.StringIO()
            with patch.object(sys, "argv", ["yams", "--inspect-list"]), \
                    patch.object(sys, "stdin", io.StringIO(json.dumps(paths))), \
                    contextlib.redirect_stdout(output), patch.dict(sys.modules, {"torch": None}):
                self.assertEqual(main(), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["event"], "previews")
            self.assertEqual([song["status"] for song in result["songs"]], ["ready", "skipped", "recovery_required", "error"])
            self.assertEqual(before, sorted(str(path.relative_to(root)) for path in root.rglob("*")))
