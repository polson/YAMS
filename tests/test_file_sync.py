import os
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from yams import folders
from yams.audio_file import process_audio_file


class FileSyncTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fsync = os.fsync

        def sync_writable_file(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.write(descriptor, b"")
            fsync(descriptor)

        if os.name != "nt":
            self.enterContext(patch("os.fsync", side_effect=sync_writable_file))

    def make_outputs(self, stage, **kwargs):
        for name in folders.OUTPUTS:
            (stage / "outputs" / name).write_bytes(name.encode())
        files = {
            name: {"sha256": folders.digest(stage / "outputs" / name)}
            for name in folders.OUTPUTS
        }
        return {"validation": {"files": files}}

    def test_audio_snapshot_and_publication_use_writable_handles(self):
        audio = types.ModuleType("yams.audio")
        audio.separate = self.make_outputs
        with patch.dict("sys.modules", {"yams.audio": audio}):
            for mode in ("inplace", "export"):
                with self.subTest(mode=mode):
                    source = self.root / f"{mode}.flac"
                    source.write_bytes(b"original")
                    destination = self.root / mode
                    process_audio_file(
                        audio_path=source, destination=destination,
                        mode=mode, models=self.root, device="cuda:0",
                    )
                    self.assertEqual(source.read_bytes(), b"original")
                    self.assertEqual((destination / "song.opus").read_bytes(), b"song.opus")
                    self.assertFalse((destination / ".yams-work").exists())
                    if mode == "inplace":
                        self.assertEqual((destination / "backup" / source.name).read_bytes(), b"original")

    def test_chart_snapshot_prepare_and_install_use_writable_handles(self):
        source = self.root / "song.ogg"
        source.write_bytes(b"original")
        stage, record = folders.begin(self.root, source)
        folders.prepare(stage, record, self.make_outputs(stage))
        folders.install(self.root, stage, record)
        self.assertEqual((self.root / "backup/song.ogg").read_bytes(), b"original")
        self.assertFalse(stage.exists())

    def test_chart_export_uses_writable_handles(self):
        source = self.root / "song.ogg"
        source.write_bytes(b"original")
        destination = self.root / "export"
        stage, record = folders.begin_export(self.root, destination, source)
        folders.prepare(stage, record, self.make_outputs(stage))
        folders.install_export(self.root, destination, stage, record)
        self.assertEqual(source.read_bytes(), b"original")
        self.assertEqual((destination / "song.opus").read_bytes(), b"song.opus")
        self.assertFalse(stage.exists())

    def test_chart_recovery_uses_writable_handles(self):
        source = self.root / "song.ogg"
        source.write_bytes(b"original")
        stage, record = folders.begin(self.root, source)
        folders.prepare(stage, record, self.make_outputs(stage))
        with patch("yams.folders.os.link", side_effect=OSError("interrupted installation")):
            with self.assertRaisesRegex(OSError, "interrupted installation"):
                folders.install(self.root, stage, record)
        source.rename(stage / "removed-original")
        self.assertEqual(folders.recover(self.root), "rolled_back")
        self.assertEqual(source.read_bytes(), b"original")
        self.assertFalse(stage.exists())
