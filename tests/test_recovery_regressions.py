import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yams import folders


class RecoveryRegressionTests(unittest.TestCase):
    def ready(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "song.ini").write_text("[song]\n")
        (folder / "notes.mid").write_bytes(b"chart")
        (folder / "song.ogg").write_bytes(b"original")
        stage, record = folders.begin(folder, folder / "song.ogg")
        for name in folders.OUTPUTS:
            (stage / "outputs" / name).write_bytes(name.encode())
        files = {name: {"sha256": folders.digest(stage / "outputs" / name)} for name in folders.OUTPUTS}
        folders.prepare(stage, record, {"validation": {"files": files}})
        return folder, stage, record

    def test_changed_snapshot_preserves_active_original(self):
        folder, stage, record = self.ready()
        (stage / "original").write_bytes(b"changed snapshot")
        with self.assertRaises(ValueError):
            folders.install(folder, stage, record)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / "backup").exists())

    def test_changed_output_invalidates_validation(self):
        folder, stage, record = self.ready()
        (stage / "outputs/guitar.opus").write_bytes(b"changed output")
        with self.assertRaises(ValueError):
            folders.prepare(stage, record, record["audio"])
        journal = json.loads((stage / "job.json").read_text())
        with self.assertRaises(ValueError):
            folders.install(folder, stage, journal)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")

    def test_identical_external_output_is_not_owned(self):
        folder, stage, record = self.ready()
        external = folder / "guitar.opus"
        external.write_bytes((stage / "outputs/guitar.opus").read_bytes())
        with self.assertRaises(FileExistsError):
            folders.install(folder, stage, record)
        with self.assertRaises(ValueError):
            folders.recover(folder)
        self.assertEqual(external.read_bytes(), b"guitar.opus")
        self.assertFalse(external.samefile(stage / "outputs/guitar.opus"))
        external.unlink()
        folders.recover(folder)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse(stage.exists())

    def test_identical_external_backup_is_not_owned(self):
        folder, stage, record = self.ready()
        (folder / "backup").mkdir()
        external = folder / "backup/song.ogg"
        external.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            folders.install(folder, stage, record)
        with self.assertRaises(ValueError):
            folders.recover(folder)
        self.assertEqual(external.read_bytes(), b"original")
        self.assertFalse(external.samefile(stage / "original"))
        self.assertTrue(stage.exists())

    def test_temporary_initial_journal_is_recoverable(self):
        folder, stage, record = self.ready()
        journal = {**record, "state": "processing", "outputs": {}}
        (stage / "job.json").unlink()
        (stage / "job.tmp").write_text(json.dumps(journal))
        self.assertEqual(folders.recover(folder), "discarded_staging")
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse(stage.exists())


class LegacyRecoveryTests(unittest.TestCase):
    def ready(self, source):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / source).write_bytes(b"legacy original")
        stage, record = folders.begin(folder, folder / source)
        for name in folders.WAV_OUTPUTS:
            (stage / "outputs" / name).write_bytes(name.encode())
        files = {name: {"sha256": folders.digest(stage / "outputs" / name)} for name in folders.WAV_OUTPUTS}
        with patch.object(folders, "OUTPUTS", folders.WAV_OUTPUTS):
            folders.prepare(stage, record, {"validation": {"files": files}})
        return folder, stage, record

    def test_wav_journal_recovers_every_installation_boundary(self):
        for source in ("song.ogg", "song.wav", "SONG.WAV", "song.opus"):
            for boundary in range(1, 13):
                with self.subTest(source=source, boundary=boundary):
                    folder, stage, record = self.ready(source)
                    calls = []
                    real_link, real_rename, real_write = os.link, os.rename, folders.write_record

                    def interrupt(function, *args, **kwargs):
                        result = function(*args, **kwargs)
                        calls.append(function.__name__)
                        if len(calls) == boundary:
                            raise OSError("legacy interruption")
                        return result

                    with patch.object(folders, "OUTPUTS", folders.WAV_OUTPUTS), \
                         patch.object(folders.os, "link", side_effect=lambda *a, **k: interrupt(real_link, *a, **k)), \
                         patch.object(folders.os, "rename", side_effect=lambda *a, **k: interrupt(real_rename, *a, **k)), \
                         patch.object(folders, "write_record", side_effect=lambda *a, **k: interrupt(real_write, *a, **k)):
                        with self.assertRaises(OSError):
                            folders.install(folder, stage, record)
                    result = folders.recover(folder)
                    if boundary == 12:
                        self.assertEqual(result, "completed_cleanup")
                        self.assertEqual((folder / "backup" / source).read_bytes(), b"legacy original")
                        self.assertEqual({path.name for path in folder.glob("*.wav")}, set(folders.WAV_OUTPUTS))
                    else:
                        self.assertEqual(result, "rolled_back")
                        self.assertEqual((folder / source).read_bytes(), b"legacy original")
                        self.assertFalse((folder / "backup").exists())
                    self.assertFalse(stage.exists())

    def test_legacy_rollback_cleanup_can_be_retried(self):
        folder, stage, record = self.ready("song.wav")
        with patch.object(folders, "OUTPUTS", folders.WAV_OUTPUTS), \
             patch.object(folders.os, "link", side_effect=OSError("installation interrupted")):
            with self.assertRaises(OSError):
                folders.install(folder, stage, record)
        with patch.object(folders, "remove_stage", side_effect=OSError("cleanup interrupted")):
            with self.assertRaises(OSError):
                folders.recover(folder)
        self.assertEqual(folders.recover(folder), "completed_rollback_cleanup")
        self.assertEqual((folder / "song.wav").read_bytes(), b"legacy original")
        self.assertFalse(stage.exists())

    def test_mixed_format_journal_is_rejected(self):
        folder, stage, record = self.ready("song.wav")
        record["outputs"]["guitar.opus"] = record["outputs"].pop("guitar.wav")
        folders.write_record(stage, record)
        with self.assertRaisesRegex(ValueError, "Invalid output names"):
            folders.recover(folder)
        self.assertEqual((folder / "song.wav").read_bytes(), b"legacy original")
        self.assertTrue(stage.exists())


if __name__ == "__main__":
    unittest.main()
