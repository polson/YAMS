import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yams import folders


class FolderTests(unittest.TestCase):
    def fixture(self, name="song.ogg"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        for filename, content in (("song.ini", b"[song]\nname = fixture\n"), ("notes.mid", b"chart"),
                                  ("album.png", b"art"), (name, b"original compressed audio")):
            (folder / filename).write_bytes(content)
        return folder

    def ready(self, folder, name="song.ogg"):
        stage, record = folders.begin(folder, folder / name)
        for output in folders.OUTPUTS:
            (stage / "outputs" / output).write_bytes(b"new stem " + output.encode())
        files = {name: {"sha256": folders.digest(stage / "outputs" / name)} for name in folders.OUTPUTS}
        folders.prepare(stage, record, {"validation": {"peak_error": 0, "files": files}})
        return stage, record

    def assert_original(self, folder, name):
        self.assertEqual((folder / name).read_bytes(), b"original compressed audio")
        self.assertEqual((folder / "notes.mid").read_bytes(), b"chart")
        self.assertEqual((folder / "album.png").read_bytes(), b"art")
        self.assertEqual((folder / "song.ini").read_bytes(), b"[song]\nname = fixture\n")
        self.assertFalse((folder / ".yams-work").exists())

    def test_lock_file_stays_stable_and_releases_lock(self):
        folder = self.fixture()
        lock = folder / ".yams.lock"
        with folders.locked(folder):
            first = lock.stat()
            with self.assertRaises(OSError):
                with folders.locked(folder):
                    pass
        self.assertTrue(lock.is_file())
        with folders.locked(folder):
            self.assertEqual(lock.stat().st_ino, first.st_ino)

    def test_success_and_rerun_skip(self):
        folder = self.fixture()
        stage, record = self.ready(folder)
        folders.install(folder, stage, record)
        self.assertEqual((folder / "backup/song.ogg").read_bytes(), b"original compressed audio")
        self.assertEqual({p.name for p in folder.glob("*.opus")}, set(folders.OUTPUTS))
        self.assertFalse((folder / "song.ogg").exists())
        self.assertIsNone(folders.discover(folder))
        manifest = json.loads((folder / "backup/separation-manifest.json").read_text())
        self.assertEqual(manifest["state"], "completed")
        self.assertEqual(set(manifest["outputs"]), set(folders.OUTPUTS))

    def test_failure_after_each_install_mutation(self):
        for name in ("song.ogg", "song.wav", "SONG.WAV", "song.opus", "SONG.OPUS"):
            for boundary in range(1, 13):
                with self.subTest(name=name, boundary=boundary):
                    folder = self.fixture(name)
                    stage, record = self.ready(folder, name)
                    calls = []
                    real_link = os.link
                    real_rename = os.rename
                    real_write = folders.write_record

                    def operation(function, *args, **kwargs):
                        result = function(*args, **kwargs)
                        calls.append(function.__name__)
                        if len(calls) == boundary:
                            raise OSError("simulated interruption after mutation")
                        return result

                    with patch.object(folders.os, "link", side_effect=lambda *a, **k: operation(real_link, *a, **k)), \
                         patch.object(folders.os, "rename", side_effect=lambda *a, **k: operation(real_rename, *a, **k)), \
                         patch.object(folders, "write_record", side_effect=lambda *a, **k: operation(real_write, *a, **k)):
                        with self.assertRaises(OSError):
                            folders.install(folder, stage, record)
                    result = folders.recover(folder)
                    if boundary == 12:
                        self.assertEqual(result, "completed_cleanup")
                        self.assertEqual((folder / "backup" / name).read_bytes(), b"original compressed audio")
                        self.assertEqual({p.name for p in folder.glob("*.opus")}, set(folders.OUTPUTS))
                    else:
                        self.assertEqual(result, "rolled_back")
                        self.assert_original(folder, name)

    def test_recovery_interrupted_after_restore(self):
        folder = self.fixture("song.wav")
        stage, record = self.ready(folder, "song.wav")
        real_link = os.link
        count = 0

        def fail_install(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 5:
                raise OSError("interrupted install")
            return real_link(*args, **kwargs)

        with patch.object(folders.os, "link", side_effect=fail_install):
            with self.assertRaises(OSError):
                folders.install(folder, stage, record)

        def fail_restore(*args, **kwargs):
            real_link(*args, **kwargs)
            raise OSError("interrupted after restore")

        with patch.object(folders.os, "link", side_effect=fail_restore):
            with self.assertRaises(OSError):
                folders.recover(folder)
        self.assertEqual(folders.recover(folder), "rolled_back")
        self.assert_original(folder, "song.wav")

    def test_committed_cleanup_does_not_need_staging_manifest(self):
        folder = self.fixture()
        stage, record = self.ready(folder)
        with patch.object(folders, "remove_stage", side_effect=OSError("cleanup interrupted")):
            with self.assertRaises(OSError):
                folders.install(folder, stage, record)
        (stage / "manifest.json").unlink()
        self.assertEqual(folders.recover(folder), "completed_cleanup")
        self.assertEqual((folder / "backup/song.ogg").read_bytes(), b"original compressed audio")

    def test_rollback_cleanup_does_not_need_removed_original(self):
        folder = self.fixture()
        stage, record = self.ready(folder)
        with patch.object(folders, "remove_stage", side_effect=OSError("cleanup interrupted")):
            with self.assertRaises(OSError):
                folders.recover(folder)
        (stage / "original").unlink()
        self.assertEqual(folders.recover(folder), "completed_rollback_cleanup")
        self.assert_original(folder, "song.ogg")

    def test_collision_is_never_overwritten(self):
        folder = self.fixture()
        stage, record = self.ready(folder)
        (folder / "guitar.opus").write_bytes(b"external addition")
        with self.assertRaises(FileExistsError):
            folders.install(folder, stage, record)
        with self.assertRaises(ValueError):
            folders.recover(folder)
        self.assertEqual((folder / "guitar.opus").read_bytes(), b"external addition")
        self.assertEqual((stage / "removed-original").read_bytes(), b"original compressed audio")
        (folder / "guitar.opus").unlink()
        folders.recover(folder)
        self.assert_original(folder, "song.ogg")

    def test_backup_collision_preserves_both(self):
        folder = self.fixture()
        stage, record = self.ready(folder)
        backup = folder / "backup"
        backup.mkdir()
        (backup / "song.ogg").write_bytes(b"older backup")
        with self.assertRaises(FileExistsError):
            folders.install(folder, stage, record)
        with self.assertRaises(ValueError):
            folders.recover(folder)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original compressed audio")
        self.assertEqual((backup / "song.ogg").read_bytes(), b"older backup")

    def test_changed_source_retains_snapshot(self):
        folder = self.fixture()
        stage, record = self.ready(folder)
        (folder / "song.ogg").write_bytes(b"changed source")
        with self.assertRaises(ValueError):
            folders.install(folder, stage, record)
        with self.assertRaises(ValueError):
            folders.recover(folder)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"changed source")
        self.assertEqual((stage / "original").read_bytes(), b"original compressed audio")

    def test_discovery_and_symlink_boundaries(self):
        folder = self.fixture("song.opus")
        self.assertEqual(folders.discover(folder), folder / "song.opus")
        (folder / "song.ogg").write_bytes(b"duplicate")
        with self.assertRaises(ValueError):
            folders.discover(folder)
        (folder / "song.ogg").unlink()
        (folder / "vocals.wav").write_bytes(b"existing stem")
        self.assertIsNone(folders.discover(folder))
        (folder / "vocals.wav").unlink()
        (folder / "backup").symlink_to(folder, target_is_directory=True)
        with self.assertRaises(ValueError):
            folders.discover(folder)

    def test_processing_and_empty_recovery(self):
        folder = self.fixture()
        stage, record = folders.begin(folder, folder / "song.ogg")
        self.assertEqual(folders.recover(folder), "discarded_staging")
        self.assert_original(folder, "song.ogg")
        stage.mkdir()
        self.assertEqual(folders.recover(folder), "discarded_empty_staging")

    def test_skip_does_not_import_ml_or_modify_folder(self):
        folder = self.fixture()
        (folder / "guitar.ogg").write_bytes(b"existing audio")
        before = sorted(p.name for p in folder.iterdir())
        result = subprocess.run([sys.executable, "-m", "yams", str(folder)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["event"], "skipped")
        self.assertEqual(before, sorted(p.name for p in folder.iterdir()))

    def test_export_success_leaves_source_intact_and_populates_destination(self):
        folder = self.fixture()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "exported_song"
        stage, record = folders.begin_export(folder, destination, folder / "song.ogg")
        for output in folders.OUTPUTS:
            (stage / "outputs" / output).write_bytes(b"stem " + output.encode())
        files = {name: {"sha256": folders.digest(stage / "outputs" / name)} for name in folders.OUTPUTS}
        folders.prepare(stage, record, {"validation": {"peak_error": 0, "files": files}})
        folders.install_export(folder, destination, stage, record)

        self.assert_original(folder, "song.ogg")
        self.assertFalse((folder / "backup").exists())
        self.assertEqual({p.name for p in destination.glob("*.opus")}, set(folders.OUTPUTS))
        self.assertEqual((destination / "notes.mid").read_bytes(), b"chart")
        self.assertEqual((destination / "album.png").read_bytes(), b"art")
        self.assertEqual((destination / "song.ini").read_bytes(), b"[song]\nname = fixture\n")
        self.assertFalse((destination / "separation-manifest.json").exists())
        self.assertFalse((destination / ".yams-work").exists())

    def test_export_abort_cleans_staging_and_preserves_source(self):
        folder = self.fixture()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "aborted_song"
        stage, record = folders.begin_export(folder, destination, folder / "song.ogg")
        self.assertTrue((destination / ".yams-work").exists())
        folders.abort_export(destination)
        self.assert_original(folder, "song.ogg")
        self.assertFalse(destination.exists())

    def test_export_refuses_to_replace_existing_destination_files(self):
        folder = self.fixture()
        for name in ("song.ini", folders.OUTPUTS[0]):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / "exported_song"
                destination.mkdir()
                target = destination / name
                target.write_bytes(b"user data")
                stage, record = folders.begin_export(folder, destination, folder / "song.ogg")
                for output in folders.OUTPUTS:
                    (stage / "outputs" / output).write_bytes(b"stem " + output.encode())
                files = {output: {"sha256": folders.digest(stage / "outputs" / output)}
                         for output in folders.OUTPUTS}
                folders.prepare(stage, record, {"validation": {"peak_error": 0, "files": files}})

                with self.assertRaises(FileExistsError):
                    folders.install_export(folder, destination, stage, record)

                self.assertEqual(target.read_bytes(), b"user data")
                folders.abort_export(destination)

    def test_export_recovery_rolls_back_only_its_published_files(self):
        from yams.__main__ import recover_transaction

        folder = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "exported_song"
            stage, record = folders.begin_export(folder, destination, folder / "song.ogg")
            payload = stage / "payload"
            payload.mkdir()
            staged_file = payload / "notes.mid"
            staged_file.write_bytes(b"copied chart")
            target = destination / "notes.mid"
            os.link(staged_file, target)
            status = target.stat()
            record.update({
                "state": "publishing",
                "files": {"notes.mid": folders.digest(staged_file)},
                "directories": [],
                "installed": {"notes.mid": [status.st_dev, status.st_ino]},
            })
            folders.write_record(stage, record)

            self.assertTrue(recover_transaction(folder, output_folder=destination))
            self.assertFalse(target.exists())
            self.assertFalse((destination / ".yams-work").exists())


if __name__ == "__main__":
    unittest.main()
