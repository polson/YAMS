import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from yams.audio_file import (
    is_audio_file,
    is_multitrack_folder,
    find_single_audio_in_folder,
    process_audio_file,
    abort_audio_process,
)
from yams.__main__ import preview, recover_transaction
from yams.folders import digest, discover, write_record


class AudioFileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="yams-test-audio-"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_audio_file_recognizes_supported_extensions(self):
        for ext in (".opus", ".ogg", ".mp3", ".wav", ".flac", ".aiff", ".m4a"):
            file_path = self.temp_dir / f"test{ext}"
            file_path.write_bytes(b"dummy audio content")
            self.assertTrue(is_audio_file(file_path), f"Failed for {ext}")

        non_audio = self.temp_dir / "notes.chart"
        non_audio.write_bytes(b"dummy")
        self.assertFalse(is_audio_file(non_audio))

        con_file = self.temp_dir / "song.con"
        con_file.write_bytes(b"CON dummy")
        self.assertFalse(is_audio_file(con_file))

    def test_is_multitrack_folder(self):
        folder = self.temp_dir / "stems"
        folder.mkdir()
        (folder / "vocals.ogg").write_bytes(b"vocal")
        (folder / "drums.ogg").write_bytes(b"drums")
        self.assertTrue(is_multitrack_folder(folder))

        album = self.temp_dir / "album"
        album.mkdir()
        (album / "01_intro.opus").write_bytes(b"track1")
        (album / "02_verse.opus").write_bytes(b"track2")
        self.assertFalse(is_multitrack_folder(album))

        single = self.temp_dir / "single"
        single.mkdir()
        (single / "song.opus").write_bytes(b"audio")
        self.assertFalse(is_multitrack_folder(single))

    def test_find_single_audio_in_folder(self):
        folder = self.temp_dir / "one_song"
        folder.mkdir()
        track = folder / "audio.mp3"
        track.write_bytes(b"mp3")
        self.assertEqual(find_single_audio_in_folder(folder), track)

        (folder / "extra.wav").write_bytes(b"wav")
        self.assertIsNone(find_single_audio_in_folder(folder))

        (folder / "extra.wav").unlink()
        (folder / "song.ini").write_text("[Song]\nname = Test", encoding="utf-8")
        self.assertIsNone(find_single_audio_in_folder(folder))

    def test_preview_standalone_audio_file(self):
        audio_file = self.temp_dir / "bohemian_rhapsody.opus"
        audio_file.write_bytes(b"12345678")
        result = preview(audio_file)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["source"], "bohemian_rhapsody.opus")
        self.assertEqual(result["bytes"], 8)
        self.assertEqual(result["backup"], "backup/bohemian_rhapsody.opus")

    def test_preview_multitrack_folder_is_skipped(self):
        folder = self.temp_dir / "band_stems"
        folder.mkdir()
        (folder / "guitar.ogg").write_bytes(b"guitar")
        (folder / "bass.ogg").write_bytes(b"bass")
        result = preview(folder)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["message"], "Already contains multitrack stems")

    def test_alternate_drum_stem_names_are_recognized(self):
        for stem in ("kick", "snare", "other"):
            with self.subTest(stem=stem):
                folder = self.temp_dir / stem
                folder.mkdir()
                (folder / "song.ogg").write_bytes(b"song")
                (folder / f"{stem}.ogg").write_bytes(stem.encode())

                self.assertTrue(is_multitrack_folder(folder))
                self.assertEqual(preview(folder)["status"], "skipped")

    def test_discover_skips_alternate_drum_stems(self):
        for stem in ("kick", "snare", "other"):
            with self.subTest(stem=stem):
                folder = self.temp_dir / f"chart_{stem}"
                folder.mkdir()
                (folder / "song.ini").write_text("[Song]\n", encoding="utf-8")
                (folder / "notes.mid").write_bytes(b"chart")
                (folder / "song.ogg").write_bytes(b"song")
                (folder / f"{stem}.ogg").write_bytes(stem.encode())

                self.assertIsNone(discover(folder))

    def test_preview_single_audio_folder(self):
        folder = self.temp_dir / "My Album Track"
        folder.mkdir()
        (folder / "track.flac").write_bytes(b"flac data")
        result = preview(folder)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["source"], "track.flac")
        self.assertEqual(result["backup"], "backup/track.flac")

    def test_abort_audio_process_rolls_back_partial_publication(self):
        source_folder = self.temp_dir / "source"
        source_folder.mkdir()
        source = source_folder / "track.mp3"
        source.write_bytes(b"source audio")
        destination = self.temp_dir / "export"
        destination.mkdir()
        existing = destination / "notes.txt"
        existing.write_text("keep", encoding="utf-8")

        def separate(stage, **kwargs):
            outputs = stage / "outputs"
            (outputs / "drums_1.opus").write_bytes(b"drums")
            (outputs / "song.opus").write_bytes(b"song")
            return {"validation": {"files": {"drums_1.opus": {}, "song.opus": {}}}}

        audio = types.ModuleType("yams.audio")
        audio.separate = separate
        original_link = os.link
        original_copy = shutil.copyfileobj

        def fail_on_song_ini(src, dst):
            dst = Path(dst)
            if dst == destination / "song.ini":
                raise OSError("hard links are unavailable")
            original_link(src, dst)

        def fail_song_ini_copy(fsrc, fdst, length=0):
            if Path(fdst.name) == destination / "song.ini":
                raise OSError("simulated install failure")
            original_copy(fsrc, fdst, length)

        with patch.dict("sys.modules", {"yams.audio": audio}):
            with patch("yams.folders.os.link", side_effect=fail_on_song_ini):
                with patch("yams.folders.shutil.copyfileobj", side_effect=fail_song_ini_copy):
                    with self.assertRaisesRegex(OSError, "simulated install failure"):
                        try:
                            process_audio_file(
                                audio_path=source,
                                destination=destination,
                                mode="export",
                                models=self.temp_dir,
                                inference_source=self.temp_dir,
                                device="cpu",
                            )
                        except OSError:
                            abort_audio_process(destination=destination, mode="export", audio_path=source)
                            raise

        self.assertTrue(source.is_file())
        self.assertFalse((destination / "drums_1.opus").exists())
        self.assertFalse((destination / "song.opus").exists())
        self.assertFalse((destination / "song.ini").exists())
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.assertFalse((destination / ".yams-work").exists())

    def test_process_audio_file_copies_when_hardlinks_are_unavailable(self):
        source = self.temp_dir / "track.mp3"
        source.write_bytes(b"source audio")
        destination = self.temp_dir / "export"
        destination.mkdir()

        def separate(stage, **kwargs):
            outputs = stage / "outputs"
            (outputs / "drums_1.opus").write_bytes(b"drums")
            (outputs / "song.opus").write_bytes(b"song")
            return {"validation": {"files": {"drums_1.opus": {}, "song.opus": {}}}}

        audio = types.ModuleType("yams.audio")
        audio.separate = separate
        with patch.dict("sys.modules", {"yams.audio": audio}):
            with patch("yams.folders.os.link", side_effect=OSError("hard links unavailable")):
                process_audio_file(
                    audio_path=source,
                    destination=destination,
                    mode="export",
                    models=self.temp_dir,
                    inference_source=self.temp_dir,
                    device="cpu",
                )

        self.assertEqual((destination / "drums_1.opus").read_bytes(), b"drums")
        self.assertEqual((destination / "song.opus").read_bytes(), b"song")
        self.assertTrue((destination / "song.ini").is_file())
        self.assertTrue(source.is_file())
        self.assertFalse((destination / ".yams-work").exists())

    def test_abort_restores_same_named_source_after_copy_publish_failure(self):
        destination = self.temp_dir / "song"
        destination.mkdir()
        source = destination / "song.opus"
        source.write_bytes(b"original song")

        def separate(stage, **kwargs):
            (stage / "outputs" / "song.opus").write_bytes(b"separated song")
            return {"validation": {"files": {"song.opus": {}}}}

        def fail_commit(stage, record):
            if record["state"] == "committed":
                raise OSError("simulated commit failure")
            write_record(stage, record)

        audio = types.ModuleType("yams.audio")
        audio.separate = separate
        with patch.dict("sys.modules", {"yams.audio": audio}):
            with patch("os.link", side_effect=OSError("hard links unavailable")):
                with patch("yams.audio_file.write_record", side_effect=fail_commit):
                    with self.assertRaisesRegex(OSError, "simulated commit failure"):
                        try:
                            process_audio_file(
                                audio_path=source,
                                destination=destination,
                                mode="inplace",
                                models=self.temp_dir,
                                inference_source=self.temp_dir,
                                device="cpu",
                            )
                        except OSError:
                            abort_audio_process(destination=destination, mode="inplace", audio_path=source)
                            raise

        self.assertEqual(source.read_bytes(), b"original song")
        self.assertFalse((destination / ".yams-work").exists())
        self.assertFalse((destination / "backup").exists())

    def test_recover_transaction_routes_missing_audio_source_from_journal(self):
        destination = self.temp_dir / "song"
        destination.mkdir()
        source = destination / "song.opus"
        stage = destination / ".yams-work"
        payload = stage / "payload"
        backup = payload / "backup" / source.name
        backup.parent.mkdir(parents=True)
        backup.write_bytes(b"original song")
        write_record(stage, {
            "version": 1,
            "format": "audio_file",
            "state": "processing",
            "source": str(source.resolve()),
            "source_sha256": digest(backup),
            "source_backup": f"backup/{source.name}",
            "remove_source": True,
            "source_replaced": True,
            "mode": "inplace",
            "destination": str(destination.resolve()),
            "files": {},
        })

        self.assertTrue(recover_transaction(destination))
        self.assertEqual(source.read_bytes(), b"original song")
        self.assertFalse(stage.exists())

    def test_process_audio_file_backs_up_same_named_source_before_publish(self):
        destination = self.temp_dir / "song"
        destination.mkdir()
        source = destination / "song.opus"
        source.write_bytes(b"original song")

        def separate(stage, **kwargs):
            (stage / "outputs" / "song.opus").write_bytes(b"separated song")
            return {"validation": {"files": {"song.opus": {}}}}

        audio = types.ModuleType("yams.audio")
        audio.separate = separate
        with patch.dict("sys.modules", {"yams.audio": audio}):
            process_audio_file(
                audio_path=source,
                destination=destination,
                mode="inplace",
                models=self.temp_dir,
                inference_source=self.temp_dir,
                device="cpu",
            )

        self.assertEqual(source.read_bytes(), b"separated song")
        self.assertEqual((destination / "backup" / "song.opus").read_bytes(), b"original song")
        self.assertTrue((destination / "backup" / "separation-manifest.json").is_file())
        self.assertFalse((destination / ".yams-work").exists())

    def test_failed_same_name_publish_restores_original_source(self):
        destination = self.temp_dir / "song"
        destination.mkdir()
        source = destination / "song.opus"
        source.write_bytes(b"original song")

        def separate(stage, **kwargs):
            (stage / "outputs" / "song.opus").write_bytes(b"separated song")
            return {"validation": {"files": {"song.opus": {}}}}

        audio = types.ModuleType("yams.audio")
        audio.separate = separate
        original_link = os.link
        original_copy = shutil.copyfileobj

        def fail_on_song_output(src, dst):
            if Path(dst) == source:
                raise OSError("hard links are unavailable")
            original_link(src, dst)

        def fail_song_output_copy(fsrc, fdst, length=0):
            if Path(fdst.name) == source:
                raise OSError("simulated install failure")
            original_copy(fsrc, fdst, length)

        with patch.dict("sys.modules", {"yams.audio": audio}):
            with self.assertRaisesRegex(OSError, "simulated install failure"):
                with patch("yams.folders.os.link", side_effect=fail_on_song_output):
                    with patch("yams.folders.shutil.copyfileobj", side_effect=fail_song_output_copy):
                        process_audio_file(
                            audio_path=source,
                            destination=destination,
                            mode="inplace",
                            models=self.temp_dir,
                            inference_source=self.temp_dir,
                            device="cpu",
                        )
        abort_audio_process(destination=destination, mode="inplace", audio_path=source)

        self.assertEqual(source.read_bytes(), b"original song")
        self.assertFalse((destination / "backup").exists())
        self.assertFalse((destination / "song.ini").exists())
        self.assertFalse((destination / ".yams-work").exists())


if __name__ == "__main__":
    unittest.main()
