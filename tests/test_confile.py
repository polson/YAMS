import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yams import confile
from yams.__main__ import recover_transaction
from yams.folders import digest, write_record


def create_synthetic_con(tmp_dir, mogg_version=10, dta_extra=""):
    import subprocess
    from pathlib import Path

    # Make a small 2-channel Ogg Vorbis file
    tmp_ogg = Path(tmp_dir) / "temp.ogg"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1",
        "-ac", "2", "-c:a", "libvorbis", str(tmp_ogg)
    ], check=True, capture_output=True)
    ogg_bytes = tmp_ogg.read_bytes()
    tmp_ogg.unlink()

    mogg_bytes = struct.pack("<II", mogg_version, 8) + ogg_bytes
    dta_text = (
        '(testsong\n'
        '  (name "Test Song")\n'
        '  (song\n'
        '    (name "songs/testsong/testsong")\n'
        '    (tracks ((drum (0 1))))\n'
        '    (pans (-1.0 1.0))\n'
        '    (vols (0.0 0.0))\n'
        '  )\n'
        '  (anim_tempo kAnimTempoMedium)\n'
        ')\n'
    )
    if dta_extra:
        dta_text += dta_extra
    dta_content = dta_text.encode("utf-8")
    mid_content = b"MThd" + b"\x00" * 20

    files = {
        "songs/songs.dta": dta_content,
        "songs/testsong/testsong.mid": mid_content,
        "songs/testsong/testsong.mogg": mogg_bytes,
    }

    # Calculate blocks
    # block 0 = file table, blocks 1..N = files
    file_blocks = []
    for content in files.values():
        cnt = (len(content) + confile.BYTES_PER_BLOCK - 1) // confile.BYTES_PER_BLOCK
        file_blocks.append(max(1, cnt))

    total_blocks = 1 + sum(file_blocks)
    total_size = confile.calculate_block_location(total_blocks + 1, 0) + confile.BYTES_PER_BLOCK
    image = bytearray(total_size)
    image[0:4] = b"CON "
    image[0x340:0x344] = struct.pack(">I", 0xB000)
    image[0x37C:0x37E] = struct.pack("<H", 1)
    image[0x37E:0x381] = bytes([0, 0, 0])

    table_offset = confile.calculate_block_location(0, 0)
    current_entry_offset = table_offset
    current_block = 1

    for name, content in files.items():
        entry = bytearray(64)
        name_bytes = name.encode("ascii")
        entry[:len(name_bytes)] = name_bytes
        entry[0x28] = len(name_bytes) | 0x40  # contiguous
        cnt = (len(content) + confile.BYTES_PER_BLOCK - 1) // confile.BYTES_PER_BLOCK
        entry[0x29:0x2C] = bytes([cnt & 0xFF, (cnt >> 8) & 0xFF, (cnt >> 16) & 0xFF])
        entry[0x2F:0x32] = bytes([current_block & 0xFF, (current_block >> 8) & 0xFF, (current_block >> 16) & 0xFF])
        entry[0x32:0x34] = struct.pack(">h", -1)
        entry[0x34:0x38] = struct.pack(">I", len(content))
        image[current_entry_offset:current_entry_offset + 64] = entry
        current_entry_offset += 64

        block_loc = confile.calculate_block_location(current_block, 0)
        image[block_loc:block_loc + len(content)] = content
        current_block += cnt

    con_path = Path(tmp_dir) / "testsong.con"
    con_path.write_bytes(bytes(image))
    return con_path


class ConFileTests(unittest.TestCase):
    def test_is_con_file(self):
        self.assertTrue(confile.is_con_file(b"CON \x00\x00\x00\x00"))
        self.assertTrue(confile.is_con_file(b"LIVE\x00\x00\x00\x00"))
        self.assertTrue(confile.is_con_file(b"PIRS\x00\x00\x00\x00"))
        self.assertFalse(confile.is_con_file(b"OggS\x00\x00\x00\x00"))
        self.assertFalse(confile.is_con_file(b"RIFF\x00\x00\x00\x00"))

    def test_constructor_closes_only_owned_streams_when_parsing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.con"
            path.write_bytes(b"invalid")
            stream = path.open("rb")
            with patch.object(Path, "open", return_value=stream):
                with self.assertRaisesRegex(ValueError, "Invalid package header"):
                    confile.ConPackage(path)
            self.assertTrue(stream.closed)

        stream = io.BytesIO(b"invalid")
        with self.assertRaisesRegex(ValueError, "Invalid package header"):
            confile.ConPackage(stream)
        self.assertFalse(stream.closed)

    def test_demux_mogg(self):
        fake_ogg = b"OggS" + b"\x00" * 32
        mogg_data = struct.pack("<II", 10, 8) + fake_ogg
        extracted = confile.demux_mogg(mogg_data)
        self.assertEqual(extracted, fake_ogg)

        encrypted_mogg = struct.pack("<II", 17, 8) + fake_ogg
        with self.assertRaises(ValueError):
            confile.demux_mogg(encrypted_mogg)

    def test_parse_dta_audio_info(self):
        dta = """
        (testsong
          (name "Test Song")
          (song
            (name "songs/testsong/testsong")
            (tracks
              ((drum (0 1))
               (bass 2)
               (guitar (3 4))
               (vocals (5 6))
              )
            )
            (pans (-1.0 1.0 0.0 -1.0 1.0 -1.0 1.0))
            (vols (0.0 0.0 -1.5 0.0 0.0 -2.0 -2.0))
            (cores (-1 -1 -1 -1 -1 -1 -1))
          )
          (rank
            (drum 178)
            (vocals 221)
            (band 243)
          )
        )
        """
        tree = confile.parse_dta(dta)
        info = confile.extract_audio_info_from_dta(tree)
        self.assertEqual(info["song_id"], "testsong")
        self.assertEqual(info["name"], "songs/testsong/testsong")
        self.assertEqual(info["pans"], [-1.0, 1.0, 0.0, -1.0, 1.0, -1.0, 1.0])
        self.assertEqual(info["vols"], [0.0, 0.0, -1.5, 0.0, 0.0, -2.0, -2.0])
        self.assertEqual(info["tracks"]["drum"], [0, 1])
        self.assertEqual(info["tracks"]["bass"], [2])
        self.assertEqual(info["tracks"]["guitar"], [3, 4])
        self.assertEqual(info["tracks"]["vocals"], [5, 6])

    def test_read_synthetic_contiguous_con_package(self):
        # Build synthetic in-memory CON container
        # Image needs to cover up to block 2 + 1 block
        total_size = confile.calculate_block_location(2, 0) + confile.BYTES_PER_BLOCK
        image = bytearray(total_size)
        image[0:4] = b"CON "

        # Entry ID at 0x340 (0xB000 for shift=0)
        image[0x340:0x344] = struct.pack(">I", 0xB000)
        # File table block count = 1 at 0x37C
        image[0x37C:0x37E] = struct.pack("<H", 1)
        # First block = 0 at 0x37E
        image[0x37E:0x381] = bytes([0, 0, 0])

        # Table block offset for block 0, shift 0:
        table_offset = confile.calculate_block_location(0, 0)

        # File 1: "songs.dta" stored at block 1
        # 64-byte entry:
        # name at 0..len, len&0x3F at 0x28, flags at 0x28 (0x40 = contiguous)
        dta_name = b"songs.dta"
        dta_content = b'(test (name "Test"))'
        entry = bytearray(64)
        entry[:len(dta_name)] = dta_name
        entry[0x28] = len(dta_name) | 0x40  # contiguous
        # block count = 1 (little-endian 24-bit at 0x29..0x2B)
        entry[0x29:0x2C] = bytes([1, 0, 0])
        # block offset = 1 (little-endian 24-bit at 0x2F..0x31)
        entry[0x2F:0x32] = bytes([1, 0, 0])
        # path index = -1 (big-endian signed 16-bit at 0x32..0x34)
        entry[0x32:0x34] = struct.pack(">h", -1)
        # file length (big-endian 32-bit at 0x34..0x38)
        entry[0x34:0x38] = struct.pack(">I", len(dta_content))

        image[table_offset:table_offset + 64] = entry

        # Put dta_content at block 1
        block1_loc = confile.calculate_block_location(1, 0)
        image[block1_loc:block1_loc + len(dta_content)] = dta_content

        import io
        stream = io.BytesIO(bytes(image))
        pkg = confile.ConPackage(stream)
        self.assertEqual(len(pkg.listings), 1)
        self.assertEqual(pkg.listings[0].name, "songs.dta")
        self.assertEqual(pkg.read_file(pkg.listings[0]), dta_content)

    def test_patch_dta_audio(self):
        original_dta = """(mysong
  (name "My Song")
  (song
    (name "songs/mysong/mysong")
    (tracks ((drum (0 1))))
    (pans (-1.0 1.0))
    (vols (0.0 0.0))
  )
  (anim_tempo kAnimTempoMedium)
  (rank (guitar 150))
)"""
        patched = confile.patch_dta_audio(original_dta)
        self.assertIn("anim_tempo kAnimTempoMedium", patched)
        self.assertIn("rank (guitar 150)", patched)
        self.assertIn("drum (0 1 2 3 4 5)", patched)
        self.assertIn("bass (6 7)", patched)
        self.assertIn("guitar (8 9)", patched)
        self.assertIn("vocals (10 11)", patched)
        self.assertIn("keys (12 13)", patched)

    def test_patch_dta_audio_6stem(self):
        original_dta = """(mysong
  (name "My Song")
  (song
    (name "songs/mysong/mysong")
    (tracks ((drum (0 1))))
    (pans (-1.0 1.0))
    (vols (0.0 0.0))
  )
  (anim_tempo kAnimTempoMedium)
  (rank (guitar 150))
)"""
        patched = confile.patch_dta_audio(original_dta, profile="torch-6stem")
        self.assertIn("anim_tempo kAnimTempoMedium", patched)
        self.assertIn("rank (guitar 150)", patched)
        self.assertIn("drum (0 1)", patched)
        self.assertIn("bass (2 3)", patched)
        self.assertIn("guitar (4 5)", patched)
        self.assertIn("vocals (6 7)", patched)
        self.assertIn("keys (8 9)", patched)
        self.assertIn("(pans (-1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0))", patched)
        self.assertIn("(cores (-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1))", patched)

    def test_create_unencrypted_mogg(self):
        import subprocess
        import tempfile
        from pathlib import Path
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp = Path(temporary.name)
        stems = []
        for i in range(8):
            stem = tmp / f"stem_{i}.wav"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                "-ac", "2", str(stem)
            ], check=True, capture_output=True)
            stems.append(stem)
        mogg_path = tmp / "output.mogg"
        confile.create_unencrypted_mogg(stems, mogg_path)
        self.assertTrue(mogg_path.exists())
        data = mogg_path.read_bytes()
        self.assertTrue(len(data) > 8)
        version, offset = struct.unpack("<II", data[:8])
        self.assertEqual(version, 10)
        self.assertEqual(offset, 8)
        self.assertEqual(data[8:12], b"OggS")

    def _create_synthetic_con_file(self, tmp_dir, mogg_version=10):
        return create_synthetic_con(tmp_dir, mogg_version=mogg_version)

    def test_preview_con_file(self):
        import tempfile
        from yams.__main__ import preview
        with tempfile.TemporaryDirectory() as tmp:
            con_path = self._create_synthetic_con_file(tmp)
            result = preview(con_path)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["source"], "testsong.con")
            self.assertEqual(result["backup"], "backup/testsong.con")

    def test_preview_encrypted_mogg(self):
        import tempfile
        from yams.__main__ import preview
        with tempfile.TemporaryDirectory() as tmp:
            con_path = self._create_synthetic_con_file(tmp, mogg_version=17)
            result = preview(con_path)
            self.assertEqual(result["status"], "skipped")
            self.assertIn("Encrypted", result["message"])

    def test_preview_multitrack_con_skipped(self):
        import tempfile
        from yams.__main__ import preview
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp, dta_extra=";Multitrack=1\n")
            result = preview(con_path)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["message"], "Already contains multitrack stems")

    def test_process_con_song_raises_on_multitrack(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp, dta_extra=";Multitrack=1\n")
            dest = Path(tmp) / "testsong"
            with self.assertRaises(ValueError):
                confile.process_con_song(
                    con_path=con_path,
                    destination=dest,
                    mode="inplace",
                    models=Path(tmp),
                    inference_source=Path(tmp),
                    device="cpu",
                    emit=lambda *a, **k: None,
                )

    def test_is_con_multitrack_detection(self):
        dta_mt0 = "(test (song (tracks ((drum (0 1)) (bass (2 3)))))) ;Multitrack=0"
        self.assertFalse(confile.is_con_multitrack(dta_mt0))

        dta_mt1 = "(test (song (tracks ((drum (0 1)) (bass (2 3)))))) ;Multitrack=1"
        self.assertTrue(confile.is_con_multitrack(dta_mt1))

        dta_stereo = "(test (song (tracks ((drum (0 1)) (bass (0 1))))))"
        self.assertFalse(confile.is_con_multitrack(dta_stereo))

        dta_single = "(test (song (tracks ((guitar (0 1))))))"
        self.assertFalse(confile.is_con_multitrack(dta_single))

    def test_process_con_song_inplace(self):
        import subprocess
        import tempfile
        import types
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            con_path = self._create_synthetic_con_file(tmp)
            dest = Path(tmp) / "testsong"

            def fake_separate(stage, **kwargs):
                outputs_dir = stage / "outputs"
                for name in confile.OUTPUTS_MOGG_ORDER:
                    stem = outputs_dir / name
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-ac", "2", str(stem)
                    ], check=True, capture_output=True)
                return {"validation": {"peak_error": 0.0}}

            from unittest.mock import patch
            audio = types.ModuleType("yams.audio")
            audio.separate = fake_separate
            with patch.object(confile, "render_stereo_from_mogg"), \
                 patch.dict("sys.modules", {"yams.audio": audio}):
                result = confile.process_con_song(
                    con_path=con_path,
                    destination=dest,
                    mode="inplace",
                    models=Path(tmp),
                    inference_source=Path(tmp),
                    device="cpu",
                    emit=lambda *a, **k: None,
                )

            self.assertEqual(result["validation"]["peak_error"], 0.0)
            self.assertFalse(con_path.exists())
            self.assertTrue((dest / "backup" / "testsong.con").exists())
            self.assertTrue((dest / "backup" / "separation-manifest.json").exists())
            dta_text = (dest / "songs/songs.dta").read_text()
            self.assertIn("drum (0 1 2 3 4 5)", dta_text)
            self.assertIn("keys (12 13)", dta_text)
            mogg_data = (dest / "songs/testsong/testsong.mogg").read_bytes()
            v, start = struct.unpack("<II", mogg_data[:8])
            self.assertEqual(v, 10)
            self.assertEqual(start, 8)

    def test_process_con_song_export(self):
        import subprocess
        import tempfile
        import types
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            con_path = self._create_synthetic_con_file(tmp)
            export_dest = Path(tmp) / "exported_testsong"

            def fake_separate(stage, **kwargs):
                outputs_dir = stage / "outputs"
                for name in confile.OUTPUTS_MOGG_ORDER:
                    stem = outputs_dir / name
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-ac", "2", str(stem)
                    ], check=True, capture_output=True)
                return {"validation": {"peak_error": 0.0}}

            from unittest.mock import patch
            audio = types.ModuleType("yams.audio")
            audio.separate = fake_separate
            with patch.object(confile, "render_stereo_from_mogg"), \
                 patch.dict("sys.modules", {"yams.audio": audio}):
                result = confile.process_con_song(
                    con_path=con_path,
                    destination=export_dest,
                    mode="export",
                    models=Path(tmp),
                    inference_source=Path(tmp),
                    device="cpu",
                    emit=lambda *a, **k: None,
                )

            # In export mode, source CON remains completely untouched
            self.assertTrue(con_path.exists())
            self.assertFalse((export_dest / "separation-manifest.json").exists())
            self.assertTrue((export_dest / "songs/songs.dta").exists())
            self.assertTrue((export_dest / "songs/testsong/testsong.mogg").exists())

    def test_abort_con_process(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            con_path = Path(tmp) / "song.con"
            con_path.write_bytes(b"CON test")
            dest = Path(tmp) / "song"
            dest.mkdir()
            (dest / ".yams-work").mkdir()
            (dest / ".yams-work" / "temp.txt").write_text("work")
            (dest / "songs.dta").write_text("dta")

            confile.abort_con_process(dest, con_path, mode="inplace")
            self.assertTrue(con_path.exists())
            self.assertTrue((dest / ".yams-work" / "temp.txt").exists())
            self.assertEqual((dest / "songs.dta").read_text(), "dta")

    def test_recover_transaction_restores_missing_con_without_hardlinks(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "song.con"
            destination = root / "song"
            destination.mkdir()
            stage = destination / ".yams-work"
            backup = stage / "payload" / "backup" / source.name
            backup.parent.mkdir(parents=True)
            backup.write_bytes(b"original CON")
            write_record(stage, {
                "version": 1,
                "format": "con_file",
                "state": "processing",
                "source": str(source.resolve()),
                "source_sha256": digest(backup),
                "source_backup": f"backup/{source.name}",
                "remove_source": True,
                "mode": "inplace",
                "destination": str(destination.resolve()),
                "files": {},
            })

            with patch("os.link", side_effect=OSError("hard links unavailable")):
                self.assertTrue(recover_transaction(source))

            self.assertEqual(source.read_bytes(), b"original CON")
            self.assertFalse(stage.exists())

    def test_abort_excon_retains_modified_sources_when_backup_hash_fails(self):
        import tempfile
        import types
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "excon_song"
            mogg_path = folder / "songs" / "testsong" / "testsong.mogg"
            dta_path = folder / "songs" / "songs.dta"
            mogg_path.parent.mkdir(parents=True)
            dta_path.parent.mkdir(parents=True, exist_ok=True)
            original_mogg = struct.pack("<II", 10, 8) + b"OggS"
            original_dta = '(testsong (song (name "songs/testsong/testsong") (tracks ((drum (0 1))))))\n'
            mogg_path.write_bytes(original_mogg)
            dta_path.write_text(original_dta, encoding="utf-8")

            def fail_after_mutating_sources(stage, **kwargs):
                mogg_path.write_bytes(b"changed mogg")
                dta_path.write_text("changed dta", encoding="utf-8")
                raise RuntimeError("inference failed")

            audio = types.ModuleType("yams.audio")
            audio.separate = fail_after_mutating_sources
            with patch.object(confile, "render_stereo_from_mogg"), \
                 patch.dict("sys.modules", {"yams.audio": audio}):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    confile.process_excon_song(
                        excon_path=folder,
                        destination=folder,
                        mode="inplace",
                        models=root,
                        inference_source=root,
                        device="cpu",
                        emit=lambda *args, **kwargs: None,
                    )

            (folder / "backup" / mogg_path.name).write_bytes(b"corrupted backup")
            with self.assertRaisesRegex(ValueError, "backup changed"):
                confile.abort_excon_process(folder, mode="inplace")

            self.assertEqual(mogg_path.read_bytes(), b"changed mogg")
            self.assertEqual(dta_path.read_text(encoding="utf-8"), "changed dta")
            self.assertTrue((folder / ".yams-work").is_dir())

    def test_abort_excon_without_journal_preserves_changed_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "excon"
            backup = folder / "backup"
            backup.mkdir(parents=True)
            (backup / "song.mogg").write_bytes(b"original")
            source = folder / "song.mogg"
            source.write_bytes(b"user data written later")

            with self.assertRaisesRegex(ValueError, "source changed"):
                confile.abort_excon_process(folder, mode="inplace")

            self.assertEqual(source.read_bytes(), b"user data written later")

    def test_process_con_export_failure_preserves_destination(self):
        import tempfile
        import types
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            con_path = root / "song.con"
            con_path.write_bytes(b"CON test")
            destination = root / "export"
            destination.mkdir()
            existing = destination / "notes.txt"
            existing.write_text("keep", encoding="utf-8")

            def unpack(con_path, package):
                dta_path = package / "songs" / "songs.dta"
                mogg_path = package / "songs" / "testsong" / "testsong.mogg"
                dta_path.parent.mkdir(parents=True)
                mogg_path.parent.mkdir(parents=True)
                dta_path.write_text("dta", encoding="utf-8")
                mogg_path.write_bytes(b"mogg")
                return {"dta_path": dta_path, "mogg_path": mogg_path}

            def fail_separation(stage, **kwargs):
                raise RuntimeError("inference failed")

            audio = types.ModuleType("yams.audio")
            audio.separate = fail_separation
            with patch.object(confile, "unpack_con_song", side_effect=unpack), \
                 patch.object(confile, "is_con_multitrack", return_value=False), \
                 patch.object(confile, "render_stereo_from_mogg"), \
                 patch.dict("sys.modules", {"yams.audio": audio}):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    try:
                        confile.process_con_song(
                            con_path=con_path,
                            destination=destination,
                            mode="export",
                            models=root,
                            inference_source=root,
                            device="cpu",
                            emit=lambda *args, **kwargs: None,
                        )
                    except RuntimeError:
                        confile.abort_con_process(destination, con_path, mode="export")
                        raise

            self.assertTrue(con_path.is_file())
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
            self.assertFalse((destination / "songs").exists())
            self.assertFalse((destination / ".yams-work").exists())


    def test_is_excon_folder(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "excon"
            folder.mkdir()
            self.assertFalse(confile.is_excon_folder(folder))

            (folder / "songs.dta").write_text("(test (name \"Test\"))")
            self.assertFalse(confile.is_excon_folder(folder))

            (folder / "song.mogg").write_bytes(b"\x00" * 8)
            self.assertTrue(confile.is_excon_folder(folder))

            # Not an excon if it has song.ini
            (folder / "song.ini").write_text("[song]\nname=Test")
            self.assertFalse(confile.is_excon_folder(folder))

    def test_excon_export_refuses_to_replace_existing_destination_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = create_synthetic_con(root)
            unpacked = root / "unpacked"
            confile.unpack_con_song(source, unpacked)
            destination = root / "destination"
            existing = destination / "songs" / "songs.dta"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"user data")

            with self.assertRaises(FileExistsError):
                confile.process_excon_song(
                    excon_path=unpacked,
                    destination=destination,
                    mode="export",
                    models=root,
                    inference_source=root,
                    device="cpu",
                    emit=lambda *args, **kwargs: None,
                )

            self.assertEqual(existing.read_bytes(), b"user data")
            self.assertEqual([path.relative_to(destination).as_posix()
                              for path in destination.rglob("*") if path.is_file()], ["songs/songs.dta"])

            nested_destination = unpacked / "export"
            with self.assertRaisesRegex(ValueError, "inside its source"):
                confile.process_excon_song(
                    excon_path=unpacked,
                    destination=nested_destination,
                    mode="export",
                    models=root,
                    inference_source=root,
                    device="cpu",
                    emit=lambda *args, **kwargs: None,
                )
            self.assertFalse(nested_destination.exists())

    def test_find_excon_info(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "song1"
            folder.mkdir()
            (folder / "songs.dta").write_text("(song1 (name \"Song 1\") (song (name \"song1\")))")
            (folder / "song1.mogg").write_bytes(b"\x00" * 8)
            (folder / "song1.mid").write_bytes(b"MThd" + b"\x00" * 10)

            info = confile.find_excon_info(folder)
            self.assertEqual(info["song_id"], "song1")
            self.assertEqual(info["sub_name"], "song1")
            self.assertEqual(info["mogg_path"].name, "song1.mogg")
            self.assertEqual(info["midi_path"].name, "song1.mid")

    def test_process_excon_song_inplace(self):
        import subprocess
        import tempfile
        import types
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "excon_song"
            folder.mkdir()

            # Create unencrypted version 10 MOGG with 2 audio channels
            temp_ogg = folder / "temp.ogg"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1",
                "-c:a", "libvorbis", "-q:a", "4", str(temp_ogg)
            ], check=True, capture_output=True)
            ogg_data = temp_ogg.read_bytes()
            temp_ogg.unlink()
            mogg_path = folder / "song.mogg"
            mogg_path.write_bytes(struct.pack("<II", 10, 8) + ogg_data)

            dta_path = folder / "songs.dta"
            dta_path.write_text("""
            (testsong
              (name "Test Song")
              (song
                (name "testsong")
                (tracks ((drum (0 1))))
                (pans (-1.0 1.0))
                (vols (0.0 0.0))
              )
            )
            """)

            def fake_separate(stage, **kwargs):
                outputs_dir = stage / "outputs"
                for name in confile.OUTPUTS_MOGG_ORDER:
                    stem = outputs_dir / name
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-ac", "2", str(stem)
                    ], check=True, capture_output=True)
                return {"validation": {"peak_error": 0.0}}

            audio = types.ModuleType("yams.audio")
            audio.separate = fake_separate
            with patch.object(confile, "render_stereo_from_mogg"), \
                 patch.dict("sys.modules", {"yams.audio": audio}):
                result = confile.process_excon_song(
                    excon_path=folder,
                    destination=folder,
                    mode="inplace",
                    models=Path(tmp),
                    inference_source=Path(tmp),
                    device="cpu",
                    emit=lambda *a, **k: None,
                )

            # Inplace mode: original backed up, mogg updated to 16 channels, manifest written
            self.assertTrue((folder / "backup" / "song.mogg").exists())
            self.assertTrue((folder / "backup" / "separation-manifest.json").exists())
            mogg_bytes = mogg_path.read_bytes()
            v, start = struct.unpack("<II", mogg_bytes[:8])
            self.assertEqual(v, 10)
            self.assertEqual(start, 8)
            self.assertIn("tracks", dta_path.read_text())

    def test_abort_excon_restores_nested_original_paths(self):
        import tempfile
        import types
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "excon_song"
            mogg_path = folder / "songs" / "testsong" / "testsong.mogg"
            dta_path = folder / "songs" / "songs.dta"
            mogg_path.parent.mkdir(parents=True)
            dta_path.parent.mkdir(parents=True, exist_ok=True)
            original_mogg = struct.pack("<II", 10, 8) + b"OggS"
            original_dta = '(testsong (song (name "songs/testsong/testsong") (tracks ((drum (0 1))))))\n'
            mogg_path.write_bytes(original_mogg)
            dta_path.write_text(original_dta)

            def fail_after_mutating_sources(stage, **kwargs):
                mogg_path.write_bytes(b"partial mogg")
                dta_path.write_text("partial dta")
                raise RuntimeError("inference failed")

            audio = types.ModuleType("yams.audio")
            audio.separate = fail_after_mutating_sources
            with patch.object(confile, "render_stereo_from_mogg"), \
                 patch.dict("sys.modules", {"yams.audio": audio}):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    confile.process_excon_song(
                        excon_path=folder,
                        destination=folder,
                        mode="inplace",
                        models=root,
                        inference_source=root,
                        device="cpu",
                        emit=lambda *args, **kwargs: None,
                    )

            confile.abort_excon_process(folder, mode="inplace")
            self.assertEqual(mogg_path.read_bytes(), original_mogg)
            self.assertEqual(dta_path.read_text(), original_dta)
            self.assertFalse((folder / "testsong.mogg").exists())
            self.assertFalse((folder / "songs.dta").exists())
            self.assertFalse((folder / ".yams-work").exists())

    def test_single_quoted_dta_parsing_and_patching(self):
        quoted_dta = """
        (
           'petergabrielinyoureyes_rb'
           (
              'name'
              "In Your Eyes"
           )
           (
              'artist'
              "Peter Gabriel"
           )
           ('master' 1)
           (
              'song'
              (
                 'name'
                 "songs/petergabrielinyoureyes/petergabrielinyoureyes"
              )
              (
                 'tracks_count'
                 (2 2 2 2 2 2)
              )
              (
                 'tracks'
                 (
                    (
                       'drum'
                       (0 1)
                    )
                    (
                       'bass'
                       (2 3)
                    )
                 )
              )
              (
                 'pans'
                 (-1.00 1.00 -1.00 1.00)
              )
              (
                 'vols'
                 (0.00 0.00 0.00 0.00)
              )
           )
        )
        """
        tree = confile.parse_dta(quoted_dta)
        info = confile.extract_audio_info_from_dta(tree)
        self.assertEqual(info["song_id"], "petergabrielinyoureyes_rb")
        self.assertEqual(info["name"], "songs/petergabrielinyoureyes/petergabrielinyoureyes")
        self.assertEqual(info["pans"], [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(info["vols"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(info["tracks"]["drum"], [0, 1])
        self.assertEqual(info["tracks"]["bass"], [2, 3])

        patched = confile.patch_dta_audio(quoted_dta)
        self.assertIn("(tracks_count (6 2 2 2 2 2))", patched)
        self.assertIn("(drum (0 1 2 3 4 5))", patched)
        self.assertIn("(bass (6 7))", patched)
        self.assertIn("(guitar (8 9))", patched)
        self.assertIn("(vocals (10 11))", patched)
        self.assertIn("(keys (12 13))", patched)
        self.assertIn("(cores (-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1))", patched)


class MoggPackagingTests(unittest.TestCase):
    def test_pcm_packaging_for_both_formats_and_modes(self):
        import io
        import json
        import tempfile
        from itertools import product
        from pathlib import Path
        from unittest.mock import patch

        import numpy as np
        import soundfile as sf

        rate = 44100
        frames = rate // 4
        timeline = np.arange(frames) / rate
        frequencies = np.arange(16) * 100 + 400
        audio = 0.1 * np.sin(2 * np.pi * timeline[:, None] * frequencies)
        pcm_validation = {"peak_error": 0.0, "files": {"song.wav": {"sha256": "pcm"}}}

        def separate(stage, **kwargs):
            pcm = stage / stem_format
            if stem_format == "pcm":
                pcm.mkdir()
            for index, name in enumerate(confile.OUTPUTS_MOGG_ORDER):
                sf.write(file=pcm / name, data=audio[:, 2 * index:2 * index + 2],
                         samplerate=rate, subtype="FLOAT")
            if stem_format == "pcm":
                return {"pcm_validation": pcm_validation, "validation": {"codec": "opus"}}
            return {"validation": pcm_validation}

        for source_format, mode, stem_format in product(("con", "excon"), ("inplace", "export"), ("pcm", "outputs")):
            with self.subTest(source_format=source_format, mode=mode, stem_format=stem_format), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = create_synthetic_con(tmp)
                original = source.read_bytes()
                destination = root / "destination"
                if source_format == "con":
                    process = confile.process_con_song
                    arguments = {"con_path": source}
                else:
                    unpacked = root / "unpacked"
                    confile.unpack_con_song(source, unpacked)
                    original_mogg = next(unpacked.rglob("*.mogg")).read_bytes()
                    process = confile.process_excon_song
                    arguments = {"excon_path": unpacked}
                    if mode == "inplace":
                        destination = unpacked
                with patch("yams.audio.separate", side_effect=separate):
                    result = process(**arguments, destination=destination, mode=mode, models=root,
                                     inference_source=root, device="cpu",
                                     emit=lambda *args, **kwargs: None)
                self.assertEqual(result["validation"], pcm_validation)
                mogg = next(destination.glob("songs/**/*.mogg"))
                data = mogg.read_bytes()
                self.assertEqual(struct.unpack("<II", data[:8]), (10, 8))
                decoded, decoded_rate = sf.read(file=io.BytesIO(confile.demux_mogg(data)),
                                               always_2d=True)
                self.assertEqual(decoded_rate, rate)
                self.assertEqual(decoded.shape, audio.shape)
                peaks = np.argmax(np.abs(np.fft.rfft(decoded, axis=0)), axis=0)
                np.testing.assert_array_equal(peaks * rate / frames, frequencies)
                info = confile.extract_audio_info_from_dta(confile.parse_dta(
                    (destination / "songs" / "songs.dta").read_text()))
                self.assertEqual(info["tracks"], {"drum": list(range(6)), "bass": [6, 7],
                                                 "guitar": [8, 9], "vocals": [10, 11], "keys": [12, 13]})
                self.assertEqual(info["pans"], [-1.0, 1.0] * 8)
                if mode == "inplace":
                    manifest = destination / "backup" / "separation-manifest.json"
                    self.assertEqual(json.loads(manifest.read_text())["validation"], pcm_validation)
                else:
                    self.assertFalse((destination / "separation-manifest.json").exists())
                self.assertFalse((destination / ".yams-work").exists())
                if source_format == "con" and mode == "inplace":
                    self.assertEqual((destination / "backup" / source.name).read_bytes(), original)
                else:
                    self.assertEqual(source.read_bytes(), original)
                if source_format == "excon":
                    preserved = (destination / "backup" / mogg.name) if mode == "inplace" else next(unpacked.rglob("*.mogg"))
                    self.assertEqual(preserved.read_bytes(), original_mogg)

    def test_pcm_packaging_6stem(self):
        import io
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import numpy as np
        import soundfile as sf

        rate = 44100
        frames = rate // 4
        timeline = np.arange(frames) / rate
        frequencies = np.arange(12) * 100 + 400
        audio = 0.1 * np.sin(2 * np.pi * timeline[:, None] * frequencies)
        pcm_validation = {"peak_error": 0.0, "files": {"song.wav": {"sha256": "pcm"}}}

        def separate(stage, **kwargs):
            pcm = stage / "pcm"
            pcm.mkdir(parents=True, exist_ok=True)
            for index, name in enumerate(confile.OUTPUTS_MOGG_ORDER_6STEM):
                sf.write(file=pcm / name, data=audio[:, 2 * index:2 * index + 2],
                         samplerate=rate, subtype="FLOAT")
            return {"pcm_validation": pcm_validation, "validation": {"codec": "opus"}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = create_synthetic_con(tmp)
            destination = root / "destination"
            with patch("yams.audio.separate", side_effect=separate):
                result = confile.process_con_song(
                    con_path=source, destination=destination, mode="export",
                    models=root, inference_source=root, device="cpu",
                    emit=lambda *args, **kwargs: None,
                    model_profile="torch-6stem",
                )
            self.assertEqual(result["validation"], pcm_validation)
            mogg = next(destination.glob("songs/**/*.mogg"))
            data = mogg.read_bytes()
            self.assertEqual(struct.unpack("<II", data[:8]), (10, 8))
            decoded, decoded_rate = sf.read(file=io.BytesIO(confile.demux_mogg(data)),
                                           always_2d=True)
            self.assertEqual(decoded_rate, rate)
            self.assertEqual(decoded.shape, audio.shape)
            peaks = np.argmax(np.abs(np.fft.rfft(decoded, axis=0)), axis=0)
            np.testing.assert_array_equal(peaks * rate / frames, frequencies)
            info = confile.extract_audio_info_from_dta(confile.parse_dta(
                (destination / "songs" / "songs.dta").read_text()))
            self.assertEqual(info["tracks"], {
                "drum": [0, 1],
                "bass": [2, 3],
                "guitar": [4, 5],
                "vocals": [6, 7],
                "keys": [8, 9],
            })
            self.assertEqual(info["pans"], [-1.0, 1.0] * 6)


if __name__ == "__main__":
    unittest.main()
