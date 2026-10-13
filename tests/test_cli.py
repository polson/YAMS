import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yams.__main__ import main
from yams import folders
from yams.folders import OUTPUTS, digest


class CommandTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "song.ini").write_text("[song]\n")
        (folder / "notes.mid").write_bytes(b"chart")
        (folder / "song.ogg").write_bytes(b"original")
        return folder

    def command(self, folder):
        return ["yams", str(folder), "--models", str(folder), "--inference-source", str(folder)]

    def test_cli_runs_without_explicit_inference_source_using_vendored_source(self):
        folder = self.fixture()
        output = io.StringIO()
        cmd = ["yams", str(folder), "--models", str(folder)]
        captured_source = None

        def fake_separate(stage, models, inference_source=None, **kwargs):
            nonlocal captured_source
            captured_source = inference_source
            raise KeyboardInterrupt

        with patch.object(sys, "argv", cmd), contextlib.redirect_stdout(output), \
             patch("yams.audio.separate", side_effect=fake_separate):
            status = main()
        self.assertEqual(status, 130)
        from yams.audio import DEFAULT_INFERENCE
        self.assertEqual(captured_source, DEFAULT_INFERENCE)

    def test_cancel_rolls_back_and_reports_130(self):
        folder = self.fixture()
        output = io.StringIO()
        with patch.object(sys, "argv", self.command(folder)), contextlib.redirect_stdout(output), \
             patch("yams.audio.separate", side_effect=KeyboardInterrupt):
            status = main()
        self.assertEqual(status, 130)
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["event"], "cancelled")
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / ".yams-work").exists())

    def test_model_failure_rolls_back_and_reports_error(self):
        folder = self.fixture()
        output = io.StringIO()
        with patch.object(sys, "argv", self.command(folder)), contextlib.redirect_stdout(output), \
             patch("yams.audio.separate", side_effect=ValueError("invalid prediction")):
            status = main()
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["event"], "error")
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / ".yams-work").exists())

    def test_recovered_commit_is_success_and_stdout_is_json(self):
        folder = self.fixture()
        output = io.StringIO()
        diagnostics = io.StringIO()
        remove_stage = folders.remove_stage
        cleanups = []

        def interrupted_cleanup(stage):
            cleanups.append(stage)
            if len(cleanups) == 1:
                raise OSError("interrupted cleanup")
            remove_stage(stage)

        def separate(stage, **kwargs):
            print("upstream diagnostic")
            for name in OUTPUTS:
                (stage / "outputs" / name).write_bytes(name.encode())
            files = {name: {"sha256": digest(stage / "outputs" / name)} for name in OUTPUTS}
            return {"validation": {"peak_error": 0, "files": files}}

        with patch.object(sys, "argv", self.command(folder)), contextlib.redirect_stdout(output), \
             contextlib.redirect_stderr(diagnostics), patch("yams.audio.separate", side_effect=separate), \
             patch("yams.folders.remove_stage", side_effect=interrupted_cleanup):
            status = main()
        self.assertEqual(status, 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["event"], "completed")
        self.assertEqual((folder / "backup/song.ogg").read_bytes(), b"original")
        self.assertIn("upstream diagnostic", diagnostics.getvalue())
        self.assertFalse((folder / ".yams-work").exists())

    def test_device_argument_passed_to_separate(self):
        folder = self.fixture()
        output = io.StringIO()
        calls = []

        def separate(stage, device_name, **kwargs):
            calls.append(device_name)
            for name in OUTPUTS:
                (stage / "outputs" / name).write_bytes(name.encode())
            files = {name: {"sha256": digest(stage / "outputs" / name)} for name in OUTPUTS}
            return {"validation": {"peak_error": 0, "files": files}}

        with patch.object(sys, "argv", [*self.command(folder), "--device", "mps"]), \
             contextlib.redirect_stdout(output), patch("yams.audio.separate", side_effect=separate):
            status = main()
        self.assertEqual(status, 0)
        self.assertEqual(calls, ["mps"])

    def test_interrupted_work_is_automatically_recovered_on_run(self):
        folder = self.fixture()
        folders.begin(folder, folder / "song.ogg")
        output = io.StringIO()

        def separate(stage, **kwargs):
            for name in OUTPUTS:
                (stage / "outputs" / name).write_bytes(name.encode())
            files = {name: {"sha256": digest(stage / "outputs" / name)} for name in OUTPUTS}
            return {"validation": {"peak_error": 0, "files": files}}

        with patch.object(sys, "argv", self.command(folder)), contextlib.redirect_stdout(output), \
             patch("yams.audio.separate", side_effect=separate):
            status = main()
        self.assertEqual(status, 0)
        self.assertEqual((folder / "backup/song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / ".yams-work").exists())
        for name in OUTPUTS:
            self.assertTrue((folder / name).is_file())

    def test_audio_recovery_waits_for_processing_lock(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        audio = root / "track.opus"
        audio.write_bytes(b"original")
        destination = root / "track"
        stage = destination / ".yams-work"
        stage.mkdir(parents=True)
        output = io.StringIO()

        with folders.locked(destination), patch.object(sys, "argv", ["yams", str(audio), "--recover"]), \
             contextlib.redirect_stdout(output):
            status = main()
        self.assertEqual(status, 1)
        self.assertTrue(stage.exists())
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["event"], "error")

        output = io.StringIO()
        with patch.object(sys, "argv", ["yams", str(audio), "--recover"]), contextlib.redirect_stdout(output):
            status = main()
        self.assertEqual(status, 0)
        self.assertFalse(stage.exists())
        self.assertEqual(audio.read_bytes(), b"original")

    def test_con_file_inspect_cli(self):
        try:
            from test_confile import create_synthetic_con
        except ImportError:
            from tests.test_confile import create_synthetic_con
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp)
            output = io.StringIO()
            with patch.object(sys, "argv", ["yams", str(con_path), "--inspect"]), contextlib.redirect_stdout(output):
                status = main()
            self.assertEqual(status, 0)
            event = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(event["event"], "preview")
            self.assertEqual(event["status"], "ready")
            self.assertEqual(event["source"], "testsong.con")

    def test_con_file_inplace_cli(self):
        try:
            from test_confile import create_synthetic_con
        except ImportError:
            from tests.test_confile import create_synthetic_con
        import subprocess
        import tempfile
        from yams import confile
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp)
            sibling = Path(tmp) / "testsong"
            output = io.StringIO()

            def fake_separate(stage, **kwargs):
                outputs_dir = stage / "pcm"
                outputs_dir.mkdir()
                for name in confile.OUTPUTS_MOGG_ORDER:
                    stem = outputs_dir / name
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-ac", "2", str(stem)
                    ], check=True, capture_output=True)
                return {"pcm_validation": {"peak_error": 0.0}, "validation": {"codec": "opus"}}

            with patch.object(sys, "argv", self.command(con_path)), contextlib.redirect_stdout(output), \
                 patch("yams.audio.separate", side_effect=fake_separate):
                status = main()
            self.assertEqual(status, 0)
            self.assertFalse(con_path.exists())
            self.assertTrue((sibling / "backup" / "testsong.con").exists())
            self.assertTrue((sibling / "songs/songs.dta").exists())
            self.assertTrue((sibling / "songs/testsong/testsong.mogg").exists())

    def test_con_file_export_cli(self):
        try:
            from test_confile import create_synthetic_con
        except ImportError:
            from tests.test_confile import create_synthetic_con
        import subprocess
        import tempfile
        from yams import confile
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp)
            export_target = Path(tmp) / "exported_song"
            output = io.StringIO()

            def fake_separate(stage, **kwargs):
                outputs_dir = stage / "pcm"
                outputs_dir.mkdir()
                for name in confile.OUTPUTS_MOGG_ORDER:
                    stem = outputs_dir / name
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-ac", "2", str(stem)
                    ], check=True, capture_output=True)
                return {"pcm_validation": {"peak_error": 0.0}, "validation": {"codec": "opus"}}

            cmd = ["yams", str(con_path), "--output-folder", str(export_target),
                   "--models", str(con_path), "--inference-source", str(con_path)]
            with patch.object(sys, "argv", cmd), contextlib.redirect_stdout(output), \
                 patch("yams.audio.separate", side_effect=fake_separate):
                status = main()
            self.assertEqual(status, 0)
            self.assertTrue(con_path.exists())
            self.assertFalse((export_target / "separation-manifest.json").exists())
            self.assertTrue((export_target / "songs/songs.dta").exists())
            self.assertTrue((export_target / "songs/testsong/testsong.mogg").exists())

    def test_multitrack_con_is_skipped(self):
        try:
            from test_confile import create_synthetic_con
        except ImportError:
            from tests.test_confile import create_synthetic_con
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            con_path = create_synthetic_con(tmp, dta_extra=";Multitrack=1\n")
            output = io.StringIO()
            cmd = ["yams", str(con_path)]
            with patch.object(sys, "argv", cmd), contextlib.redirect_stdout(output):
                status = main()
            self.assertEqual(status, 0)
            events = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
            self.assertEqual(events[-1]["event"], "skipped")
            self.assertEqual(events[-1]["reason"], "already_contains_multitrack_stems")

    def test_torch_6stem_profile_passed_to_separate(self):
        folder = self.fixture()
        output = io.StringIO()
        cmd = ["yams", str(folder), "--models", str(folder), "--model-profile", "torch-6stem"]
        captured_profile = None

        def fake_separate(stage, models, model_profile="mega-8stem", **kwargs):
            nonlocal captured_profile
            captured_profile = model_profile
            raise KeyboardInterrupt

        with patch.object(sys, "argv", cmd), contextlib.redirect_stdout(output), \
             patch("yams.audio.separate", side_effect=fake_separate):
            status = main()
        self.assertEqual(status, 130)
        self.assertEqual(captured_profile, "torch-6stem")

    def test_torch_6stem_accepted_on_con_package(self):
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

            cmd = ["yams", str(con_path), "--models", str(con_path), "--model-profile", "torch-6stem"]
            with patch.object(sys, "argv", cmd), contextlib.redirect_stdout(output), \
                 patch("yams.audio.separate", side_effect=fake_separate):
                status = main()
            self.assertEqual(status, 0)
            self.assertEqual(captured_profile, "torch-6stem")
            dta_text = (sibling / "songs/songs.dta").read_text()
            self.assertIn("drum (0 1)", dta_text)
            self.assertIn("bass (2 3)", dta_text)
            self.assertIn("guitar (4 5)", dta_text)
            self.assertIn("vocals (6 7)", dta_text)
            self.assertIn("keys (8 9)", dta_text)


if __name__ == "__main__":
    unittest.main()
