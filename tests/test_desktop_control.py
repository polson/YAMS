import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from yams.__main__ import main
from yams.folders import OUTPUTS, digest


class DesktopControlTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "song.ini").write_text("[song]\n")
        (folder / "notes.mid").write_bytes(b"chart")
        (folder / "song.ogg").write_bytes(b"original")
        return folder

    def test_inspection_is_read_only_without_ml(self):
        folder = self.fixture()
        before = {path.name: path.read_bytes() for path in folder.iterdir()}
        result = subprocess.run([sys.executable, "-m", "yams", str(folder), "--inspect"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual(preview["status"], "ready")
        self.assertEqual(preview["backup"], "backup/song.ogg")
        self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir()})
        (folder / ".yams-work").mkdir()
        result = subprocess.run([sys.executable, "-m", "yams", str(folder), "--inspect"], capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["status"], "recovery_required")

    def test_control_eof_cancels_before_commit(self):
        folder = self.fixture()
        argv = [sys.executable, "-m", "yams", str(folder), "--models", str(folder),
                "--inference-source", str(folder), "--control-stdin"]
        result = subprocess.run(argv, input="", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / ".yams-work").exists())
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["event"], "cancelled")

    def test_cancel_during_inference_restores_original(self):
        folder = self.fixture()
        output = io.StringIO()
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "r") as control, os.fdopen(write_fd, "w") as writer:
            argv = ["yams", str(folder), "--models", str(folder), "--inference-source", str(folder), "--control-stdin"]

            def separate(**kwargs):
                self.assertTrue((folder / ".yams-work/original").exists())
                writer.write("cancel\n")
                writer.flush()
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    kwargs["emit"]("model_progress", target="vocal", pass_number=1, chunk=1)
                    time.sleep(.001)
                self.fail("Control message was not received")

            with patch.object(sys, "argv", argv), patch.object(sys, "stdin", control), \
                 contextlib.redirect_stdout(output), patch("yams.audio.separate", side_effect=separate):
                status = main()
        self.assertEqual(status, 130)
        self.assertEqual((folder / "song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / ".yams-work").exists())
        events = [json.loads(line)["event"] for line in output.getvalue().splitlines()]
        self.assertEqual(events[-2:], ["recovered", "cancelled"])

    def test_cancel_during_installation_finishes_commit(self):
        folder = self.fixture()
        output = io.StringIO()
        cancelled = threading.Event()
        argv = ["yams", str(folder), "--models", str(folder), "--inference-source", str(folder)]
        from yams.folders import install

        def separate(stage, **kwargs):
            for name in OUTPUTS:
                (stage / "outputs" / name).write_bytes(name.encode())
            files = {name: {"sha256": digest(stage / "outputs" / name)} for name in OUTPUTS}
            return {"validation": {"peak_error": 0, "files": files}}

        def cancel_and_install(*args):
            cancelled.set()
            install(*args)

        with patch.object(sys, "argv", argv), patch("yams.__main__.threading.Event", return_value=cancelled), \
             contextlib.redirect_stdout(output), patch("yams.audio.separate", side_effect=separate), \
             patch("yams.__main__.install", side_effect=cancel_and_install):
            status = main()
        self.assertEqual(status, 0)
        self.assertEqual((folder / "backup/song.ogg").read_bytes(), b"original")
        self.assertFalse((folder / ".yams-work").exists())
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["event"], "completed")


if __name__ == "__main__":
    unittest.main()
