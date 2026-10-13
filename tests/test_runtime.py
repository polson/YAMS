import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from yams.runtime import ensure_runtime, run_setup, torch_backend


class RuntimeTests(unittest.TestCase):
    def test_runtime_packages_follow_gpu_zero_capability_and_driver(self):
        with patch("yams.runtime.platform.machine", return_value="x86_64"):
            for system, driver, capability, expected in [
                ("linux", "580.82.09", "8.6", "cu126"),
                ("linux", "580.82.09", "12.0", "cu128"),
                ("linux", "550.54.15", "8.6", "cu118"),
                ("linux", "450.00", "8.6", "cpu"),
                ("win32", "572.61", "12.0", "cu128"),
                ("win32", "560.94", "8.6", "cu126"),
                ("win32", "522.06", "8.6", "cu118"),
            ]:
                with self.subTest(system=system, driver=driver, capability=capability):
                    result = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"{driver}, {capability}\n")
                    with patch.object(sys, "platform", system), patch("yams.runtime.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True), patch("yams.runtime.subprocess.run", return_value=result) as query:
                        self.assertEqual(torch_backend(), expected)
                        self.assertIn("--id=0", query.call_args.kwargs["args"])
            with patch.object(sys, "platform", "linux"), patch("yams.runtime.subprocess.run", side_effect=FileNotFoundError):
                self.assertEqual(torch_backend(), "cpu")
    def test_successful_setup_is_reused_and_requirements_changes_invalidate_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "requirements.txt").write_text("dependency==1\n")
            (root / "check_runtime.py").write_text("pass\n")
            with patch.object(sys, "prefix", str(runtime)), patch("yams.runtime.torch_backend", return_value="cpu"), patch("yams.runtime.run_setup") as setup:
                ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 2)
                self.assertTrue((runtime / ".yams-ready").is_file())
                ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 2)
                (root / "requirements.txt").write_text("dependency==2\n")
                ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 4)

    def test_failed_verification_does_not_publish_ready_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "requirements.txt").write_text("dependency==1\n")
            (root / "check_runtime.py").write_text("pass\n")
            with patch.object(sys, "prefix", str(runtime)), patch("yams.runtime.torch_backend", return_value="cpu"), patch("yams.runtime.run_setup", side_effect=[None, RuntimeError("verification failed")]):
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
            self.assertFalse((runtime / ".yams-ready").exists())

    def test_cancellation_terminates_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid = root / "pid"
            code = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            cancelled = threading.Event()
            timer = threading.Timer(.5, cancelled.set)
            timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    run_setup(arguments=[sys.executable, "-c", code, str(pid)], cache=root, cancelled=cancelled)
            finally:
                timer.join()
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_runtime_failure_leaves_song_untouched_and_preview_recovery_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            song = Path(temporary)
            (song / "song.ini").write_text("[song]\n")
            (song / "notes.mid").write_bytes(b"chart")
            (song / "song.wav").write_bytes(b"original")
            command = [sys.executable, "-m", "yams", str(song), "--inference-source", str(song),
                       "--runtime-installer", str(song / "missing-installer"), "--runtime-resources", str(song / "missing-resources")]
            failed = subprocess.run(args=command, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn("runtime_error", [json.loads(line)["event"] for line in failed.stdout.splitlines()])
            self.assertFalse((song / ".yams-work").exists())
            self.assertFalse((song / ".yams.lock").exists())
            self.assertEqual((song / "song.wav").read_bytes(), b"original")
            preview = subprocess.run(args=[*command, "--inspect"], capture_output=True, text=True)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(json.loads(preview.stdout)["status"], "ready")
            (song / ".yams-work").mkdir()
            recovery = subprocess.run(args=[*command, "--recover"], capture_output=True, text=True)
            self.assertEqual(recovery.returncode, 0, recovery.stderr)
            self.assertEqual(json.loads(recovery.stdout)["event"], "recovered")
            self.assertEqual((song / "song.wav").read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
