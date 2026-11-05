import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from yams.runtime import MissingVisualCRuntime, ensure_runtime, missing_packages, run_setup, torch_backend, visual_c_runtime_installed


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
                with patch("yams.runtime._has_amd_gpu", return_value=True):
                    self.assertEqual(torch_backend(), "rocm6.2.4")
                with patch("yams.runtime._has_amd_gpu", return_value=False):
                    self.assertEqual(torch_backend(), "cpu")
    def test_successful_setup_is_reused_and_requirements_changes_invalidate_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "requirements.txt").write_text("dependency==1\n")
            (root / "check_runtime.py").write_text("pass\n")
            with patch.object(sys, "prefix", str(runtime)), patch("yams.runtime.torch_backend", return_value="cpu"), patch("yams.runtime.run_setup") as setup, \
                    patch("yams.runtime.missing_packages", return_value=[]):
                ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 2)
                self.assertTrue((runtime / ".yams-ready").is_file())
                ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 2)
                (root / "requirements.txt").write_text("dependency==2\n")
                ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 4)

    def test_missing_package_with_a_current_marker_reinstalls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "requirements.txt").write_text("dependency==1\n")
            (root / "check_runtime.py").write_text("pass\n")
            with patch.object(sys, "prefix", str(runtime)), patch("yams.runtime.torch_backend", return_value="cpu"), patch("yams.runtime.run_setup") as setup:
                with patch("yams.runtime.missing_packages", return_value=[]):
                    ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                with patch("yams.runtime.missing_packages", return_value=["dependency"]), self.assertLogs("yams.runtime", level="WARNING"):
                    ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
                self.assertEqual(setup.call_count, 4)

    def test_missing_packages_lists_pinned_requirements_that_are_not_installed(self):
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("# pinned\ndefinitely-not-installed-yams-test==1.0\n")
            self.assertEqual(missing_packages(requirements), ["definitely-not-installed-yams-test"])

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

    def test_missing_visual_c_runtime_stops_before_installing_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "requirements.txt").write_text("dependency==1\n")
            (root / "check_runtime.py").write_text("pass\n")
            with patch.object(sys, "prefix", str(runtime)), patch("yams.runtime.visual_c_runtime_installed", return_value=False), patch("yams.runtime.run_setup") as setup:
                with self.assertRaisesRegex(MissingVisualCRuntime, "Visual C\\+\\+ Redistributable") as failure:
                    ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())
            self.assertEqual(failure.exception.code, "vc_redist_missing")
            setup.assert_not_called()
            self.assertFalse((runtime / ".yams-ready").exists())

    def test_visual_c_runtime_is_read_from_the_windows_registry(self):
        registry = MagicMock()
        registry.HKEY_LOCAL_MACHINE = object()
        with patch.object(sys, "platform", "win32"), patch("yams.runtime.platform.machine", return_value="AMD64"), patch.dict(sys.modules, {"winreg": registry}):
            registry.QueryValueEx.return_value = (1, 4)
            self.assertTrue(visual_c_runtime_installed())
            registry.QueryValueEx.return_value = (0, 4)
            self.assertFalse(visual_c_runtime_installed())
            registry.OpenKey.side_effect = FileNotFoundError
            self.assertFalse(visual_c_runtime_installed())
        with patch.object(sys, "platform", "linux"):
            self.assertTrue(visual_c_runtime_installed())

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

    def test_installer_output_is_reported_and_kept_for_failures(self):
        code = "import sys; print('Resolved 42 packages'); print('Downloading torch\\r', end=''); print('Downloading torch 2/3\\r'); sys.exit(3)"
        reported = []
        with self.assertRaisesRegex(RuntimeError, "Resolved 42 packages") as failure:
            run_setup(arguments=[sys.executable, "-c", code], cache=Path(tempfile.gettempdir()), cancelled=threading.Event(), emit=reported.append)
        self.assertIn("Downloading torch 2/3", str(failure.exception))
        self.assertEqual(reported[-1], "Downloading torch 2/3")

    def test_child_output_keeps_non_ascii_paths_intact(self):
        code = "import sys; print('C:\\\\Users\\\\José\\\\python.exe'); sys.exit(1)"
        with self.assertRaisesRegex(RuntimeError, "José"):
            run_setup(arguments=[sys.executable, "-c", code], cache=Path(tempfile.gettempdir()), cancelled=threading.Event())

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
            self.assertEqual(recovery.returncode, 1)
            self.assertIn("runtime_error", [json.loads(line)["event"] for line in recovery.stdout.splitlines()])
            self.assertFalse((song / ".yams-work").exists())
            self.assertEqual((song / "song.wav").read_bytes(), b"original")


    def test_unreadable_nvidia_smi_output_falls_back_to_cpu_with_a_warning(self):
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="unexpected output\n")
        with patch.object(sys, "platform", "linux"), patch("yams.runtime.platform.machine", return_value="x86_64"), \
                patch("yams.runtime.subprocess.run", return_value=result):
            with self.assertLogs("yams.runtime", level="WARNING") as logs:
                self.assertEqual(torch_backend(), "cpu")
        self.assertIn("unexpected output", logs.output[0])

    def test_setup_failure_reports_the_exit_status_and_output(self):
        code = "import sys; print('Resolved 42 packages'); sys.exit(3)"
        with self.assertRaisesRegex(RuntimeError, "exited with status 3:\nResolved 42 packages"):
            run_setup(arguments=[sys.executable, "-c", code], cache=Path(tempfile.gettempdir()), cancelled=threading.Event())

    def test_verification_failure_names_the_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "requirements.txt").write_text("dependency==1\n")
            (root / "check_runtime.py").write_text("pass\n")
            with patch.object(sys, "prefix", str(runtime)), patch("yams.runtime.torch_backend", return_value="cpu"), \
                    patch("yams.runtime.run_setup", side_effect=[None, RuntimeError("import torch failed")]):
                with self.assertRaisesRegex(RuntimeError, "Audio engine verification failed: import torch failed"):
                    ensure_runtime(installer=root / "uv", resources=root, emit=lambda *args, **kwargs: None, cancelled=threading.Event())


if __name__ == "__main__":
    unittest.main()
