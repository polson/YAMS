import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from .folders import locked


def torch_backend():
    if platform.machine().lower() not in ("x86_64", "amd64") or sys.platform == "darwin":
        return "cpu"
    try:
        result = subprocess.run(args=["nvidia-smi", "--id=0", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except (OSError, subprocess.CalledProcessError):
        return "cpu"
    driver, capability = result.stdout.strip().split(",")
    version = tuple(int(part) for part in driver.strip().split("."))
    if float(capability) >= 10:
        minimum = (572, 61) if sys.platform == "win32" else (570, 124, 6)
        return "cu128" if version >= minimum else "cpu"
    minimum = (560, 94) if sys.platform == "win32" else (560, 35, 3)
    if version >= minimum:
        return "cu126"
    minimum = (522, 6) if sys.platform == "win32" else (520, 61, 5)
    if version >= minimum:
        return "cu118"
    return "cpu"


def run_setup(arguments, cache, cancelled):
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")}
    env["UV_CACHE_DIR"] = str(cache / "uv")
    with tempfile.TemporaryFile(mode="w+b") as diagnostics:
        process = subprocess.Popen(args=arguments, stdin=subprocess.DEVNULL, stdout=diagnostics, stderr=diagnostics,
                                   env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        try:
            while True:
                if cancelled.is_set():
                    raise KeyboardInterrupt
                try:
                    status = process.wait(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if status != 0:
                diagnostics.seek(max(0, diagnostics.tell() - 4000))
                raise RuntimeError(diagnostics.read().decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def ensure_runtime(installer, resources, emit, cancelled):
    if sys.prefix == sys.base_prefix:
        raise ValueError("Runtime installation requires YAMS's private virtual environment")
    runtime = Path(sys.prefix)
    requirements = resources / "requirements.txt"
    check = resources / "check_runtime.py"
    version = hashlib.sha256(requirements.read_bytes() + check.read_bytes()).hexdigest()
    with locked(runtime):
        from .ffmpeg import ensure_ffmpeg
        ensure_ffmpeg(cache_dir=runtime.parent, resources_dir=resources,
                      emit=lambda event, **kwargs: emit("runtime_setup", message=kwargs.get("message", "Preparing FFmpeg…")),
                      cancelled=cancelled)
        marker = runtime / ".yams-ready"
        legacy_marker = runtime / ".yass-ready"
        if (marker.is_file() and marker.read_text() == version) or (legacy_marker.is_file() and legacy_marker.read_text() == version):
            return
        emit("runtime_setup", message="Preparing audio engine… First setup may take several minutes.")
        marker.unlink(missing_ok=True)
        legacy_marker.unlink(missing_ok=True)
        run_setup(arguments=[str(installer), "--no-config", "pip", "install", "--reinstall", "--python", sys.executable,
                             "--torch-backend", torch_backend(), "--requirements", str(requirements)],
                  cache=runtime.parent, cancelled=cancelled)
        emit("runtime_setup", message="Verifying audio engine…")
        run_setup(arguments=[sys.executable, str(check)], cache=runtime.parent, cancelled=cancelled)
        if cancelled.is_set():
            raise KeyboardInterrupt
        marker.write_text(version)
