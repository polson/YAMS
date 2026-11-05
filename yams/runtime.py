import hashlib
import logging
import os
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading

from .folders import locked


logger = logging.getLogger(__name__)

VC_RUNTIME_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


class MissingVisualCRuntime(RuntimeError):
    code = "vc_redist_missing"

    def __init__(self):
        super().__init__(f"Microsoft Visual C++ Redistributable is not installed. Install it from {VC_RUNTIME_URL} and try again.")


def visual_c_runtime_installed():
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        return True
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64") as key:
            return winreg.QueryValueEx(key, "Installed")[0] == 1
    except OSError as error:
        logger.info("Visual C++ runtime registry key is not readable: %s", error)
        return False


def _has_amd_gpu():
    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            if vendor.is_file() and vendor.read_text().strip().lower() == "0x1002":
                return True
        except (OSError, UnicodeDecodeError) as error:
            logger.warning("Could not read GPU vendor file %s: %s", vendor, error)
    try:
        result = subprocess.run(args=["rocm-smi", "--showid"], capture_output=True)
    except OSError as error:
        logger.info("AMD GPU check: rocm-smi is unavailable: %s", error)
        return False
    if result.returncode != 0:
        logger.info("AMD GPU check: rocm-smi exited with status %s", result.returncode)
    return result.returncode == 0


def torch_backend():
    if platform.machine().lower() not in ("x86_64", "amd64") or sys.platform == "darwin":
        return "cpu"
    try:
        result = subprocess.run(args=["nvidia-smi", "--id=0", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        logger.info("NVIDIA GPU check failed: %s", detail)
        if sys.platform.startswith("linux") and _has_amd_gpu():
            return "rocm6.2.4"
        return "cpu"
    try:
        driver, capability = result.stdout.strip().split(",")
        version = tuple(int(part) for part in driver.strip().split("."))
        capability_value = float(capability)
    except ValueError as error:
        logger.warning("Could not read nvidia-smi output %r (%s); using the CPU PyTorch build", result.stdout.strip(), error)
        return "cpu"
    if capability_value >= 10:
        minimum = (572, 61) if sys.platform == "win32" else (570, 124, 6)
        return "cu128" if version >= minimum else "cpu"
    minimum = (560, 94) if sys.platform == "win32" else (560, 35, 3)
    if version >= minimum:
        return "cu126"
    minimum = (522, 6) if sys.platform == "win32" else (520, 61, 5)
    if version >= minimum:
        return "cu118"
    return "cpu"


def missing_packages(requirements):
    """Lists requirements that apply to this platform and are not installed. Reads package metadata only, so nothing is imported."""
    # Imported here, not at module level: first-time setup imports this module before packaging is installed.
    from packaging.requirements import InvalidRequirement, Requirement

    missing = []
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            requirement = Requirement(line.strip())
        except InvalidRequirement:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            distribution_version(requirement.name)
        except PackageNotFoundError:
            missing.append(requirement.name)
    return missing


def run_setup(arguments, cache, cancelled, emit=None):
    logger.info("Running %s", " ".join(str(argument) for argument in arguments))
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")}
    env["UV_CACHE_DIR"] = str(cache / "uv")
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(args=arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    output = []

    def read_output():
        for raw in process.stdout:
            output.extend(line.strip() for line in re.split(r"[\r\n]+", raw.decode(errors="replace")) if line.strip())

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    reported = 0

    def report():
        nonlocal reported
        if emit is not None and len(output) > reported:
            reported = len(output)
            emit(output[-1][:200])

    try:
        while True:
            if cancelled.is_set():
                raise KeyboardInterrupt
            report()
            try:
                status = process.wait(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                pass
        reader.join(timeout=5)
        report()
        if status != 0:
            detail = "\n".join(output[-200:])[-4000:]
            failure = f"{Path(str(arguments[0])).name} exited with status {status}"
            raise RuntimeError(f"{failure}:\n{detail}" if detail else f"{failure}. It printed no output.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=5)
        if not reader.is_alive():
            process.stdout.close()


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
        current = (marker.is_file() and marker.read_text() == version) or (legacy_marker.is_file() and legacy_marker.read_text() == version)
        if current:
            missing = missing_packages(requirements)
            if not missing:
                return
            logger.warning("Audio engine packages are missing: %s; reinstalling them", ", ".join(missing))
        elif marker.exists() or legacy_marker.exists():
            logger.info("Audio engine packages need setup: requirements or the setup script changed")
        else:
            logger.info("Audio engine packages need setup: no ready marker in %s", runtime)
        if not visual_c_runtime_installed():
            raise MissingVisualCRuntime
        backend = torch_backend()
        logger.info("Installing PyTorch backend %s", backend)
        emit("runtime_setup", message="Preparing audio engine… First setup may take several minutes.")
        marker.unlink(missing_ok=True)
        legacy_marker.unlink(missing_ok=True)
        try:
            run_setup(arguments=[str(installer), "--no-config", "pip", "install", "--reinstall", "--python", sys.executable,
                                 "--torch-backend", backend, "--requirements", str(requirements)],
                      cache=runtime.parent, cancelled=cancelled, emit=lambda message: emit("runtime_setup", message=message))
        except RuntimeError as error:
            raise RuntimeError(f"Installing audio engine packages failed: {error}") from error
        emit("runtime_setup", message="Verifying audio engine…")
        try:
            run_setup(arguments=[sys.executable, str(check)], cache=runtime.parent, cancelled=cancelled)
        except RuntimeError as error:
            raise RuntimeError(f"Audio engine verification failed: {error}") from error
        if cancelled.is_set():
            raise KeyboardInterrupt
        marker.write_text(version)
        logger.info("Audio engine packages are ready")
