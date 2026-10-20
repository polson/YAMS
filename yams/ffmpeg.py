import gzip
import hashlib
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request


OPUS_SOXR_TEST_ARGS = [
    "-nostdin",
    "-v", "error",
    "-f", "lavfi",
    "-i", "anullsrc=r=44100:cl=stereo",
    "-t", "0.02",
    "-af", "aresample=48000:resampler=soxr:precision=28",
    "-c:a", "libopus",
    "-f", "opus",
    "-"
]

FFMPEG_ASSETS = {
    ("linux", "x86_64"): {
        "url": "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-linux-x64.gz",
        "bytes": 29354986,
        "sha256": "bfe8a8fc511530457b528c48d77b5737527b504a3797a9bc4866aeca69c2dffa",
        "binary": "ffmpeg",
    },
    ("darwin", "arm64"): {
        "url": "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64.gz",
        "bytes": 19246198,
        "sha256": "8923876afa8db5585022d7860ec7e589af192f441c56793971276d450ed3bbfa",
        "binary": "ffmpeg",
    },
    ("darwin", "x86_64"): {
        "url": "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-x64.gz",
        "bytes": 25296431,
        "sha256": "929b375c1182d956c51f7ac25e0b2b0411fb01f6f407aa15c9758efeb4242106",
        "binary": "ffmpeg",
    },
    ("win32", "x86_64"): {
        "url": "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-win32-x64.gz",
        "bytes": 29581307,
        "sha256": "8883a3dffbd0a16cf4ef95206ea05283f78908dbfb118f73c83f4951dcc06d77",
        "binary": "ffmpeg.exe",
    },
}


def _current_asset_key():
    os_name = "win32" if sys.platform == "win32" else ("darwin" if sys.platform == "darwin" else "linux")
    raw_arch = platform.machine().lower()
    arch = "arm64" if raw_arch in ("arm64", "aarch64") else ("x86_64" if raw_arch in ("x86_64", "amd64") else raw_arch)
    return (os_name, arch)


def test_ffmpeg_binary(binary_path):
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        result = subprocess.run(
            [str(binary_path)] + OPUS_SOXR_TEST_ARGS,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags
        )
        return result.returncode == 0
    except (OSError, subprocess.CalledProcessError):
        return False


def prepend_path(directory):
    directory_str = str(directory)
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    if directory_str not in parts:
        os.environ["PATH"] = os.pathsep.join([directory_str] + parts) if parts else directory_str


def find_ffmpeg(cache_bin=None, resources_bin=None):
    candidates = []
    binary_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    if cache_bin is not None:
        candidates.append(Path(cache_bin) / binary_name)
    if resources_bin is not None:
        candidates.append(Path(resources_bin) / binary_name)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        candidates.append(Path(system_ffmpeg))

    for candidate in candidates:
        if candidate.is_file() and test_ffmpeg_binary(candidate):
            prepend_path(candidate.parent)
            return candidate
    return None


def ensure_ffmpeg(cache_dir, resources_dir=None, emit=None, cancelled=None):
    cache_path = Path(cache_dir)
    cache_bin = cache_path / "bin"
    resources_bin = (Path(resources_dir) / "bin") if resources_dir is not None else None

    existing = find_ffmpeg(cache_bin=cache_bin, resources_bin=resources_bin)
    if existing is not None:
        return existing

    key = _current_asset_key()
    asset = FFMPEG_ASSETS.get(key)
    if asset is None:
        raise RuntimeError(f"No prebuilt static FFmpeg available for platform {key}")

    cache_bin.mkdir(parents=True, exist_ok=True)
    target_binary = cache_bin / asset["binary"]
    temp_file = cache_bin / f".{asset['binary']}.part"

    if emit:
        emit("ffmpeg_download", message="Downloading FFmpeg with Opus & SoXR (~30 MiB)…", received=0, total=asset["bytes"])

    request = urllib.request.Request(asset["url"], headers={"User-Agent": "YAMS/0.1"})
    received = 0
    checksum = hashlib.sha256()
    reported = 0.0

    with open(temp_file, "wb") as output:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                while block := response.read(1024 * 1024):
                    if cancelled is not None and cancelled.is_set():
                        raise KeyboardInterrupt
                    received += len(block)
                    if received > asset["bytes"]:
                        raise ValueError(f"Unexpected FFmpeg download size: received {received} > {asset['bytes']}")
                    output.write(block)
                    checksum.update(block)
                    now = time.monotonic()
                    if emit and (now - reported >= 0.25):
                        emit("ffmpeg_download", message="Downloading FFmpeg with Opus & SoXR…", received=received, total=asset["bytes"])
                        reported = now

            if received != asset["bytes"] or checksum.hexdigest() != asset["sha256"]:
                raise ValueError("FFmpeg download integrity check failed")

            output.flush()
            os.fsync(output.fileno())
        except Exception:
            temp_file.unlink(missing_ok=True)
            raise

    try:
        compressed_bytes = temp_file.read_bytes()
        decompressed_bytes = gzip.decompress(compressed_bytes)
        target_binary.write_bytes(decompressed_bytes)
        if sys.platform != "win32":
            target_binary.chmod(target_binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    finally:
        temp_file.unlink(missing_ok=True)

    prepend_path(cache_bin)

    if not test_ffmpeg_binary(target_binary):
        raise RuntimeError(f"Installed FFmpeg at {target_binary} failed verification")

    if emit:
        emit("ffmpeg_ready", message="FFmpeg ready")

    return target_binary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find or ensure FFmpeg with Opus and SoXR support")
    parser.add_argument("--cache", type=Path, required=True, help="Cache directory")
    parser.add_argument("--resources", type=Path, default=None, help="Resources directory")
    args = parser.parse_args()

    binary = ensure_ffmpeg(cache_dir=args.cache, resources_dir=args.resources,
                           emit=lambda event, **kwargs: print(f"[{event}] {kwargs.get('message', '')}", flush=True))
    print(f"FFmpeg ready at: {binary}")
