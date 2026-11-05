import functools
import gzip
import hashlib
import logging
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


logger = logging.getLogger(__name__)

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

OPUS_TEST_ARGS = [
    "-nostdin",
    "-v", "error",
    "-f", "lavfi",
    "-i", "anullsrc=r=44100:cl=stereo",
    "-t", "0.02",
    "-af", "aresample=48000",
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


def _probe_ffmpeg(binary_path, arguments):
    """Runs ffmpeg with test arguments. Returns (succeeded, reason) and keeps ffmpeg's error output for the reason."""
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            [str(binary_path)] + arguments,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags
        )
    except OSError as error:
        return False, str(error)
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    return False, stderr[-2000:] or f"exit status {result.returncode}"


def test_ffmpeg_binary(binary_path):
    succeeded, reason = _probe_ffmpeg(binary_path, OPUS_TEST_ARGS)
    if not succeeded:
        logger.warning("FFmpeg at %s cannot encode Opus: %s", binary_path, reason)
    return succeeded


def run_ffmpeg(arguments, action):
    """Runs ffmpeg and raises a RuntimeError that includes its error output when it fails."""
    result = subprocess.run(["ffmpeg"] + arguments, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not {action} (exit status {result.returncode}): {stderr[-2000:] or 'no error output'}")


@functools.lru_cache(maxsize=4)
def supports_soxr(binary_path="ffmpeg"):
    succeeded, _reason = _probe_ffmpeg(binary_path, OPUS_SOXR_TEST_ARGS)
    if not succeeded:
        logger.info("FFmpeg at %s has no SoXR resampler; using the default resampler", binary_path)
    return succeeded


def resample_filter(rate, binary_path="ffmpeg"):
    if supports_soxr(binary_path):
        return f"aresample={rate}:resampler=soxr:precision=28"
    return f"aresample={rate}"


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
        emit("ffmpeg_download", message="Downloading FFmpeg (~30 MiB)…", received=0, total=asset["bytes"])

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
                        emit("ffmpeg_download", message="Downloading FFmpeg…", received=received, total=asset["bytes"])
                        reported = now

            if received != asset["bytes"] or checksum.hexdigest() != asset["sha256"]:
                raise ValueError(
                    f"FFmpeg download integrity check failed: received {received} of {asset['bytes']} bytes, "
                    f"SHA-256 {checksum.hexdigest()} (expected {asset['sha256']})"
                )

            output.flush()
            os.fsync(output.fileno())
        except OSError as error:
            temp_file.unlink(missing_ok=True)
            raise OSError(f"Could not download FFmpeg from {asset['url']}: {error}") from error
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

    succeeded, reason = _probe_ffmpeg(target_binary, OPUS_TEST_ARGS)
    if not succeeded:
        raise RuntimeError(f"Installed FFmpeg at {target_binary} failed verification: {reason}")
    logger.info("Installed FFmpeg at %s", target_binary)

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
