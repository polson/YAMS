import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from .folders import digest, sync_directory


ASSETS = json.loads(Path(__file__).with_name("model_assets.json").read_text())


def _model_selection(profile):
    if profile in ("mega-8stem", "8stem"):
        is_mlx = sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")
        profile = "mlx-8stem" if is_mlx else "torch-8stem"
        stem_files = ASSETS["mlx_files"] if is_mlx else ASSETS["torch_6stem_files"]
        files = {**stem_files, **ASSETS["drumsep_files"]}
    elif profile == "mlx-6stem":
        files = ASSETS["mlx_files"]
    elif profile == "torch-6stem":
        files = ASSETS["torch_6stem_files"]
    elif profile == "mlx-8stem":
        files = {**ASSETS["mlx_files"], **ASSETS["drumsep_files"]}
    elif profile == "torch-8stem":
        files = {**ASSETS["torch_6stem_files"], **ASSETS["drumsep_files"]}
    else:
        raise ValueError(f"Unknown model profile: {profile}")
    return profile, files


def default_models(profile="mega-8stem"):
    if os.name == "nt":
        cache = Path(os.environ["LOCALAPPDATA"])
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    suffix, _ = _model_selection(profile)
    yams_path = cache / "yams" / "models" / suffix
    yass_path = cache / "yass" / "models" / suffix
    if not yams_path.exists() and yass_path.exists():
        return yass_path
    return yams_path


def download(folder, name, asset, emit):
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "YAMS/0.1"})
    with tempfile.NamedTemporaryFile(dir=folder, prefix=f".{name}-", suffix=".part", delete=False) as output:
        temporary = Path(output.name)
        try:
            received = 0
            checksum = hashlib.sha256()
            reported = 0.0
            emit("model_download", asset=name, received=0, total=asset["bytes"])
            with urllib.request.urlopen(request, timeout=15) as response:
                while block := response.read(1024**2):
                    received += len(block)
                    if received > asset["bytes"]:
                        raise ValueError(f"Unexpected model download size: {name}")
                    output.write(block)
                    checksum.update(block)
                    now = time.monotonic()
                    if now - reported >= 0.2:
                        emit("model_download", asset=name, received=received, total=asset["bytes"])
                        reported = now
            if received != asset["bytes"] or checksum.hexdigest() != asset["sha256"]:
                raise ValueError(f"Model download verification failed: {name}")
            output.flush()
            os.fsync(output.fileno())
            output.close()
            emit("model_download", asset=name, received=received, total=asset["bytes"])
            os.replace(temporary, folder / name)
            sync_directory(folder)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def ensure_models(folder, emit, profile="mega-8stem"):
    folder.mkdir(parents=True, exist_ok=True)
    emit("checking_models")
    _, files = _model_selection(profile)
    missing = []
    for name, asset in files.items():
        path = folder / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"]:
            continue
        missing.append((name, asset))
    required = sum(asset["bytes"] for _, asset in missing)
    if missing and shutil.disk_usage(folder).free < required + 64 * 1024**2:
        raise ValueError(f"Insufficient model cache space: need approximately {required / 1024**2:.0f} MiB")
    for name, asset in missing:
        download(folder=folder, name=name, asset=asset, emit=emit)
    emit("models_ready")
