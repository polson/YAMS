from pathlib import Path
import shutil
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "desktop/.cache/engine"
UV_VERSION = "0.11.7"


def main():
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(f"Install uv {UV_VERSION} on the build machine before preparing YAMS engine resources")
    version = subprocess.run(args=[uv, "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
    if version != UV_VERSION:
        raise RuntimeError(f"YAMS engine resources require uv {UV_VERSION}; found {version}")
    ENGINE.mkdir(parents=True, exist_ok=True)
    for path in ENGINE.rglob("__pycache__"):
        shutil.rmtree(path)
    licenses = ENGINE / "licenses"
    licenses.mkdir(exist_ok=True)
    for name in ("LICENSE-MIT", "LICENSE-APACHE"):
        target = licenses / f"uv-{name}"
        if not target.is_file():
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/astral-sh/uv/{UV_VERSION}/{name}", timeout=60) as response:
                target.write_bytes(response.read())
    inference_target = ENGINE / "inference"
    if inference_target.exists():
        shutil.rmtree(inference_target)
    shutil.copytree(src=ROOT / "yams/inference", dst=inference_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    yams_target = ENGINE / "yams"
    if yams_target.exists():
        shutil.rmtree(yams_target)
    shutil.copytree(src=ROOT / "yams", dst=yams_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(src=ROOT / "desktop/engine/check_runtime.py", dst=ENGINE / "check_runtime.py")
    shutil.copy2(src=ROOT / "desktop/engine/requirements.txt", dst=ENGINE / "requirements.txt")
    shutil.copy2(src=ROOT / "desktop/engine/THIRD_PARTY_NOTICES.md", dst=ENGINE / "THIRD_PARTY_NOTICES.md")
    (ENGINE / "bin").mkdir(exist_ok=True)
    shutil.copy2(src=uv, dst=ENGINE / "bin" / Path(uv).name)
    print("Prepared YAMS worker, pinned inference source, and runtime installer")


if __name__ == "__main__":
    main()
