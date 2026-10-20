import os
from pathlib import Path
import platform
import subprocess
import sys


resources = Path(__file__).resolve().parent
for bin_dir in (resources / "bin", Path(sys.prefix).parent / "bin"):
    if bin_dir.is_dir() and str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
sys.path.insert(0, str(resources))
sys.path.insert(0, str(resources / "inference"))
sys.path.insert(0, str(resources / "yams/inference"))
sys.path.insert(0, str(resources / "yass/inference"))
if sys.version_info < (3, 11):
    raise RuntimeError("YAMS requires Python 3.11 or later")
import yams.__main__
if "--base" not in sys.argv:
    import yams.audio
    from utils import model_utils, settings
    from models.bs_roformer import BSRoformer
    if sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        import mlx.core
        import mlx_audio_separator
        from mlx_audio_separator.separator.models.roformer.loader import load_roformer_model
    subprocess.run(args=["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "0.02", "-af", "aresample=48000:resampler=soxr:precision=28", "-c:a", "libopus", "-f", "opus", "-"], check=True, stdout=subprocess.DEVNULL)
print(sys.executable)
