<p align="center">
  <img src="Images/Logo_Green.png" alt="YAMS" width="300" />
</p>

# YAMS: Yet Another Music Separator

YAMS separates songs into instrument tracks for [YARG](https://yarg.in/). Add song folders, audio files, or CON packages to the desktop app’s queue.

Choose six or eight tracks. Both include vocals, drums, guitar, bass, keys, and backing; the eight-track option splits drums into kick, snare, and the rest of the kit. Processing runs on your computer, and the app downloads its models the first time you use it. Export to a folder you choose; your original files stay untouched.

## Releases

Prebuilt, portable zero-install packages are available on the [Releases](https://github.com/polson/YAMS/releases) page:
- **Linux (x86_64)**: Standalone AppImage and portable `.tar.gz` bundle.
- **macOS (Apple Silicon)**: DMG disk image and portable `.tar.gz` containing `YAMS.app`.
- **Windows (x64)**: Portable `.zip` archive containing `YAMS.exe` and bundled engine resources.

No installers or system configuration required. FFmpeg (with Opus and SoXR support) is detected automatically or downloaded into the app's local cache on first launch.

## System requirements

- **Operating System**: Windows 10/11 (64-bit), Linux (x86_64), or macOS (Apple Silicon).
- **GPU Acceleration**:
  - **NVIDIA**: Supported via CUDA (requires recent NVIDIA graphics drivers).
  - **Apple Silicon**: Supported natively via Metal.
  - **AMD**: Supported via ROCm on Linux and DirectML (ONNX) on Windows.
  - **CPU**: Falls back to CPU mode if a compatible GPU is not detected.
- **Memory**: 8 GB RAM minimum (16 GB recommended).
- **Storage & Network**: ~2 GB free disk space for engine dependencies and models; internet connection required on first launch.

## Run from source

To run the desktop app from source:

```bash
cd desktop
npm ci --cache .cache/npm
npm run tauri dev
```

See the [desktop guide](desktop/README.md) for build requirements and setup details.

## License

YAMS is licensed under the [MIT License](LICENSE).

