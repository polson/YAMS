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

No installers or system configuration required. FFmpeg (with Opus and SoXR support) is detected automatically or downloaded into the app's local cache on first launch. GPU acceleration supports NVIDIA (CUDA) and Apple Silicon (Metal); AMD GPU support is in progress (runs in CPU mode).

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

