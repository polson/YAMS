# YAMS desktop

YAMS separates songs into instrument tracks for [YARG](https://yarg.in/). Add song folders, audio files, or Rock Band packages to the queue, choose an export folder, then start processing. Your source files stay untouched.

Choose six or eight tracks. Six separates vocals, drums, guitar, bass, piano, and backing. Eight also splits drums into kick, snare, and the rest of the kit.

## First use

The app prepares its audio engine automatically. The first separation downloads model files: about 1.1 GiB for the default eight-track profile on Linux, or 667 MiB for six tracks. An internet connection is needed. FFmpeg with Opus and SoXR support must be installed on the system. See [model notices](MODEL_NOTICES.md) for the weights and their licensing status.

## Run from source

Building requires Node.js, Rust, Python 3.11+, uv 0.11.7, and the native [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/).

```bash
npm ci --cache .cache/npm
npm run tauri dev
```

Run these commands from the `desktop` directory.

## Releases and portable packages

Pushing a `vX.Y.Z` tag (or triggering the GitHub Actions release workflow) builds and publishes ready-to-run portable packages for each platform directly to the GitHub release without requiring installers:

- **Linux (x86_64)**: An AppImage and portable `.tar.gz` bundle. Download, make executable (`chmod +x`), and run directly.
- **macOS (Apple Silicon)**: A DMG disk image and portable `.tar.gz` containing `YAMS.app`.
- **Windows (x64)**: A portable `.zip` containing `YAMS.exe` and bundled engine resources. Extract and run `YAMS.exe` directly.

The release workflow signs AppImage updates when the `TAURI_SIGNING_PRIVATE_KEY` repository secret is present. Packages publish straight to GitHub Releases without using Actions artifact storage.

## Current limits

The queue is kept in memory and is not restored when the app restarts. Add the folders again to continue; completed exports remain in the destination folder. CPU separation and playback in YARG still need verification.
