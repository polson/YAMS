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

## Build a Linux preview package

```bash
npm run tauri build
```

This creates a Fedora-family x86_64 RPM at `src-tauri/target/release/bundle/rpm/`. Pushing a `vX.Y.Z` tag sets the bundled app version from the tag, builds an AppImage and RPM, then publishes a GitHub release. AppImage installs check for signed updates on launch and install them automatically. RPM installs continue to use manual package updates. The release workflow signs AppImage updates with the `TAURI_SIGNING_PRIVATE_KEY` GitHub Actions secret. The AppImage is intended for broader Linux use, but has not been checked across distributions. The RPM has been installed and launched in a disposable Fedora 44 container; desktop interaction and audio processing still need a real-machine check. Windows and macOS are not release targets yet.

## Current limits

The queue is kept in memory and is not restored when the app restarts. Add the folders again to continue; completed exports remain in the destination folder. CPU separation and playback in YARG still need verification.
