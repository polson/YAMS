<p align="center">
  <img src="Images/Logo_Green.png" alt="YAMS" width="300" />
</p>

# YAMS — Yet Another Music Separator

YAMS separates songs into instrument tracks for [YARG](https://yarg.in/). Add song folders, audio files, or CON packages to the desktop app’s queue.

Choose six or eight tracks. Both include vocals, drums, guitar, bass, keys, and backing; the eight-track option splits drums into kick, snare, and the rest of the kit. Processing runs on your computer, and the app downloads its models the first time you use it. Export to a folder you choose; your original files stay untouched.

## Run from source

There isn’t an installer in this repository yet. To run the desktop app from source:

```bash
cd desktop
npm ci --cache .cache/npm
npm run tauri dev
```

See the [desktop guide](desktop/README.md) for build requirements and setup details.
