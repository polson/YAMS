import io
import math
import re
import struct
from pathlib import Path


CON_TAGS = {b"CON ", b"LIVE", b"PIRS"}
FIRSTBLOCK_OFFSET = 0xC000
BYTES_PER_BLOCK = 0x1000
BLOCKS_PER_SECTION = 170
NUM_BLOCKS_SQUARED = 170 * 170
BYTES_PER_HASH_ENTRY = 0x18
NEXT_BLOCK_HASH_OFFSET = 0x15


def is_con_file(path_or_data):
    if isinstance(path_or_data, (str, Path)):
        path = Path(path_or_data)
        if not path.is_file():
            return False
        with path.open("rb") as f:
            header = f.read(4)
            return header in CON_TAGS
    elif isinstance(path_or_data, (bytes, bytearray)):
        return len(path_or_data) >= 4 and bytes(path_or_data[:4]) in CON_TAGS
    return False


def calculate_block_location(block_num, shift):
    block_adjust = 0
    if block_num >= BLOCKS_PER_SECTION:
        block_adjust += ((block_num // BLOCKS_PER_SECTION) + 1) << shift
        if block_num >= NUM_BLOCKS_SQUARED:
            block_adjust += ((block_num // NUM_BLOCKS_SQUARED) + 1) << shift
    return FIRSTBLOCK_OFFSET + (block_adjust + block_num) * BYTES_PER_BLOCK


class ConListing:
    def __init__(self, name, flags, block_count, block_offset, path_index, length, shift):
        self.name = name
        self.flags = flags
        self.block_count = block_count
        self.block_offset = block_offset
        self.path_index = path_index
        self.length = length
        self.shift = shift

    @property
    def is_contiguous(self):
        return bool(self.flags & 0x40)

    @property
    def is_directory(self):
        return bool(self.flags & 0x80)


class ConPackage:
    def __init__(self, source):
        if isinstance(source, (str, Path)):
            self._stream = Path(source).open("rb")
            self._close_stream = True
        else:
            self._stream = source
            self._close_stream = False
        self.listings = []
        try:
            self._parse()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._close_stream and self._stream:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _parse(self):
        self._stream.seek(0)
        tag = self._stream.read(4)
        if tag not in CON_TAGS:
            raise ValueError(f"Invalid package header: {tag!r}")

        self._stream.seek(0x340)
        entry_id_bytes = self._stream.read(4)
        if len(entry_id_bytes) != 4:
            raise ValueError("Unexpected EOF reading entry ID")
        entry_id = struct.unpack(">I", entry_id_bytes)[0]
        shift = 0 if ((entry_id + 0xFFF & 0xF000) >> 0xC == 0xB) else 1

        self._stream.seek(0x37C)
        block_count_bytes = self._stream.read(2)
        if len(block_count_bytes) != 2:
            raise ValueError("Unexpected EOF reading filetable block count")
        table_block_count = struct.unpack("<H", block_count_bytes)[0]
        table_length = table_block_count * BYTES_PER_BLOCK

        self._stream.seek(0x37E)
        first_block_bytes = self._stream.read(3)
        if len(first_block_bytes) != 3:
            raise ValueError("Unexpected EOF reading filetable first block")
        first_block = (first_block_bytes[0] << 16) | (first_block_bytes[1] << 8) | first_block_bytes[2]

        table_offset = calculate_block_location(first_block, shift)
        self._stream.seek(table_offset)
        table_data = self._stream.read(table_length)

        offset = 0
        while offset + 0x40 <= len(table_data) and table_data[offset] != 0:
            entry = table_data[offset:offset + 0x40]
            name_len = entry[0x28] & 0x3F
            flags = entry[0x28] & 0xC0
            name_str = entry[:name_len].decode("utf-8", errors="replace")
            path_idx = struct.unpack(">h", entry[0x32:0x34])[0]
            if path_idx >= 0 and path_idx < len(self.listings):
                full_name = f"{self.listings[path_idx].name}/{name_str}"
            else:
                full_name = name_str

            block_count = entry[0x29] | (entry[0x2A] << 8) | (entry[0x2B] << 16)
            block_offset = entry[0x2F] | (entry[0x30] << 8) | (entry[0x31] << 16)
            file_len = struct.unpack(">I", entry[0x34:0x38])[0]

            self.listings.append(ConListing(
                name=full_name,
                flags=flags,
                block_count=block_count,
                block_offset=block_offset,
                path_index=path_idx,
                length=file_len,
                shift=shift,
            ))
            offset += 0x40

    def find(self, pattern):
        pattern_lower = pattern.lower()
        for listing in self.listings:
            if not listing.is_directory and pattern_lower in listing.name.lower():
                return listing
        return None

    def read_file(self, listing):
        if listing.is_directory:
            raise ValueError(f"Cannot read directory: {listing.name}")
        if listing.length == 0:
            return b""

        output = bytearray(listing.length)
        current_block = listing.block_offset

        if listing.is_contiguous:
            num_blocks = BLOCKS_PER_SECTION - (current_block % BLOCKS_PER_SECTION)
            read_size = BYTES_PER_BLOCK * num_blocks
            remaining = listing.length
            pos = 0
            while remaining > 0:
                self._stream.seek(calculate_block_location(current_block, listing.shift))
                chunk = self._stream.read(min(read_size, remaining))
                output[pos:pos + len(chunk)] = chunk
                pos += len(chunk)
                remaining -= len(chunk)
                current_block += num_blocks
                num_blocks = BLOCKS_PER_SECTION
                read_size = BLOCKS_PER_SECTION * BYTES_PER_BLOCK
        else:
            pos = 0
            for i in range(listing.block_count):
                block_loc = calculate_block_location(current_block, listing.shift)
                self._stream.seek(block_loc)
                read_count = min(BYTES_PER_BLOCK, listing.length - (i * BYTES_PER_BLOCK))
                chunk = self._stream.read(read_count)
                output[pos:pos + len(chunk)] = chunk
                pos += len(chunk)

                if i + 1 < listing.block_count:
                    block_offset_in_sec = current_block % BLOCKS_PER_SECTION
                    hash_loc = block_loc - ((block_offset_in_sec + 1) * BYTES_PER_BLOCK)
                    self._stream.seek(hash_loc)
                    hash_block = self._stream.read(BYTES_PER_BLOCK)
                    idx = block_offset_in_sec * BYTES_PER_HASH_ENTRY + NEXT_BLOCK_HASH_OFFSET
                    current_block = (hash_block[idx] << 16) | (hash_block[idx + 1] << 8) | hash_block[idx + 2]

        return bytes(output)

    def extract_unpacked_rbcon(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for listing in self.listings:
            if listing.is_directory:
                continue
            data = self.read_file(listing)
            target = destination / listing.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def demux_mogg(mogg_bytes):
    if len(mogg_bytes) < 8:
        raise ValueError("MOGG data is too short")
    version, start_offset = struct.unpack("<II", mogg_bytes[:8])
    if version != 10:
        raise ValueError(f"Unsupported or encrypted MOGG version: {version} (expected unencrypted version 10)")
    if start_offset >= len(mogg_bytes):
        raise ValueError("Invalid MOGG stream offset")
    return mogg_bytes[start_offset:]


def parse_dta(dta_text):
    text = re.sub(r";[^\r\n]*", "", dta_text)
    tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', text)

    def parse_tokens(token_iter):
        items = []
        for token in token_iter:
            if token == "(":
                items.append(parse_tokens(token_iter))
            elif token == ")":
                return items
            else:
                if token.startswith('"') and token.endswith('"'):
                    items.append(token[1:-1])
                elif token.startswith("'") and token.endswith("'"):
                    items.append(token[1:-1])
                elif token.startswith("'"):
                    items.append(token[1:])
                else:
                    items.append(token)
        return items

    tree = parse_tokens(iter(tokens))
    return tree


def extract_audio_info_from_dta(tree):
    info = {"pans": [], "vols": [], "tracks": {}, "name": "", "song_id": ""}
    if not tree or not isinstance(tree, list):
        return info

    def find_node(node, name):
        if not isinstance(node, list):
            return None
        target = name.strip("'").lower()
        for item in node:
            if isinstance(item, list) and len(item) > 0 and str(item[0]).strip("'").lower() == target:
                return item
        return None

    root = tree[0] if len(tree) > 0 and isinstance(tree[0], list) else tree
    if isinstance(root, list) and len(root) > 0 and isinstance(root[0], str):
        info["song_id"] = str(root[0]).strip("'")

    song_node = find_node(root, "song")
    if not song_node:
        return info

    name_node = find_node(song_node, "name")
    if name_node and len(name_node) > 1:
        info["name"] = str(name_node[1]).strip("'")

    pans_node = find_node(song_node, "pans")
    if pans_node and len(pans_node) > 1 and isinstance(pans_node[1], list):
        info["pans"] = [float(p) for p in pans_node[1]]

    vols_node = find_node(song_node, "vols")
    if vols_node and len(vols_node) > 1 and isinstance(vols_node[1], list):
        info["vols"] = [float(v) for v in vols_node[1]]

    tracks_node = find_node(song_node, "tracks")
    if tracks_node and len(tracks_node) > 1 and isinstance(tracks_node[1], list):
        for instrument_node in tracks_node[1]:
            if isinstance(instrument_node, list) and len(instrument_node) > 1:
                inst_name = str(instrument_node[0]).strip("'")
                channels = instrument_node[1]
                if isinstance(channels, list):
                    info["tracks"][inst_name] = [int(c) for c in channels]
                else:
                    info["tracks"][inst_name] = [int(channels)]

    return info


def _find_dta_entry(tree, target_name=None):
    if isinstance(tree, list) and tree and isinstance(tree[0], list):
        for entry in tree:
            if isinstance(entry, list) and entry:
                if target_name and str(entry[0]).strip("'").lower() == target_name.lower():
                    return entry
        return tree[0]
    return tree


def is_con_multitrack(dta_text, mogg_source=None):
    if ";Multitrack=0" in dta_text:
        return False
    if ";Multitrack=1" in dta_text:
        return True

    tree = parse_dta(dta_text)
    target_name = mogg_source.stem if isinstance(mogg_source, Path) else None
    info = extract_audio_info_from_dta(_find_dta_entry(tree, target_name))
    tracks = info.get("tracks", {})
    distinct_channel_sets = {tuple(sorted(chs)) for chs in tracks.values()}
    if len(distinct_channel_sets) < 2:
        return False

    if mogg_source is not None:
        try:
            import io
            import numpy as np
            import soundfile as sf
            if isinstance(mogg_source, (str, Path)):
                with open(mogg_source, "rb") as f:
                    mogg_bytes = f.read()
            else:
                mogg_bytes = mogg_source
            raw_ogg = demux_mogg(mogg_bytes)
            with sf.SoundFile(io.BytesIO(raw_ogg)) as f:
                if f.channels <= 2:
                    return False
                sample_frames = int(f.samplerate)
                positions = [int(f.frames * 0.25), int(f.frames * 0.5), int(f.frames * 0.75)]
                chunks = []
                for pos in positions:
                    if pos < f.frames:
                        f.seek(pos)
                        chunks.append(f.read(min(sample_frames, f.frames - pos)))
                if not chunks:
                    return False
                combined = np.concatenate(chunks, axis=0)
                peaks = np.max(np.abs(combined), axis=0)
                active_instruments = 0
                for chs in tracks.values():
                    if any(c < len(peaks) and peaks[c] > 0.01 for c in chs):
                        active_instruments += 1
                return active_instruments >= 2
        except Exception:
            return len(distinct_channel_sets) >= 2

    return len(distinct_channel_sets) >= 2


def patch_dta_audio(text, profile="mega-8stem"):
    song_match = re.search(r"\(\s*'?song'?\b", text)
    if not song_match:
        raise ValueError("Could not find (song ...) block in songs.dta")
    song_start = song_match.start()
    depth = 0
    song_end = len(text)
    for i in range(song_start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                song_end = i + 1
                break
    song_body = text[song_start:song_end]

    if profile in ("mlx-6stem", "torch-6stem"):
        replacements = [
            ("tracks_count", "(tracks_count (2 2 2 2 2 2))"),
            ("tracks", "(tracks\n      ((drum (0 1))\n       (bass (2 3))\n       (guitar (4 5))\n       (vocals (6 7))\n       (keys (8 9))\n      )\n    )"),
            ("pans", "(pans (-1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0))"),
            ("vols", "(vols (0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0))"),
            ("cores", "(cores (-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1))"),
        ]
    else:
        replacements = [
            ("tracks_count", "(tracks_count (6 2 2 2 2 2))"),
            ("tracks", "(tracks\n      ((drum (0 1 2 3 4 5))\n       (bass (6 7))\n       (guitar (8 9))\n       (vocals (10 11))\n       (keys (12 13))\n      )\n    )"),
            ("pans", "(pans (-1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0 -1.0 1.0))"),
            ("vols", "(vols (0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0))"),
            ("cores", "(cores (-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1))"),
        ]

    for tag, replacement in replacements:
        tag_match = re.search(rf"\(\s*'?{tag}'?\b", song_body)
        if tag_match:
            t_start = tag_match.start()
            t_depth = 0
            t_end = len(song_body)
            for j in range(t_start, len(song_body)):
                if song_body[j] == "(":
                    t_depth += 1
                elif song_body[j] == ")":
                    t_depth -= 1
                    if t_depth == 0:
                        t_end = j + 1
                        break
            song_body = song_body[:t_start] + replacement + song_body[t_end:]
        elif tag != "tracks_count":
            song_body = song_body[:-1].rstrip() + "\n    " + replacement + "\n  )"

    return text[:song_start] + song_body + text[song_end:]


def create_unencrypted_mogg(stem_files, output_mogg_path):
    import subprocess
    temp_ogg = output_mogg_path.with_suffix(".tmp.ogg")
    inputs = []
    for f in stem_files:
        inputs.extend(["-i", str(f)])
    filter_graph = "".join(f"[{i}:a]" for i in range(len(stem_files))) + f"amerge=inputs={len(stem_files)}[out]"
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        *inputs,
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-c:a", "libvorbis",
        "-q:a", "8",
        str(temp_ogg),
    ]
    subprocess.run(cmd, check=True)
    raw_ogg = temp_ogg.read_bytes()
    temp_ogg.unlink(missing_ok=True)
    header = struct.pack("<II", 10, 8)
    output_mogg_path.write_bytes(header + raw_ogg)


def unpack_con_song(con_path, destination_folder):
    con_path = Path(con_path)
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)

    with ConPackage(con_path) as pkg:
        pkg.extract_unpacked_rbcon(destination)

    # Locate songs.dta
    dta_path = destination / "songs.dta"
    if not dta_path.exists():
        # Might be in songs/songs.dta
        alt = destination / "songs/songs.dta"
        if alt.exists():
            dta_path = alt
        else:
            candidates = list(destination.glob("**/songs.dta"))
            if candidates:
                dta_path = candidates[0]
            else:
                raise ValueError("No songs.dta found in unpacked CON")

    dta_text = dta_path.read_text(encoding="utf-8", errors="replace")
    tree = parse_dta(dta_text)
    info = extract_audio_info_from_dta(tree)
    name = info.get("name") or info.get("song_id") or "song"
    sub_name = name.split("/")[-1]

    # Look for .mid and .mogg
    midi_candidates = list(destination.glob(f"**/{sub_name}.mid")) or list(destination.glob("**/*.mid"))
    mogg_candidates = list(destination.glob(f"**/{sub_name}.mogg")) or list(destination.glob("**/*.mogg"))

    midi_path = midi_candidates[0] if midi_candidates else None
    mogg_path = mogg_candidates[0] if mogg_candidates else None

    return {
        "song_id": info.get("song_id", sub_name),
        "sub_name": sub_name,
        "dta_path": dta_path,
        "midi_path": midi_path,
        "mogg_path": mogg_path,
        "destination": destination,
    }


def render_stereo_from_mogg(mogg_path, dta_path, output_wav_path):
    import io
    import numpy as np
    import soundfile as sf

    mogg_bytes = Path(mogg_path).read_bytes()
    raw_ogg = demux_mogg(mogg_bytes)

    dta_text = Path(dta_path).read_text(encoding="utf-8", errors="replace")
    tree = parse_dta(dta_text)
    info = extract_audio_info_from_dta(tree)
    pans = info.get("pans", [])
    vols = info.get("vols", [])

    data, sr = sf.read(io.BytesIO(raw_ogg), dtype="float32", always_2d=True)
    num_channels = data.shape[1]

    left = np.zeros(data.shape[0], dtype=np.float32)
    right = np.zeros(data.shape[0], dtype=np.float32)

    for c in range(num_channels):
        vol = vols[c] if c < len(vols) else 0.0
        pan = pans[c] if c < len(pans) else 0.0
        scale = 10.0 ** (vol / 20.0)
        pan_rad = (pan + 1.0) * math.pi / 4.0
        left_gain = math.cos(pan_rad) * scale
        right_gain = math.sin(pan_rad) * scale
        left += data[:, c] * left_gain
        right += data[:, c] * right_gain

    stereo = np.stack([left, right], axis=1)
    sf.write(output_wav_path, stereo, sr, format="WAV", subtype="FLOAT")
    return output_wav_path


OUTPUTS_MOGG_ORDER = (
    "drums_1.wav", "drums_2.wav", "drums_3.wav", "bass.wav",
    "guitar.wav", "vocals.wav", "keys.wav", "song.wav"
)
OUTPUTS_MOGG_ORDER_6STEM = (
    "drums.wav", "bass.wav", "guitar.wav",
    "vocals.wav", "keys.wav", "song.wav"
)
DEFAULT_INFERENCE = Path(__file__).resolve().parent / "inference"


def mogg_outputs_for_profile(profile):
    if profile in ("mlx-6stem", "torch-6stem"):
        return OUTPUTS_MOGG_ORDER_6STEM
    return OUTPUTS_MOGG_ORDER


def process_con_song(con_path, destination, mode, models, inference_source=None, device="auto", emit=None, cancelled=None, model_profile="mega-8stem"):
    if emit is None:
        emit = lambda *args, **kwargs: None
    if inference_source is None:
        inference_source = DEFAULT_INFERENCE
    import json
    from .folders import copy_file_exclusive, digest, publish_staged_tree, remove_stage, restore_file, sync_directory, write_record

    con_path = Path(con_path)
    destination = Path(destination)
    source_hash = digest(con_path)
    destination.mkdir(parents=True, exist_ok=True)

    workspace = destination / ".yams-work"
    workspace.mkdir()
    record = {
        "version": 1,
        "format": "con_file",
        "state": "processing",
        "source": str(con_path.resolve()),
        "source_sha256": source_hash,
        "source_backup": f"backup/{con_path.name}" if mode == "inplace" else None,
        "remove_source": mode == "inplace",
        "mode": mode,
        "destination": str(destination.resolve()),
        "files": {},
    }
    write_record(workspace, record)

    payload = workspace / "payload"
    payload.mkdir()
    info = unpack_con_song(con_path, payload)
    dta_path = info["dta_path"]
    mogg_path = info["mogg_path"]
    if not mogg_path or not dta_path:
        raise ValueError("Missing mogg or songs.dta in CON package")

    dta_text = dta_path.read_text(encoding="utf-8", errors="replace")
    if is_con_multitrack(dta_text, mogg_path):
        raise ValueError("Cannot perform full separation on a song that already contains multitrack stems")

    stage = payload / ".yams-work"
    stage.mkdir()
    outputs_dir = stage / "outputs"
    outputs_dir.mkdir()

    ref_wav = stage / "original"
    render_stereo_from_mogg(mogg_path, dta_path, ref_wav)

    if cancelled and cancelled.is_set():
        raise KeyboardInterrupt

    from .audio import separate

    result = separate(stage=stage, models=models, inference_source=inference_source,
                      device_name=device, emit=emit,
                      model_profile=model_profile, cancelled=cancelled)

    if cancelled and cancelled.is_set():
        raise KeyboardInterrupt

    if "pcm_validation" in result:
        outputs_dir = stage / "pcm"
        result["validation"] = result["pcm_validation"]
    stem_files = [outputs_dir / name for name in mogg_outputs_for_profile(model_profile)]
    create_unencrypted_mogg(stem_files=stem_files, output_mogg_path=mogg_path)

    patched_dta = patch_dta_audio(dta_path.read_text(encoding="utf-8", errors="replace"), profile=model_profile)
    dta_path.write_text(patched_dta, encoding="utf-8")

    manifest_record = {
        "version": 1,
        "state": "completed",
        "format": "unpacked_rbcon",
        "source": con_path.name,
        "validation": result.get("validation", {}),
    }

    if mode == "inplace":
        backup_dir = payload / "backup"
        if backup_dir.is_symlink() or (backup_dir.exists() and not backup_dir.is_dir()):
            raise ValueError("Expected an ordinary CON backup directory")
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / con_path.name
        manifest_path = backup_dir / "separation-manifest.json"
        if backup_path.exists() or backup_path.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
            raise ValueError("CON backup destination already contains a file")
        copy_file_exclusive(con_path, backup_path)
        if digest(backup_path) != record["source_sha256"]:
            raise ValueError("CON source changed while creating its backup")
        manifest_path.write_text(json.dumps(manifest_record, indent=2) + "\n", encoding="utf-8")

    remove_stage(stage)
    if digest(con_path) != record["source_sha256"]:
        raise ValueError("CON source changed before installation")
    publish_staged_tree(stage=workspace, payload=payload, destination=destination, record=record)
    if record["remove_source"]:
        con_path.unlink()
        sync_directory(con_path.parent)
    record["state"] = "committed"
    write_record(workspace, record)
    sync_directory(destination)
    remove_stage(workspace)
    return result


def abort_con_process(destination, con_path, mode):
    import json
    from .folders import digest, pending_stage, remove_stage, restore_file, rollback_staged_tree, sync_directory

    destination = Path(destination)
    if not destination.exists():
        return
    stage = pending_stage(destination)
    if stage.is_symlink():
        raise ValueError("Pending CON work directory must not be a symlink")
    journal = stage / "job.json"
    if not journal.is_file():
        journal = stage / "job.tmp"
    if not journal.is_file():
        if stage.is_dir() and not any(stage.iterdir()):
            stage.rmdir()
            sync_directory(destination)
        return
    record = json.loads(journal.read_text(encoding="utf-8"))
    if record.get("format") != "con_file":
        raise ValueError("Pending work belongs to another file format")
    if record["version"] != 1 or record["state"] not in {"processing", "publishing", "committed"}:
        raise ValueError("Unsupported CON recovery record")
    if record["mode"] != mode:
        raise ValueError("Pending CON work has a different mode")
    if Path(record["destination"]) != destination.resolve():
        raise ValueError("Invalid CON recovery destination")
    if record["state"] != "committed":
        source = Path(record["source"])
        if source != Path(con_path).resolve():
            raise ValueError("Invalid CON recovery source")
        if record["remove_source"] and not source.exists() and not source.is_symlink():
            backup_name = Path("backup") / source.name
            if Path(record["source_backup"]) != backup_name:
                raise ValueError("Invalid CON recovery backup")
            backup = stage / "payload" / backup_name
            if digest(backup) != record["source_sha256"]:
                raise ValueError("CON backup changed; recovery retained")
            restore_file(source=backup, target=source, expected_hash=record["source_sha256"])
        elif record["remove_source"] and source.is_symlink():
            raise ValueError("CON recovery source changed; recovery retained")
        elif record["remove_source"] and digest(source) != record["source_sha256"]:
            raise ValueError("CON recovery source changed; recovery retained")
        if record["state"] == "publishing":
            rollback_staged_tree(
                payload=stage / "payload", destination=destination, record=record
            )
    if stage.exists() and stage.is_dir():
        remove_stage(stage)


def is_excon_folder(folder):
    if not isinstance(folder, Path):
        folder = Path(folder)
    if not folder.is_dir():
        return False
    if (folder / "song.ini").is_file():
        return False
    has_dta = (folder / "songs.dta").is_file() or (folder / "songs/songs.dta").is_file()
    if not has_dta and folder.parent.exists():
        has_dta = (folder.parent / "songs.dta").is_file()
    if not has_dta:
        has_dta = any(folder.glob("**/songs.dta"))
    if not has_dta:
        return False
    return any(folder.glob("*.mogg")) or any(folder.glob("**/*.mogg"))


def find_excon_info(folder):
    folder = Path(folder)
    dta_path = folder / "songs.dta"
    if not dta_path.is_file():
        alt = folder / "songs/songs.dta"
        if alt.is_file():
            dta_path = alt
        elif (folder.parent / "songs.dta").is_file():
            dta_path = folder.parent / "songs.dta"
        else:
            candidates = list(folder.glob("**/songs.dta"))
            if candidates:
                dta_path = candidates[0]
            else:
                raise ValueError(f"No songs.dta found in exCON folder: {folder}")

    mogg_candidates = list(folder.glob("*.mogg")) or list(folder.glob("**/*.mogg"))
    if not mogg_candidates:
        raise ValueError(f"No .mogg found in exCON folder: {folder}")
    mogg_path = mogg_candidates[0]

    dta_text = dta_path.read_text(encoding="utf-8", errors="replace")
    tree = parse_dta(dta_text)
    info = extract_audio_info_from_dta(_find_dta_entry(tree, mogg_path.stem))
    name = info.get("name") or info.get("song_id") or mogg_path.stem
    sub_name = name.split("/")[-1]

    midi_candidates = list(folder.glob(f"**/{sub_name}.mid")) or list(folder.glob("**/*.mid"))

    return {
        "song_id": info.get("song_id", sub_name),
        "sub_name": sub_name,
        "dta_path": dta_path,
        "midi_path": midi_candidates[0] if midi_candidates else None,
        "mogg_path": mogg_path,
        "destination": folder,
    }


def process_excon_song(excon_path, destination, mode, models, inference_source=None, device="auto", emit=None, cancelled=None, model_profile="mega-8stem"):
    if emit is None:
        emit = lambda *args, **kwargs: None
    if inference_source is None:
        inference_source = DEFAULT_INFERENCE
    import json
    from .audio import separate
    from .folders import copy_file_exclusive, digest, remove_stage, write_record

    excon_path = Path(excon_path)
    destination = Path(destination)

    if mode == "export":
        source_root = excon_path.resolve()
        destination_root = destination.resolve()
        if destination_root.is_relative_to(source_root):
            raise ValueError("An exCON export destination cannot be inside its source folder")
        source_files = [
            path for path in excon_path.rglob("*")
            if path.is_file() and not path.is_symlink()
            and not any(part.startswith((".yams", ".yass")) for part in path.relative_to(excon_path).parts)
        ]
        targets = [(path, path.relative_to(excon_path), destination / path.relative_to(excon_path))
                   for path in source_files]
        for _, relative, target in targets:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Export would overwrite an existing file: {target}")
            if not target.parent.resolve().is_relative_to(destination_root):
                raise ValueError(f"Export path escapes its destination: {relative}")
            for parent in target.parents:
                if parent == destination:
                    break
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise ValueError(f"Expected an ordinary export directory: {parent}")
        destination.mkdir(parents=True, exist_ok=True)
        for path, _, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file_exclusive(path, target)
        working_folder = destination
    else:
        destination.mkdir(parents=True, exist_ok=True)
        working_folder = excon_path

    info = find_excon_info(working_folder)
    dta_path = info["dta_path"]
    mogg_path = info["mogg_path"]

    dta_text = dta_path.read_text(encoding="utf-8", errors="replace")
    if is_con_multitrack(dta_text, mogg_path):
        raise ValueError("Cannot perform full separation on a song that already contains multitrack stems")

    stage = working_folder / ".yams-work"
    stage.mkdir()
    write_record(stage, {"version": 1, "format": "excon", "state": "processing", "mode": mode, "originals": []})
    outputs_dir = stage / "outputs"
    outputs_dir.mkdir()

    backup_dir = working_folder / "backup"
    if mode == "inplace":
        if backup_dir.is_symlink() or (backup_dir.exists() and not backup_dir.is_dir()):
            raise ValueError("Expected an ordinary exCON backup directory")
        backup_dir.mkdir(exist_ok=True)
        backup_mogg = backup_dir / mogg_path.name
        backup_dta = backup_dir / dta_path.name
        if any(path.is_symlink() or (path.exists() and not path.is_file()) for path in (backup_mogg, backup_dta)):
            raise ValueError("Expected ordinary exCON backup files")
        if not backup_mogg.exists():
            copy_file_exclusive(mogg_path, backup_mogg)
        if not backup_dta.exists():
            copy_file_exclusive(dta_path, backup_dta)
        mogg_hash = digest(mogg_path)
        dta_hash = digest(dta_path)
        if digest(backup_mogg) != mogg_hash or digest(backup_dta) != dta_hash:
            raise ValueError("Existing exCON backup does not match its source")
        write_record(stage, {
            "version": 1,
            "format": "excon",
            "state": "processing",
            "mode": mode,
            "originals": [
                {"path": mogg_path.relative_to(working_folder).as_posix(), "backup": backup_mogg.name, "sha256": mogg_hash},
                {"path": dta_path.relative_to(working_folder).as_posix(), "backup": backup_dta.name, "sha256": dta_hash},
            ],
        })
        ref_source = backup_mogg
    else:
        ref_source = mogg_path

    ref_wav = stage / "original"
    render_stereo_from_mogg(ref_source, dta_path, ref_wav)

    if cancelled and cancelled.is_set():
        raise KeyboardInterrupt

    result = separate(stage=stage, models=models, inference_source=inference_source,
                      device_name=device, emit=emit,
                      model_profile=model_profile, cancelled=cancelled)

    if cancelled and cancelled.is_set():
        raise KeyboardInterrupt

    if "pcm_validation" in result:
        outputs_dir = stage / "pcm"
        result["validation"] = result["pcm_validation"]
    stem_files = [outputs_dir / name for name in mogg_outputs_for_profile(model_profile)]
    create_unencrypted_mogg(stem_files=stem_files, output_mogg_path=mogg_path)

    patched_dta = patch_dta_audio(dta_path.read_text(encoding="utf-8", errors="replace"), profile=model_profile)
    dta_path.write_text(patched_dta, encoding="utf-8")

    manifest_record = {
        "version": 1,
        "state": "completed",
        "format": "excon",
        "source": mogg_path.name,
        "validation": result.get("validation", {}),
    }

    if mode == "inplace":
        (backup_dir / "separation-manifest.json").write_text(
            json.dumps(manifest_record, indent=2) + "\n", encoding="utf-8"
        )

    remove_stage(stage)
    return result


def abort_excon_process(destination, mode):
    import json
    from .folders import copy_file_exclusive, digest, pending_stage, remove_stage, restore_file

    destination = Path(destination)
    if not destination.exists():
        return
    stage = pending_stage(destination)
    if stage.is_symlink():
        raise ValueError("Pending exCON work directory must not be a symlink")
    journal = stage / "job.json"
    if journal.is_symlink():
        raise ValueError("Pending exCON journal must not be a symlink")
    if not journal.is_file():
        journal = stage / "job.tmp"
    if journal.is_symlink():
        raise ValueError("Pending exCON journal must not be a symlink")
    record = json.loads(journal.read_text(encoding="utf-8")) if journal.is_file() else None
    if record is not None and (
        record.get("format") != "excon"
        or record.get("version") != 1
        or record.get("state") != "processing"
        or record.get("mode") != mode
        or not isinstance(record.get("originals"), list)
    ):
        raise ValueError("Unsupported exCON recovery record")
    if mode == "inplace":
        backup_dir = destination / "backup"
        if record is not None:
            if record["originals"]:
                if backup_dir.is_symlink() or not backup_dir.is_dir():
                    raise ValueError("Invalid exCON backup directory")
                for original in record["originals"]:
                    relative_path = Path(original["path"])
                    backup_name = Path(original["backup"])
                    expected_hash = original["sha256"]
                    if relative_path.is_absolute() or ".." in relative_path.parts or backup_name.name != original["backup"]:
                        raise ValueError("Invalid exCON recovery path")
                    target = destination / relative_path
                    backup = backup_dir / backup_name
                    if target.is_symlink() or backup.is_symlink() or not backup.is_file():
                        raise ValueError("Invalid exCON recovery file")
                    if not target.parent.resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Invalid exCON recovery path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if digest(backup) != expected_hash:
                        raise ValueError(f"exCON backup changed; recovery retained: {backup}")
                    restore_file(source=backup, target=target, expected_hash=expected_hash)
        elif backup_dir.exists():
            if backup_dir.is_symlink() or not backup_dir.is_dir():
                raise ValueError("Invalid exCON backup directory")
            for item in backup_dir.iterdir():
                if item.suffix in (".mogg", ".dta"):
                    target = destination / item.name
                    if item.is_symlink() or not item.is_file():
                        raise ValueError("Invalid exCON recovery backup")
                    expected_hash = digest(item)
                    if target.exists() or target.is_symlink():
                        if target.is_symlink() or not target.is_file() or digest(target) != expected_hash:
                            raise ValueError(f"exCON source changed; recovery retained: {target}")
                    else:
                        copy_file_exclusive(item, target)
    if stage.exists() and stage.is_dir():
        remove_stage(stage)
