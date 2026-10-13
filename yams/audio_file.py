import json
import os
from pathlib import Path
import shutil

from .confile import is_con_file
from .folders import (
    digest,
    pending_stage,
    publish_staged_tree,
    remove_stage,
    restore_file,
    rollback_staged_tree,
    sync_directory,
    write_record,
    STEMS,
)

AUDIO_FORMATS = {".opus", ".ogg", ".mp3", ".wav", ".flac", ".aiff", ".m4a", ".aac"}


def is_audio_file(path):
    if not isinstance(path, Path):
        path = Path(path)
    return path.is_file() and not path.is_symlink() and path.suffix.casefold() in AUDIO_FORMATS and not is_con_file(path)


def is_multitrack_folder(folder):
    if not isinstance(folder, Path):
        folder = Path(folder)
    if not folder.is_dir():
        return False
    try:
        audio_files = [path for path in folder.iterdir() if is_audio_file(path)]
    except OSError:
        return False
    if len(audio_files) < 2:
        return False
    stems = {p.stem.casefold() for p in audio_files}
    stem_matches = stems.intersection(STEMS)
    return any(stem != "song" for stem in stem_matches)


def find_single_audio_in_folder(folder):
    if not isinstance(folder, Path):
        folder = Path(folder)
    if not folder.is_dir():
        return None
    try:
        entries = list(folder.iterdir())
    except OSError:
        return None
    if any(e.name.casefold() == "song.ini" for e in entries):
        return None
    audio_files = [path for path in entries if is_audio_file(path)]
    if len(audio_files) == 1:
        return audio_files[0]
    return None


DEFAULT_INFERENCE = Path(__file__).resolve().parent / "inference"


def process_audio_file(audio_path, destination, mode, models, inference_source=None, device="auto", emit=None, cancelled=None, model_profile="mega-8stem"):
    if emit is None:
        emit = lambda *args, **kwargs: None
    if inference_source is None:
        inference_source = DEFAULT_INFERENCE
    audio_path = Path(audio_path)
    destination = Path(destination)
    source_hash = digest(audio_path)
    source_in_destination = audio_path.parent.resolve() == destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    stage = destination / ".yams-work"
    stage.mkdir()
    record = {
        "version": 1,
        "format": "audio_file",
        "state": "processing",
        "source": str(audio_path.resolve()),
        "source_sha256": source_hash,
        "source_backup": f"backup/{audio_path.name}" if mode == "inplace" else None,
        "remove_source": mode == "inplace" and source_in_destination,
        "source_replaced": False,
        "mode": mode,
        "destination": str(destination.resolve()),
        "files": {},
    }
    write_record(stage, record)

    outputs_dir = stage / "outputs"
    outputs_dir.mkdir()
    payload = stage / "payload"
    payload.mkdir()
    ref_audio = stage / "original"
    shutil.copy2(src=audio_path, dst=ref_audio)
    with ref_audio.open("rb") as f:
        os.fsync(f.fileno())
    if digest(ref_audio) != source_hash or digest(audio_path) != source_hash:
        raise ValueError("Source changed while creating the processing snapshot")

    if cancelled and cancelled.is_set():
        raise KeyboardInterrupt

    from .audio import separate

    result = separate(stage=stage, models=models, inference_source=inference_source,
                      device_name=device, emit=emit,
                      model_profile=model_profile, cancelled=cancelled)

    if cancelled and cancelled.is_set():
        raise KeyboardInterrupt

    for name in result["validation"]["files"]:
        shutil.copy2(src=outputs_dir / name, dst=payload / name)

    manifest_record = {
        "version": 1,
        "state": "completed",
        "format": "audio_file",
        "source": audio_path.name,
        "validation": result.get("validation", {}),
    }

    if mode == "inplace":
        backup_dir = payload / "backup"
        backup_dir.mkdir()
        shutil.copy2(src=ref_audio, dst=backup_dir / audio_path.name)
        (backup_dir / "separation-manifest.json").write_text(
            json.dumps(manifest_record, indent=2) + "\n", encoding="utf-8"
        )

    source_output = record["remove_source"] and audio_path.name in result["validation"]["files"]
    if source_output:
        if digest(audio_path) != source_hash:
            raise ValueError("Source changed before installation")
        record["source_replaced"] = True
        write_record(stage, record)
        audio_path.unlink()
        sync_directory(destination)

    ini_path = destination / "song.ini"
    if not ini_path.exists():
        song_title = audio_path.stem.replace("_", " ").strip() or "Untitled"
        ini_content = f"[Song]\nname = {song_title}\nartist = \nalbum = \nyear = \n"
        (payload / "song.ini").write_text(ini_content, encoding="utf-8")

    publish_staged_tree(stage=stage, payload=payload, destination=destination, record=record)
    if record["remove_source"] and not source_output:
        if digest(audio_path) != source_hash:
            raise ValueError("Source changed before installation")
        audio_path.unlink()
        sync_directory(destination)
    record["state"] = "committed"
    write_record(stage, record)
    sync_directory(destination)
    remove_stage(stage)
    return result


def abort_audio_process(destination, mode, audio_path):
    destination = Path(destination)
    stage = pending_stage(destination)
    if stage.exists():
        if stage.is_symlink():
            raise ValueError("Pending audio work directory must not be a symlink")
        journal = stage / "job.json"
        if not journal.is_file():
            journal = stage / "job.tmp"
        if not journal.is_file():
            if stage.is_dir() and not any(stage.iterdir()):
                stage.rmdir()
                sync_directory(destination)
            return
        record = json.loads(journal.read_text(encoding="utf-8"))
        if record.get("format") != "audio_file":
            raise ValueError("Pending work belongs to another audio format")
        if record["version"] != 1 or record["state"] not in {"processing", "publishing", "committed"}:
            raise ValueError("Unsupported audio recovery record")
        if record["mode"] != mode:
            raise ValueError("Pending audio work has a different mode")
        if Path(record["source"]) != Path(audio_path).resolve():
            raise ValueError("Invalid audio recovery source")
        if Path(record["destination"]) != destination.resolve():
            raise ValueError("Invalid audio recovery destination")
        if record["state"] != "committed":
            source = Path(record["source"])
            source_output = False
            if record["remove_source"] and record["source_replaced"] and source.exists() and not source.is_symlink():
                expected_output = record.get("files", {}).get(source.name)
                installed = record.get("installed", {}).get(source.name)
                if record.get("installing") == source.name:
                    installed = record.get("installing_identity") or installed
                status = source.stat()
                source_identity = [status.st_dev, status.st_ino]
                staged_output = stage / "payload" / source.name
                source_output = (
                    expected_output is not None
                    and digest(source) == expected_output
                    and (
                        installed == source_identity
                        or (
                            staged_output.is_file()
                            and os.path.samefile(staged_output, source)
                        )
                    )
                )
            source_missing = not source.exists() and not source.is_symlink()
            try:
                if record["state"] == "publishing":
                    rollback_staged_tree(
                        payload=stage / "payload", destination=destination, record=record
                    )
            finally:
                if record["remove_source"] and source.is_symlink():
                    raise ValueError("Audio recovery source changed; recovery retained")
                if (
                    record["remove_source"]
                    and source.exists()
                    and not source_output
                    and digest(source) != record["source_sha256"]
                ):
                    raise ValueError("Audio recovery source changed; recovery retained")
                if (
                    record["remove_source"]
                    and (source_missing or source_output)
                    and not source.exists()
                    and not source.is_symlink()
                ):
                    if source.parent.resolve() != destination.resolve():
                        raise ValueError("Invalid audio recovery source")
                    backup_name = Path("backup") / source.name
                    if Path(record["source_backup"]) != backup_name:
                        raise ValueError("Invalid audio recovery backup")
                    backup = stage / "payload" / backup_name
                    if digest(backup) != record["source_sha256"]:
                        raise ValueError("Audio backup changed; recovery retained")
                    restore_file(source=backup, target=source, expected_hash=record["source_sha256"])
        remove_stage(stage)
    if mode == "inplace" and destination.exists():
        try:
            if not any(destination.iterdir()):
                destination.rmdir()
        except OSError:
            pass
