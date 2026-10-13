import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


FORMATS = {".ogg", ".opus", ".mp3", ".wav", ".aiff"}
STEMS = {"song", "guitar", "bass", "rhythm", "keys", "vocals", "vocals_1", "vocals_2",
         "drums", "drums_1", "drums_2", "drums_3", "drums_4", "crowd", "song_clean",
         "vocals_clean", "vocals_1_clean", "vocals_2_clean", "crowd_clean", "vocals_explicit",
         "vocals_1_explicit", "vocals_2_explicit", "crowd_explicit", "kick", "snare", "other"}
OUTPUTS = ("drums_1.opus", "drums_2.opus", "drums_3.opus", "vocals.opus",
           "guitar.opus", "bass.opus", "keys.opus", "song.opus")
OUTPUTS_8STEM = OUTPUTS
OUTPUTS_6STEM = ("drums.opus", "vocals.opus", "guitar.opus", "bass.opus", "keys.opus", "song.opus")
WAV_OUTPUTS = tuple(name.replace(".opus", ".wav") for name in OUTPUTS)
WAV_OUTPUTS_6STEM = tuple(name.replace(".opus", ".wav") for name in OUTPUTS_6STEM)


def digest(path):
    if path.is_symlink():
        raise ValueError(f"Symlink is not supported: {path}")
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def remove_stage(stage):
    for path in stage.iterdir():
        if path.name != "job.json":
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    (stage / "job.json").unlink(missing_ok=True)
    stage.rmdir()


def pending_stage(folder):
    stage = folder / ".yams-work"
    return stage if stage.exists() else folder / ".yass-work"


def write_record(stage, record):
    temporary = stage / "job.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, stage / "job.json")
    sync_directory(stage)


def restore_file(source, target, expected_hash):
    source = Path(source)
    target = Path(target)
    if digest(source) != expected_hash:
        raise ValueError(f"Recovery backup changed; recovery retained: {source}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".yams-restore-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target_file, source.open("rb") as source_file:
            shutil.copyfileobj(fsrc=source_file, fdst=target_file)
            target_file.flush()
            os.fsync(target_file.fileno())
        shutil.copystat(source, temporary)
        if digest(temporary) != expected_hash:
            raise ValueError(f"Recovery copy changed; recovery retained: {temporary}")
        os.replace(temporary, target)
        sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def copy_file_exclusive(source, target):
    source = Path(source)
    target = Path(target)
    identity = None
    try:
        with source.open("rb") as source_file, target.open("xb") as target_file:
            status = os.fstat(target_file.fileno())
            identity = [status.st_dev, status.st_ino]
            shutil.copyfileobj(fsrc=source_file, fdst=target_file)
            target_file.flush()
            os.fsync(target_file.fileno())
        shutil.copystat(source, target)
    except BaseException:
        if identity is not None and target.exists() and not target.is_symlink():
            status = target.stat()
            if identity == [status.st_dev, status.st_ino]:
                target.unlink()
        raise


def publish_staged_tree(stage, payload, destination, record):
    stage = Path(stage)
    payload = Path(payload)
    destination = Path(destination)
    paths = sorted(payload.rglob("*"), key=lambda path: path.as_posix())
    if any(path.is_symlink() for path in paths):
        raise ValueError("Staged output trees must not contain symlinks")
    for path in paths:
        if path.is_file():
            with path.open("rb") as file:
                os.fsync(file.fileno())
            sync_directory(path.parent)
    files = {path.relative_to(payload).as_posix(): digest(path) for path in paths if path.is_file()}
    if not files:
        raise ValueError("Cannot publish an empty output tree")

    root = destination.resolve()
    directories = set()
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid staged output path")
        target = destination / relative
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Output already exists: {target}")
        if not target.parent.resolve().is_relative_to(root):
            raise ValueError("Staged output path escapes its destination")
        for parent in target.parents:
            if parent == destination:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f"Expected an ordinary output directory: {parent}")
            if not parent.exists():
                directories.add(parent.relative_to(destination).as_posix())

    record["files"] = files
    record["directories"] = sorted(directories, key=lambda name: (len(Path(name).parts), name))
    record["installed"] = {}
    record["state"] = "publishing"
    write_record(stage, record)
    for name in files:
        source = payload / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        record["installing"] = name
        record["installing_identity"] = None
        write_record(stage, record)
        created = False
        try:
            try:
                os.link(src=source, dst=target)
                created = True
                status = target.stat()
                record["installing_identity"] = [status.st_dev, status.st_ino]
                write_record(stage, record)
            except OSError:
                if target.exists() or target.is_symlink():
                    raise
                with source.open("rb") as source_file, target.open("xb") as target_file:
                    created = True
                    status = os.fstat(target_file.fileno())
                    record["installing_identity"] = [status.st_dev, status.st_ino]
                    write_record(stage, record)
                    shutil.copyfileobj(fsrc=source_file, fdst=target_file)
                    target_file.flush()
                    os.fsync(target_file.fileno())
                shutil.copystat(source, target)
            sync_directory(target.parent)
            status = target.stat()
            record["installed"][name] = [status.st_dev, status.st_ino]
            record.pop("installing")
            record.pop("installing_identity")
            write_record(stage, record)
        except BaseException:
            if created and target.exists() and not target.is_symlink():
                status = target.stat()
                identity = record.get("installing_identity") or record.get("installed", {}).get(name)
                if identity == [status.st_dev, status.st_ino]:
                    target.unlink()
                    sync_directory(target.parent)
            raise


def rollback_staged_tree(payload, destination, record):
    payload = Path(payload)
    destination = Path(destination)
    if payload.is_symlink() or not payload.is_dir():
        raise ValueError("Staged recovery payload must be an ordinary directory")
    root = destination.resolve()
    installed = record.get("installed", {})
    for name, expected in record.get("files", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid staged recovery path")
        target = destination / relative
        if not target.parent.resolve().is_relative_to(root):
            raise ValueError("Staged recovery path escapes its destination")
        for parent in target.parents:
            if parent == destination:
                break
            if parent.is_symlink():
                raise ValueError(f"Output directory changed during recovery: {parent}")
        if target.exists() or target.is_symlink():
            source = payload / relative
            identity = installed.get(name)
            if record.get("installing") == name:
                identity = record.get("installing_identity") or identity
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"Published output changed; recovery retained: {target}")
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Staged output changed; recovery retained: {source}")
            status = target.stat()
            matches_identity = identity == [status.st_dev, status.st_ino]
            matches_source = os.path.samefile(source, target)
            if not (matches_identity or matches_source):
                raise ValueError(f"Published output changed; recovery retained: {target}")
            if digest(target) != expected:
                raise ValueError(f"Published output changed; recovery retained: {target}")
            target.unlink()
            sync_directory(target.parent)
    for name in sorted(record.get("directories", []), key=lambda path: len(Path(path).parts), reverse=True):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid staged recovery directory")
        directory = destination / relative
        if directory.is_symlink():
            raise ValueError(f"Output directory changed during recovery: {directory}")
        if directory.exists() and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            sync_directory(directory.parent)


@contextlib.contextmanager
def locked(folder):
    path = folder / ".yams.lock"
    if path.is_symlink():
        raise ValueError("The worker lock must not be a symlink")
    with path.open("a+b") as file:
        file.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                file.write(b"0")
                file.flush()
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt
                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file, fcntl.LOCK_UN)
            except OSError:
                pass


def discover(folder):
    if not folder.is_dir():
        raise ValueError("Provide an unpacked song folder")
    entries = list(folder.iterdir())
    names = [path.name.casefold() for path in entries]
    if len(names) != len(set(names)):
        raise ValueError("Case-colliding filenames require review")
    metadata = {path.name.casefold() for path in entries if path.is_file() and not path.is_symlink()}
    if "song.ini" not in metadata or not {"notes.mid", "notes.chart"}.intersection(metadata):
        raise ValueError("Expected song.ini and notes.mid or notes.chart")
    for name in ("backup", ".yams-work", ".yass-work"):
        path = folder / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Expected an ordinary directory: {path}")
        if name in names and not path.exists():
            raise ValueError(f"Case-sensitive reserved directory collision: {name}")
    audio = [path for path in entries if path.suffix.casefold() in FORMATS and path.stem.casefold() in STEMS]
    if any(path.is_symlink() or not path.is_file() for path in audio):
        raise ValueError("Active audio must be ordinary files")
    if any(path.stem.casefold() != "song" for path in audio):
        return None
    if len(audio) != 1:
        raise ValueError("Expected one unambiguous song audio file")
    source = audio[0]
    lock = folder / ".yams.lock"
    legacy_lock = folder / ".yass.lock"
    if lock.is_symlink() or (lock.exists() and not lock.is_file()) or (".yams.lock" in names and not lock.exists()):
        raise ValueError("Expected an ordinary .yams.lock file without case collisions")
    if legacy_lock.is_symlink() or (legacy_lock.exists() and not legacy_lock.is_file()) or (".yass.lock" in names and not legacy_lock.exists()):
        raise ValueError("Expected an ordinary .yass.lock file without case collisions")
    backup = folder / "backup"
    backup_names = {path.name.casefold() for path in backup.iterdir()} if backup.exists() else set()
    if source.name.casefold() in backup_names or "separation-manifest.json" in backup_names:
        raise ValueError("Backup collision: existing originals and manifests will not be overwritten")
    return source


def begin(folder, source):
    stage = folder / ".yams-work"
    stage.mkdir()
    record = {"version": 1, "state": "processing", "source": source.name, "source_sha256": digest(source),
              "backup_existed": (folder / "backup").exists(), "outputs": {}}
    write_record(stage, record)
    shutil.copy2(source, stage / "original")
    with (stage / "original").open("rb") as file:
        os.fsync(file.fileno())
    if digest(stage / "original") != record["source_sha256"] or digest(source) != record["source_sha256"]:
        raise ValueError("Source changed while creating the processing snapshot")
    sync_directory(stage)
    (stage / "outputs").mkdir()
    return stage, record


def prepare(stage, record, audio_report):
    record["audio"] = audio_report
    record["model_profile"] = audio_report.get("model_profile", "mega-8stem")
    outputs = tuple(audio_report["validation"]["files"].keys())
    record["outputs"] = {name: digest(stage / "outputs" / name) for name in outputs}
    if any(record["outputs"][name] != audio_report["validation"]["files"][name]["sha256"] for name in outputs):
        raise ValueError("Staged outputs changed after audio validation")
    for name in outputs:
        with (stage / "outputs" / name).open("rb") as file:
            os.fsync(file.fileno())
    sync_directory(stage / "outputs")
    record["state"] = "ready"
    with (stage / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump({**record, "state": "completed"}, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    record["manifest_sha256"] = digest(stage / "manifest.json")
    write_record(stage, record)


def install(folder, stage, record):
    source = folder / record["source"]
    if digest(source) != record["source_sha256"] or digest(stage / "original") != record["source_sha256"]:
        raise ValueError("Source changed before installation")
    for name in record["outputs"]:
        if digest(stage / "outputs" / name) != record["outputs"][name]:
            raise ValueError(f"Staged output changed before installation: {name}")
    backup = folder / "backup"
    if backup.is_symlink() or (backup.exists() and not backup.is_dir()):
        raise ValueError("Backup destination changed; installation stopped")
    backup.mkdir(exist_ok=True)
    os.link(stage / "original", backup / source.name)
    sync_directory(backup)
    sync_directory(folder)
    if digest(backup / source.name) != record["source_sha256"]:
        raise ValueError("Backup verification failed before source removal")
    os.rename(source, stage / "removed-original")
    sync_directory(stage)
    sync_directory(folder)
    if digest(stage / "removed-original") != record["source_sha256"]:
        raise ValueError("Source changed during installation")
    for name in record["outputs"]:
        os.link(stage / "outputs" / name, folder / name)
        sync_directory(folder)
    if digest(backup / source.name) != record["source_sha256"]:
        raise ValueError("Backup changed during installation")
    for name in record["outputs"]:
        if digest(folder / name) != record["outputs"][name]:
            raise ValueError(f"Output changed during installation: {name}")
    os.link(stage / "manifest.json", backup / "separation-manifest.json")
    sync_directory(backup)
    record["state"] = "committed"
    write_record(stage, record)
    remove_stage(stage)
    sync_directory(folder)


def begin_export(folder, destination, source):
    destination.mkdir(parents=True, exist_ok=True)
    stage = destination / ".yams-work"
    stage.mkdir()
    record = {"version": 1, "format": "folder_export", "mode": "export", "state": "processing",
              "source": source.name, "source_sha256": digest(source),
              "source_folder": str(folder), "destination_folder": str(destination), "backup_existed": False,
              "outputs": {}}
    write_record(stage, record)
    shutil.copy2(source, stage / "original")
    with (stage / "original").open("rb") as file:
        os.fsync(file.fileno())
    if digest(stage / "original") != record["source_sha256"]:
        raise ValueError("Source changed while creating the processing snapshot")
    sync_directory(stage)
    (stage / "outputs").mkdir()
    return stage, record


def install_export(folder, destination, stage, record):
    for name in record["outputs"]:
        if digest(stage / "outputs" / name) != record["outputs"][name]:
            raise ValueError(f"Staged output changed before installation: {name}")
    payload = stage / "payload"
    payload.mkdir()
    for path in folder.iterdir():
        if path.is_file() and not path.is_symlink():
            if path.suffix.casefold() not in FORMATS and not path.name.startswith((".yams", ".yass")):
                shutil.copy2(path, payload / path.name)
    for name in record["outputs"]:
        shutil.copy2(stage / "outputs" / name, payload / name)
    publish_staged_tree(stage=stage, payload=payload, destination=destination, record=record)
    sync_directory(destination)
    record["state"] = "committed"
    write_record(stage, record)
    remove_stage(stage)
    sync_directory(destination)


def abort_export(destination):
    stage = pending_stage(destination)
    if stage.exists() and stage.is_dir():
        journal = stage / "job.json"
        if not journal.is_file():
            journal = stage / "job.tmp"
        if journal.is_file():
            record = json.loads(journal.read_text(encoding="utf-8"))
            if record.get("format") != "folder_export" or record.get("version") != 1:
                raise ValueError("Unsupported folder export recovery record")
            if record.get("state") == "publishing":
                rollback_staged_tree(stage / "payload", destination, record)
        remove_stage(stage)
    if destination.exists() and destination.is_dir() and not any(destination.iterdir()):
        destination.rmdir()


def recover(folder):
    stage = pending_stage(folder)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("No ordinary pending work directory to recover")
    if not any(stage.iterdir()):
        stage.rmdir()
        sync_directory(folder)
        return "discarded_empty_staging"
    journal = stage / "job.json"
    if not journal.exists():
        journal = stage / "job.tmp"
    record = json.loads(journal.read_text(encoding="utf-8"))
    name = record["source"]
    if Path(name).name != name or Path(name).stem.casefold() != "song" or Path(name).suffix.casefold() not in FORMATS:
        raise ValueError("Invalid source filename in recovery journal")
    if record["version"] != 1 or record["state"] not in {"processing", "ready", "committed", "rolled_back"}:
        raise ValueError("Unsupported recovery journal")
    if not re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"]):
        raise ValueError("Invalid source hash in recovery journal")
    if record["state"] == "processing":
        if (stage / "removed-original").exists():
            raise ValueError("Unexpected original movement in processing state")
        if digest(folder / name) != record["source_sha256"]:
            raise ValueError("Source changed; the processing snapshot has been retained")
        remove_stage(stage)
        sync_directory(folder)
        return "discarded_staging"
    if set(record["outputs"]) not in (set(OUTPUTS_8STEM), set(OUTPUTS_6STEM),
                                      set(WAV_OUTPUTS), set(WAV_OUTPUTS_6STEM)):
        raise ValueError("Invalid output names in recovery journal")
    if record["state"] == "rolled_back":
        if digest(folder / name) != record["restored_source_sha256"]:
            raise ValueError("Restored source changed; recovery files have been retained")
        remove_stage(stage)
        sync_directory(folder)
        return "completed_rollback_cleanup"
    backup = folder / "backup"
    if backup.is_symlink():
        raise ValueError("Backup directory must not be a symlink")
    original_backup = backup / name
    manifest = backup / "separation-manifest.json"
    if original_backup.exists() and digest(original_backup) != record["source_sha256"]:
        raise ValueError("Backup changed; recovery files have been retained")
    if manifest.exists() and digest(manifest) != record["manifest_sha256"]:
        raise ValueError("Manifest changed; recovery files have been retained")
    moved = stage / "removed-original"
    if record["state"] == "committed":
        if not original_backup.exists() or not manifest.exists():
            raise ValueError("Committed backup is incomplete; recovery files have been retained")
        for output, expected in record["outputs"].items():
            if digest(folder / output) != expected:
                raise ValueError("Installed output changed; recovery files have been retained")
        remove_stage(stage)
        sync_directory(folder)
        return "completed_cleanup"
    if original_backup.exists() and not os.path.samefile(original_backup, stage / "original"):
        raise ValueError("Backup belongs to another file; recovery files have been retained")
    if manifest.exists() and not os.path.samefile(manifest, stage / "manifest.json"):
        raise ValueError("Manifest belongs to another file; recovery files have been retained")
    if moved.exists():
        source = folder / name
        restored = source.exists() and not source.is_symlink() and os.path.samefile(moved, source)
        if not restored:
            for output, expected in record["outputs"].items():
                path = folder / output
                if path.exists() or path.is_symlink():
                    if digest(path) != expected or not os.path.samefile(path, stage / "outputs" / output):
                        raise ValueError(f"Output changed or belongs to another file; recovery files retained: {path}")
            if source.exists() or source.is_symlink():
                source_is_output = any(name.casefold() == output and not source.is_symlink()
                                       and (folder / output).exists() and os.path.samefile(source, folder / output)
                                       for output in record["outputs"])
                if not source_is_output:
                    raise ValueError("Source path occupied; recovery files have been retained")
            for output in record["outputs"]:
                path = folder / output
                if path.exists():
                    path.unlink()
            os.link(moved, source)
            with source.open("rb") as file:
                os.fsync(file.fileno())
            sync_directory(folder)
    elif digest(folder / name) != record["source_sha256"]:
        raise ValueError("Source changed; the processing snapshot has been retained")
    if manifest.exists():
        manifest.unlink()
    changed = moved.exists() and digest(moved) != record["source_sha256"]
    if original_backup.exists() and not changed:
        original_backup.unlink()
    if not record["backup_existed"] and backup.exists() and not any(backup.iterdir()):
        backup.rmdir()
    record["restored_source_sha256"] = digest(folder / name)
    record["state"] = "rolled_back"
    write_record(stage, record)
    remove_stage(stage)
    sync_directory(folder)
    return "rolled_back"
