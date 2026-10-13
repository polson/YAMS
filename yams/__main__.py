import argparse
import contextlib
import json
import platform
import sys
import threading
from pathlib import Path

import struct

from .folders import abort_export, begin, begin_export, discover, install, install_export, locked, pending_stage, prepare, recover, OUTPUTS, OUTPUTS_6STEM
from .confile import is_con_file, is_excon_folder, find_excon_info, ConPackage, is_con_multitrack
from .audio_file import is_audio_file, is_multitrack_folder, find_single_audio_in_folder, process_audio_file, abort_audio_process


def has_work_marker(path):
    return pending_stage(path).exists()


def preview(folder):
    fields = {"folder": str(folder), "source": "", "backup": "", "message": ""}
    if is_con_file(folder):
        sibling = folder.parent / (folder.stem if folder.suffix else f"{folder.name}_unpacked")
        if has_work_marker(sibling):
            return {**fields, "status": "recovery_required", "message": "Interrupted work needs recovery"}
        try:
            with ConPackage(folder) as pkg:
                dta_listing = pkg.find("songs.dta")
                mogg_listing = pkg.find(".mogg")
                if not dta_listing or not mogg_listing:
                    return {**fields, "status": "skipped", "message": "Not a Rock Band song package"}
                mogg_bytes = pkg.read_file(mogg_listing)
                if len(mogg_bytes) < 8 or struct.unpack("<I", mogg_bytes[:4])[0] != 10:
                    return {**fields, "status": "skipped", "message": "Encrypted or unsupported MOGG"}
                dta_text = pkg.read_file(dta_listing).decode("latin1", errors="replace")
                if is_con_multitrack(dta_text, mogg_bytes):
                    return {**fields, "status": "skipped", "message": "Already contains multitrack stems"}
                return {**fields, "status": "ready", "source": folder.name,
                        "backup": f"backup/{folder.name}", "bytes": folder.stat().st_size}
        except Exception as error:
            return {**fields, "status": "error", "message": str(error)}
    if is_excon_folder(folder):
        if has_work_marker(folder):
            return {**fields, "status": "recovery_required", "message": "Interrupted work needs recovery"}
        try:
            info = find_excon_info(folder)
            mogg = info["mogg_path"]
            if not mogg or not mogg.exists():
                return {**fields, "status": "skipped", "message": "Missing .mogg in exCON folder"}
            with mogg.open("rb") as f:
                mogg_header = f.read(8)
            if len(mogg_header) < 8 or struct.unpack("<I", mogg_header[:4])[0] != 10:
                return {**fields, "status": "skipped", "message": "Encrypted or unsupported MOGG"}
            dta_path = info["dta_path"]
            dta_text = dta_path.read_text(encoding="utf-8", errors="replace")
            if is_con_multitrack(dta_text, mogg):
                return {**fields, "status": "skipped", "message": "Already contains multitrack stems"}
            return {**fields, "status": "ready", "source": mogg.name,
                    "backup": f"backup/{mogg.name}", "bytes": mogg.stat().st_size}
        except Exception as error:
            return {**fields, "status": "error", "message": str(error)}
    if is_audio_file(folder):
        sibling = folder.parent / folder.stem
        if has_work_marker(sibling):
            return {**fields, "status": "recovery_required", "message": "Interrupted work needs recovery"}
        return {**fields, "status": "ready", "source": folder.name,
                "backup": f"backup/{folder.name}", "bytes": folder.stat().st_size}
    if is_multitrack_folder(folder):
        return {**fields, "status": "skipped", "message": "Already contains multitrack stems"}
    single_audio = find_single_audio_in_folder(folder)
    if single_audio is not None and not (folder / "song.ini").exists():
        if has_work_marker(folder):
            return {**fields, "status": "recovery_required", "message": "Interrupted work needs recovery"}
        return {**fields, "status": "ready", "source": single_audio.name,
                "backup": f"backup/{single_audio.name}", "bytes": single_audio.stat().st_size}
    if has_work_marker(folder):
        return {**fields, "status": "recovery_required", "message": "Interrupted work needs recovery"}
    source = discover(folder)
    if source is None:
        return {**fields, "status": "skipped", "message": "Existing instrument or alternate stems"}
    return {**fields, "status": "ready", "source": source.name,
            "backup": f"backup/{source.name}", "bytes": source.stat().st_size}


def recover_transaction(folder, output_folder=None):
    folder = Path(folder)
    if output_folder is not None:
        destinations = [Path(output_folder).resolve()]
    elif folder.is_dir():
        destinations = [folder]
    elif folder.suffix.casefold() == ".con":
        name = folder.stem if folder.suffix else f"{folder.name}_unpacked"
        destinations = [folder.parent / name]
    elif folder.suffix:
        destinations = [folder.parent / folder.stem]
    else:
        destinations = [folder.parent / f"{folder.name}_unpacked", folder]

    for destination in destinations:
        if not destination.is_dir():
            continue
        if not pending_stage(destination).exists():
            continue
        with locked(destination):
            stage = pending_stage(destination)
            if stage.is_symlink():
                raise ValueError("Pending work directory must not be a symlink")
            journal = stage / "job.json"
            if not journal.is_file():
                journal = stage / "job.tmp"
            if journal.is_symlink():
                raise ValueError("Pending work journal must not be a symlink")
            if not journal.is_file():
                continue
            record = json.loads(journal.read_text(encoding="utf-8"))
            if not folder.is_dir() and record.get("format") in {"audio_file", "con_file"}:
                if Path(record["source"]).resolve() != folder.resolve():
                    continue
            if record.get("format") == "folder_export":
                if Path(record["source_folder"]).resolve() != folder.resolve():
                    continue
            mode = record.get("mode")
            if record.get("format") == "audio_file":
                abort_audio_process(destination=destination, mode=mode, audio_path=record["source"])
            elif record.get("format") == "con_file":
                from .confile import abort_con_process
                abort_con_process(destination=destination, con_path=record["source"], mode=mode)
            elif record.get("format") == "excon":
                from .confile import abort_excon_process
                abort_excon_process(destination=destination, mode=mode)
            elif record.get("format") == "folder_export":
                abort_export(destination)
            else:
                continue
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Separate one INI song in place with verified original backups")
    parser.add_argument("folder", type=Path, nargs="?")
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--inference-source", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0", "mps"), default="auto")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-list", action="store_true")
    parser.add_argument("--control-stdin", action="store_true")
    parser.add_argument("--runtime-installer", type=Path)
    parser.add_argument("--runtime-resources", type=Path)
    parser.add_argument("--model-profile", choices=("mega-8stem", "8stem", "torch-8stem", "mlx-8stem", "mlx-6stem", "torch-6stem"), default="mega-8stem")
    args = parser.parse_args()
    if args.model_profile in ("mlx-6stem", "mlx-8stem") and (sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64")):
        parser.error(f"The {args.model_profile} model profile is only supported on macOS Apple Silicon")
    if args.folder is None and not args.inspect_list:
        parser.error("Provide a song folder")
    if args.inspect_list and args.control_stdin:
        parser.error("Batch inspection cannot use worker control input")
    folder = (args.folder or Path.cwd()).resolve()
    event_stream = sys.stdout
    cancelled = threading.Event()

    def read_control():
        for line in sys.stdin:
            if line.strip() == "cancel":
                break
        cancelled.set()

    if args.control_stdin:
        threading.Thread(target=read_control, daemon=True).start()

    def emit(event, **fields):
        print(json.dumps({"event": event, "folder": str(folder), **fields}), file=event_stream, flush=True)

    def emit_progress(event, **fields):
        if cancelled.is_set():
            raise KeyboardInterrupt
        emit(event, **fields)

    try:
        if args.inspect_list:
            paths = json.load(sys.stdin)
            if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                raise ValueError("Provide a JSON list of song folder paths on stdin")
            songs = []
            for path in paths:
                candidate = Path(path).resolve()
                try:
                    songs.append(preview(candidate))
                except Exception as error:
                    songs.append({"folder": str(candidate), "status": "error", "message": str(error),
                                  "source": "", "backup": ""})
            emit("previews", songs=songs)
            return 0
        if args.inspect:
            fields = preview(folder)
            del fields["folder"]
            emit("preview", **fields)
            return 0
        if args.recover:
            if recover_transaction(folder, output_folder=args.output_folder):
                emit("recovered", result="discarded_staging")
                return 0
            if is_con_file(folder):
                sibling = folder.parent / (folder.stem if folder.suffix else f"{folder.name}_unpacked")
                from .confile import abort_con_process
                if sibling.exists():
                    with locked(sibling):
                        abort_con_process(sibling, folder, "inplace")
                emit("recovered", result="discarded_staging")
                return 0
            if is_excon_folder(folder):
                from .confile import abort_excon_process
                with locked(folder):
                    abort_excon_process(folder, "inplace")
                emit("recovered", result="discarded_staging")
                return 0
            if is_audio_file(folder):
                sibling = folder.parent / folder.stem
                if sibling.exists():
                    with locked(sibling):
                        abort_audio_process(destination=sibling, mode="inplace", audio_path=folder)
                emit("recovered", result="discarded_staging")
                return 0
            if is_multitrack_folder(folder):
                emit("skipped", reason="already_contains_multitrack_stems")
                return 0
            single_audio = find_single_audio_in_folder(folder)
            if single_audio is not None and not (folder / "song.ini").exists():
                with locked(folder):
                    abort_audio_process(destination=folder, mode="inplace", audio_path=single_audio)
                emit("recovered", result="discarded_staging")
                return 0
            with locked(folder):
                result = recover(folder)
                emit("recovered", result=result)
            return 0
        if is_multitrack_folder(folder):
            emit("skipped", reason="already_contains_multitrack_stems")
            return 0
        if is_con_file(folder):
            try:
                with ConPackage(folder) as pkg:
                    dta_listing = pkg.find("songs.dta")
                    mogg_listing = pkg.find(".mogg")
                    if dta_listing and mogg_listing:
                        dta_text = pkg.read_file(dta_listing).decode("latin1", errors="replace")
                        mogg_bytes = pkg.read_file(mogg_listing)
                        if is_con_multitrack(dta_text, mogg_bytes):
                            emit("skipped", reason="already_contains_multitrack_stems")
                            return 0
            except Exception:
                pass
        if is_excon_folder(folder):
            try:
                info = find_excon_info(folder)
                dta_text = info["dta_path"].read_text(encoding="utf-8", errors="replace")
                if is_con_multitrack(dta_text, info["mogg_path"]):
                    emit("skipped", reason="already_contains_multitrack_stems")
                    return 0
            except Exception:
                pass
        single_audio = find_single_audio_in_folder(folder)
        direct_audio = is_audio_file(folder)
        audio_source = folder if direct_audio else single_audio
        if not is_con_file(folder) and not is_excon_folder(folder) and not direct_audio and single_audio is None and not has_work_marker(folder):
            source = discover(folder)
            if source is None:
                emit("skipped", reason="existing_instrument_or_alternate_stems")
                return 0
        if args.runtime_installer is not None:
            if args.runtime_resources is None:
                raise ValueError("Provide --runtime-resources with --runtime-installer")
            from .runtime import ensure_runtime
            try:
                ensure_runtime(installer=args.runtime_installer, resources=args.runtime_resources,
                               emit=emit_progress, cancelled=cancelled)
            except Exception as error:
                emit("runtime_error", message=str(error))
                raise
        from .models import default_models, ensure_models
        models = (args.models or default_models(profile=args.model_profile)).resolve()
        inference_source = args.inference_source.resolve() if args.inference_source else (Path(__file__).resolve().parent / "inference")
        if args.models is None or args.download_models:
            try:
                ensure_models(folder=models, emit=emit_progress, profile=args.model_profile)
            except Exception as error:
                emit("models_error", message=str(error))
                raise
        expected_outputs = OUTPUTS_6STEM if args.model_profile in ("mlx-6stem", "torch-6stem") else OUTPUTS
        if is_con_file(folder):
            from .confile import process_con_song, abort_con_process
            mode = "export" if args.output_folder is not None else "inplace"
            if mode == "export":
                destination = args.output_folder.resolve()
            else:
                destination = folder.parent / (folder.stem if folder.suffix else f"{folder.name}_unpacked")
            if destination.exists() and ((destination / "separation-manifest.json").is_file() or (destination / "songs/songs.dta").is_file()):
                emit("skipped", reason="already_separated_in_destination")
                return 0
            destination.mkdir(parents=True, exist_ok=True)
            with locked(destination):
                try:
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    emit("started", source=folder.name, destination=str(destination))
                    with contextlib.redirect_stdout(sys.stderr):
                        result = process_con_song(
                            con_path=folder,
                            destination=destination,
                            mode=mode,
                            models=models,
                            inference_source=inference_source,
                            device=args.device,
                            emit=emit_progress,
                            cancelled=cancelled,
                            model_profile=args.model_profile,
                        )
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    backup_info = f"backup/{folder.name}" if mode == "inplace" else ""
                    emit("completed", destination=str(destination), backup=backup_info, validation=result.get("validation", {}))
                    return 0
                except BaseException:
                    abort_con_process(destination, folder, mode)
                    raise
        if is_excon_folder(folder):
            from .confile import process_excon_song, abort_excon_process
            mode = "export" if args.output_folder is not None else "inplace"
            if mode == "export":
                destination = args.output_folder.resolve()
            else:
                destination = folder.resolve()
            if destination.exists() and ((destination / "separation-manifest.json").is_file() or (destination / "songs/songs.dta").is_file()):
                emit("skipped", reason="already_separated_in_destination")
                return 0
            if mode == "inplace" and ((destination / "backup" / "separation-manifest.json").is_file() or (destination / "songs/songs.dta").is_file()):
                emit("skipped", reason="already_separated_in_destination")
                return 0
            destination.mkdir(parents=True, exist_ok=True)
            with locked(destination):
                try:
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    info = find_excon_info(folder)
                    mogg = info["mogg_path"]
                    emit("started", source=mogg.name, destination=str(destination))
                    with contextlib.redirect_stdout(sys.stderr):
                        result = process_excon_song(
                            excon_path=folder,
                            destination=destination,
                            mode=mode,
                            models=models,
                            inference_source=inference_source,
                            device=args.device,
                            emit=emit_progress,
                            cancelled=cancelled,
                            model_profile=args.model_profile,
                        )
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    backup_info = f"backup/{mogg.name}" if mode == "inplace" else ""
                    emit("completed", destination=str(destination), backup=backup_info, validation=result.get("validation", {}))
                    return 0
                except BaseException:
                    abort_excon_process(destination, mode)
                    raise
        if audio_source is not None:
            mode = "export" if args.output_folder is not None else "inplace"
            if mode == "export":
                destination = args.output_folder.resolve()
            elif direct_audio:
                destination = folder.parent / folder.stem
            else:
                destination = folder.resolve()
            if destination.exists() and ((destination / "separation-manifest.json").is_file() or all((destination / out).is_file() for out in expected_outputs)):
                emit("skipped", reason="already_separated_in_destination")
                return 0
            if mode == "inplace" and not direct_audio and (destination / "backup" / "separation-manifest.json").is_file():
                emit("skipped", reason="already_separated_in_destination")
                return 0
            destination.mkdir(parents=True, exist_ok=True)
            with locked(destination):
                try:
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    emit("started", source=audio_source.name, destination=str(destination))
                    with contextlib.redirect_stdout(sys.stderr):
                        result = process_audio_file(
                            audio_path=audio_source,
                            destination=destination,
                            mode=mode,
                            models=models,
                            inference_source=inference_source,
                            device=args.device,
                            emit=emit_progress,
                            cancelled=cancelled,
                            model_profile=args.model_profile,
                        )
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    backup_info = f"backup/{audio_source.name}" if mode == "inplace" else ""
                    emit("completed", destination=str(destination), backup=backup_info, validation=result.get("validation", {}))
                    return 0
                except BaseException:
                    abort_audio_process(destination=destination, mode=mode, audio_path=audio_source)
                    raise
        if args.output_folder is not None:
            destination = args.output_folder.resolve()
            if destination.exists() and all((destination / out).is_file() for out in expected_outputs):
                emit("skipped", reason="already_separated_in_destination")
                return 0
            source = discover(folder)
            if source is None:
                emit("skipped", reason="existing_instrument_or_alternate_stems")
                return 0
            destination.mkdir(parents=True, exist_ok=True)
            with locked(destination):
                try:
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    stage, record = begin_export(folder, destination, source)
                    emit("started", source=source.name, destination=str(destination))
                    with contextlib.redirect_stdout(sys.stderr):
                        from .audio import separate
                        result = separate(stage=stage, models=models, inference_source=inference_source,
                                          device_name=args.device, emit=emit_progress,
                                          model_profile=args.model_profile, cancelled=cancelled)
                    if cancelled.is_set():
                        raise KeyboardInterrupt
                    prepare(stage, record, result)
                    emit("installing")
                    install_export(folder, destination, stage, record)
                    emit("completed", destination=str(destination), validation=result["validation"])
                    return 0
                except BaseException:
                    abort_export(destination)
                    raise
        with locked(folder):
            if has_work_marker(folder):
                recover(folder)
            source = discover(folder)
            if source is None:
                emit("skipped", reason="existing_instrument_or_alternate_stems")
                return 0
            try:
                if cancelled.is_set():
                    raise KeyboardInterrupt
                stage, record = begin(folder, source)
                emit("started", source=source.name)
                with contextlib.redirect_stdout(sys.stderr):
                    from .audio import separate
                    result = separate(stage=stage, models=models, inference_source=inference_source,
                                      device_name=args.device, emit=emit_progress,
                                      model_profile=args.model_profile, cancelled=cancelled)
                if cancelled.is_set():
                    raise KeyboardInterrupt
                prepare(stage, record, result)
                emit("installing")
                install(folder, stage, record)
                emit("completed", backup=f"backup/{source.name}", validation=result["validation"])
                return 0
            except BaseException:
                if has_work_marker(folder):
                    try:
                        recovery = recover(folder)
                        emit("recovered", result=recovery)
                        if recovery == "completed_cleanup":
                            emit("completed", backup=f"backup/{source.name}", validation=result["validation"])
                            return 0
                    except Exception as recovery_error:
                        emit("recovery_required", message=str(recovery_error))
                raise
    except KeyboardInterrupt:
        emit("cancelled")
        return 130
    except Exception as error:
        emit("error", message=str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
