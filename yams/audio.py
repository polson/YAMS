import gc
import importlib
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
try:
    import torch
except ImportError:
    torch = None

from .folders import OUTPUTS, WAV_OUTPUTS, OUTPUTS_6STEM, WAV_OUTPUTS_6STEM, digest


OPUS_RATE = 48000
OPUS_BITRATE = 192000
STEM_MAPPING_6STEM = {
    "drums": "drums.wav",
    "vocals": "vocals.wav",
    "guitar": "guitar.wav",
    "bass": "bass.wav",
    "piano": "keys.wav",
}


def convert(source, destination, rate):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
         "-map", "0:a:0", "-vn", "-af", f"aresample={rate}:resampler=soxr:precision=28",
         "-c:a", "pcm_f32le", str(destination)], check=True,
    )


def _decode_reference(stage):
    reference_path = stage / "reference.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(stage / "original"),
         "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", str(reference_path)], check=True,
    )
    reference, rate = sf.read(file=reference_path, dtype="float32", always_2d=True)
    if len(reference) == 0 or reference.shape[1] not in (1, 2) or not np.isfinite(reference).all():
        raise ValueError("Expected finite mono or stereo audio")
    return reference_path, reference, rate


def restore_timeline(audio, sample_rate, reference, rate, name):
    tail = len(audio) - len(reference)
    if sample_rate != rate or abs(tail) > math.ceil(rate / 44100):
        raise ValueError(f"Unexpected resampling timeline: {name}")
    audio = audio[:len(reference)]
    if tail < 0:
        audio = np.pad(audio, ((0, -tail), (0, 0)))
    if reference.shape[1] == 1:
        audio = audio.mean(axis=1, keepdims=True)
    return audio, tail


def _audio_file_report(audio, path):
    return {
        "peak": float(np.max(np.abs(audio))),
        "samples_above_full_scale": int(np.count_nonzero(np.abs(audio) > 1)),
        "sha256": digest(path),
    }


def _is_out_of_memory(error):
    err_str = str(error).lower()
    return (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in err_str
        or "dml allocator" in err_str
        or "e_outofmemory" in err_str
        or "hiperrormemoryallocation" in err_str
        or "device_removed" in err_str
        or "device_reset" in err_str
    )


def reconstruct(reference, outputs, rate, filenames=WAV_OUTPUTS):
    combined = np.zeros(reference.shape, dtype=np.float64)
    files = {}
    for name in filenames:
        path = outputs / name
        audio, stem_rate = sf.read(file=path, dtype="float64", always_2d=True)
        if stem_rate != rate or audio.shape != reference.shape or not np.isfinite(audio).all():
            raise ValueError(f"Invalid exported timeline or samples: {name}")
        if sf.info(path).subtype != "FLOAT":
            raise ValueError(f"Expected floating-point WAV: {name}")
        combined += audio
        files[name] = _audio_file_report(audio, path)
    difference = combined - reference.astype(np.float64)
    peak = float(np.max(np.abs(difference)))
    if peak > 1e-6:
        raise ValueError(f"Exported reconstruction failed: peak error {peak}")
    return {"peak_error": peak, "rms_error": float(np.sqrt(np.mean(difference ** 2))), "files": files}


def export_opus(reference_path, pcm, outputs, emit, filenames=OUTPUTS):
    reference_export = pcm.parent / "opus-reference.wav"
    decoded = pcm.parent / "opus-decoded.wav"
    convert(source=reference_path, destination=reference_export, rate=OPUS_RATE)
    reference, rate = sf.read(file=reference_export, dtype="float64", always_2d=True)
    combined = np.zeros(reference.shape, dtype=np.float64)
    files = {}
    total = len(filenames)
    for index, name in enumerate(filenames):
        emit("encoding", asset=name, index=index + 1, total=total)
        path = outputs / name
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(pcm / name.replace(".opus", ".wav")),
             "-map", "0:a:0", "-vn", "-af", f"aresample={OPUS_RATE}:resampler=soxr:precision=28",
             "-c:a", "libopus", "-b:a", str(OPUS_BITRATE), "-vbr", "on",
             "-application", "audio", "-compression_level", "10", "-f", "opus", str(path)], check=True,
        )
        convert(source=path, destination=decoded, rate=OPUS_RATE)
        audio, stem_rate = sf.read(file=decoded, dtype="float64", always_2d=True)
        if stem_rate != rate or audio.shape != reference.shape or not np.isfinite(audio).all():
            raise ValueError(f"Invalid Opus timeline or samples: {name}")
        combined += audio
        files[name] = _audio_file_report(audio, path)
    difference = combined - reference
    return {"codec": "opus", "encoder": "libopus", "bitrate": OPUS_BITRATE, "vbr": True,
            "sample_rate": rate, "frames": len(reference), "channels": reference.shape[1], "lossy": True,
            "peak_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(difference ** 2))), "files": files}


if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def budget_gpu(model, device):
    if torch.cuda.get_allocator_backend() != "native" or os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":
        raise ValueError("GPU memory protection requires PyTorch's native caching allocator. Remove custom allocator settings or choose CPU.")
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    nominal_headroom = max(1024**3, total // 5)
    model_bytes = sum(tensor.numel() * tensor.element_size()
                      for tensor in (*model.parameters(), *model.buffers()))
    minimum = model_bytes + 512 * 1024**2
    desirable_workspace = model_bytes + 1536 * 1024**2
    if free - nominal_headroom >= desirable_workspace:
        headroom = nominal_headroom
    else:
        min_headroom = min(nominal_headroom, max(512 * 1024**2, min(1024**3, total // 10)))
        headroom = max(min_headroom, min(nominal_headroom, free - desirable_workspace))
    budget = free - headroom
    if budget < minimum:
        raise ValueError(f"Insufficient safe GPU memory: model plus minimum workspace needs {minimum / 1024**3:.2f} GiB; "
                         f"{max(0, budget) / 1024**3:.2f} GiB available after reserving desktop memory. Close GPU apps or choose CPU.")
    torch.cuda.set_per_process_memory_fraction(fraction=budget / total, device=device)


def initial_chunk_size(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        if free >= 6 * 1024**3:
            return 882000
        if free >= 3 * 1024**3:
            return 441000
        return 220500
    return 441000


DEFAULT_INFERENCE = Path(__file__).resolve().parent / "inference"


def _load_inference_runtime(inference_source, device_name):
    for path in (inference_source / "utils/model_utils.py", inference_source / "utils/settings.py"):
        if not path.is_file():
            raise ValueError(f"Missing upstream inference source: {path}")
    source_str = str(inference_source)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    model_utils = importlib.import_module("utils.model_utils")
    settings = importlib.import_module("utils.settings")
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this runtime")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS is unavailable in this runtime")
    return model_utils, settings, device


def separate_torch_6stem(stage, models, inference_source=None, device_name="auto", emit=None, cancelled=None):
    if emit is None:
        emit = lambda *args, **kwargs: None
    if inference_source is None:
        inference_source = DEFAULT_INFERENCE
    model_utils, settings, device = _load_inference_runtime(inference_source, device_name)
    config_path = models / "BS-Rofo-SW-Fixed.yaml"
    weights_path = models / "BS-Rofo-SW-Fixed.ckpt"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"Missing 6-stem model files in {models}")
    assets = {
        config_path.name: digest(config_path),
        weights_path.name: digest(weights_path),
    }
    emit("device", name=str(device))
    reference_path, reference, rate = _decode_reference(stage)
    stereo_bytes = len(reference) * 2 * 4
    model_bytes = math.ceil(len(reference) * 44100 / rate) * 2 * 4
    opus_bytes = math.ceil(len(reference) * OPUS_RATE / rate) * reference.shape[1] * 4
    required = 6 * stereo_bytes + 6 * model_bytes + reference.nbytes + 8 * opus_bytes + 64 * 1024**2
    if shutil.disk_usage(stage).free < required:
        raise ValueError(f"Insufficient working space: need approximately {required} more bytes")
    emit("decoded", sample_rate=rate, frames=len(reference), channels=reference.shape[1])

    started = time.monotonic()
    assets, elapsed, chunks = run_torch_6stem_pcm(
        stage=stage, models=models, reference=reference, rate=rate,
        settings=settings, model_utils=model_utils, device=device,
        emit=emit, cancelled=cancelled, total_passes=1,
    )
    pcm = stage / "pcm"
    pcm_validation = reconstruct(reference=reference, outputs=pcm, rate=rate, filenames=WAV_OUTPUTS_6STEM)
    emit("validated", pass_number=1, codec="pcm_f32le", **pcm_validation)
    outputs = stage / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    validation = export_opus(reference_path=reference_path, pcm=pcm, outputs=outputs,
                             emit=emit, filenames=OUTPUTS_6STEM)
    emit("validated", **validation)

    runs = [{"target": "6stem", "pass_number": 1, "seconds": elapsed,
             "chunks": len(chunks), "raw_chunks_finite": True, "tail_rounding_frames": 0}]
    return {
        "sample_rate": rate, "frames": len(reference), "channels": reference.shape[1],
        "device": str(device), "model_type": "bs_roformer_torch", "model_profile": "torch-6stem",
        "model_assets_sha256": assets, "inference_source": str(inference_source),
        "model_runs": runs, "pcm_validation": pcm_validation,
        "validation": validation,
        "seconds": time.monotonic() - started, "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def run_torch_6stem_pcm(stage, models, reference, rate, settings, model_utils, device, emit, cancelled, total_passes=1):
    config_path = models / "BS-Rofo-SW-Fixed.yaml"
    weights_path = models / "BS-Rofo-SW-Fixed.ckpt"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"Missing 6-stem model files in {models}")
    assets = {
        config_path.name: digest(config_path),
        weights_path.name: digest(weights_path),
    }
    work = stage / "pass-1"
    work.mkdir(exist_ok=True)
    stereo = np.repeat(reference, 2, axis=1) if reference.shape[1] == 1 else reference
    sf.write(file=work / "stereo.wav", data=stereo, samplerate=rate, subtype="FLOAT")
    convert(work / "stereo.wav", work / "model-input.wav", rate=44100)
    model_input, _ = sf.read(file=work / "model-input.wav", dtype="float32", always_2d=True)

    model, config = settings.get_model_from_config("bs_roformer", str(config_path))
    if config.model.num_stems != 6:
        raise ValueError("Expected 6-stem BS-RoFormer configuration")
    if config.audio.sample_rate != 44100 or config.inference.normalize is not False:
        raise ValueError("Expected 44.1 kHz model input with normalization disabled")
    config.training.use_amp = bool(config.training.use_amp and device.type == "cuda")
    if device.type == "cuda":
        budget_gpu(model=model, device=device)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    try:
        model = model.to(device).eval()
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if _is_out_of_memory(error):
            raise ValueError("Model exceeded the safe GPU memory limit. Close GPU apps or choose CPU; the original song will be retained.") from error
        raise

    chunk_size = initial_chunk_size(device)
    while True:
        config.inference.chunk_size = chunk_size
        chunks = []
        step = chunk_size // config.inference.num_overlap
        border = chunk_size - step
        padded_length = len(model_input)
        if padded_length > 2 * border and border > 0:
            padded_length += 2 * border
        batches = math.ceil(math.ceil(padded_length / step) / config.inference.batch_size)

        def verify_chunk(module, inputs, prediction):
            if cancelled is not None and cancelled.is_set():
                raise KeyboardInterrupt
            if not torch.isfinite(prediction).all().item():
                raise ValueError("Non-finite raw prediction: 6stem")
            chunks.append(prediction.shape[-1])
            emit("model_progress", target="6stem", pass_number=1, chunk=len(chunks),
                 fraction=len(chunks) / batches, detail="Separating 6 stems", total_passes=total_passes,
                 chunk_size=chunk_size)

        hook = model.register_forward_hook(verify_chunk)
        emit("model_started", target="6stem", pass_number=1, detail="Separating 6 stems",
             total_passes=total_passes, chunk_size=chunk_size)
        model_started = time.monotonic()
        try:
            estimates = model_utils.demix(config, model, model_input.T.copy(), device, "bs_roformer")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.monotonic() - model_started
            break
        except (torch.OutOfMemoryError, RuntimeError) as error:
            is_oom = _is_out_of_memory(error)
            if is_oom and chunk_size > 55125:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                chunk_size //= 2
                emit("model_retry", target="6stem", pass_number=1, chunk_size=chunk_size,
                     detail=f"GPU memory limit reached; reducing chunk size to {chunk_size}")
                continue
            if is_oom:
                raise ValueError("Separation exceeded the safe GPU memory limit. Close GPU apps or choose CPU; the original song will be retained.") from error
            raise
        finally:
            hook.remove()

    emit("model_finished", target="6stem", pass_number=1, seconds=elapsed, detail="Separating 6 stems",
         total_passes=total_passes, chunk_size=chunk_size)
    if total_passes == 1:
        emit("finalizing", detail="Finalizing stems…")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    pcm = stage / "pcm"
    pcm.mkdir(exist_ok=True)
    for inst_name, out_file in STEM_MAPPING_6STEM.items():
        stem_audio = estimates[inst_name].T
        sf.write(file=work / f"{inst_name}-model.wav", data=stem_audio, samplerate=44100, subtype="FLOAT")
        path = work / f"{inst_name}.wav"
        convert(work / f"{inst_name}-model.wav", path, rate=rate)
        restored, restored_rate = sf.read(file=path, dtype="float32", always_2d=True)
        restored, _ = restore_timeline(restored, restored_rate, reference, rate, inst_name)
        sf.write(file=pcm / out_file, data=restored, samplerate=rate, subtype="FLOAT")

    combined = np.zeros(reference.shape, dtype=np.float64)
    for out_file in STEM_MAPPING_6STEM.values():
        audio, _ = sf.read(file=pcm / out_file, dtype="float64", always_2d=True)
        combined += audio
    sf.write(file=pcm / "song.wav", data=reference.astype(np.float64) - combined, samplerate=rate, subtype="FLOAT")
    return assets, elapsed, chunks


def find_drumsep_files(models):
    ckpt = models / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt"
    if not ckpt.is_file():
        ckpt = models / "drumsep.ckpt"
    yaml = models / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
    if not yaml.is_file():
        yaml = models / "drumsep.yaml"
    if not ckpt.is_file() or not yaml.is_file():
        raise ValueError(f"Missing DrumSep model files in {models}")
    return ckpt, yaml


def separate_drumsep_pcm(stage, models, reference, rate, settings, model_utils, device, emit, cancelled):
    ckpt_path, yaml_path = find_drumsep_files(models)
    assets = {
        ckpt_path.name: digest(ckpt_path),
        yaml_path.name: digest(yaml_path),
    }
    pcm = stage / "pcm"
    drums_path = pcm / "drums.wav"
    if not drums_path.is_file():
        raise ValueError(f"Missing separated drums stem for drum refinement: {drums_path}")
    work = stage / "pass-drumsep"
    work.mkdir(parents=True, exist_ok=True)
    convert(drums_path, work / "drum-model-input.wav", rate=44100)
    drum_input, _ = sf.read(work / "drum-model-input.wav", dtype="float32", always_2d=True)

    model, config = settings.get_model_from_config("mdx23c", str(yaml_path))
    config.training.use_amp = bool(config.training.use_amp and device.type == "cuda")
    if device.type == "cuda":
        budget_gpu(model=model, device=device)
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    try:
        model = model.to(device).eval()
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if _is_out_of_memory(error):
            raise ValueError("Model exceeded the safe GPU memory limit. Close GPU apps or choose CPU; the original song will be retained.") from error
        raise

    chunk_size = config.audio.chunk_size
    while True:
        config.inference.chunk_size = chunk_size
        step = chunk_size // config.inference.num_overlap
        border = chunk_size - step
        padded_length = len(drum_input)
        if padded_length > 2 * border and border > 0:
            padded_length += 2 * border
        batches = math.ceil(math.ceil(padded_length / step) / config.inference.batch_size)

        drum_chunks = []

        def verify_chunk(module, inputs, prediction):
            if cancelled is not None and cancelled.is_set():
                raise KeyboardInterrupt
            if not torch.isfinite(prediction).all().item():
                raise ValueError("Non-finite raw prediction: drumsep")
            drum_chunks.append(prediction.shape[-1])
            emit("model_progress", target="drumsep", pass_number=2, chunk=len(drum_chunks),
                 fraction=len(drum_chunks) / batches, detail="Separating drums", total_passes=2,
                 chunk_size=chunk_size)

        hook = model.register_forward_hook(verify_chunk)
        emit("model_started", target="drumsep", pass_number=2, detail="Separating drums",
             total_passes=2, chunk_size=chunk_size)
        started = time.monotonic()
        try:
            estimates = model_utils.demix(config, model, drum_input.T.copy(), device, "mdx23c")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.monotonic() - started
            break
        except (torch.OutOfMemoryError, RuntimeError) as error:
            is_oom = _is_out_of_memory(error)
            if is_oom and chunk_size > 55125:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                chunk_size //= 2
                emit("model_retry", target="drumsep", pass_number=2, chunk_size=chunk_size,
                     detail=f"GPU memory limit reached; reducing chunk size to {chunk_size}")
                continue
            if is_oom:
                raise ValueError("Separation exceeded the safe GPU memory limit. Close GPU apps or choose CPU; the original song will be retained.") from error
            raise
        finally:
            hook.remove()

    emit("model_finished", target="drumsep", pass_number=2, seconds=elapsed, detail="Separating drums",
         total_passes=2, chunk_size=chunk_size)
    emit("finalizing", detail="Finalizing stems…")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    for target_key, out_file in (("Kick", "drums_1.wav"), ("Snare", "drums_2.wav")):
        stem_audio = estimates[target_key].T
        raw_name = target_key.lower()
        sf.write(work / f"{raw_name}-model.wav", stem_audio, samplerate=44100, subtype="FLOAT")
        path = work / f"{raw_name}.wav"
        convert(work / f"{raw_name}-model.wav", path, rate=rate)
        restored, restored_rate = sf.read(path, dtype="float32", always_2d=True)
        restored, _ = restore_timeline(restored, restored_rate, reference, rate, raw_name)
        sf.write(pcm / out_file, restored, samplerate=rate, subtype="FLOAT")

    drums_full, _ = sf.read(drums_path, dtype="float64", always_2d=True)
    kick, _ = sf.read(pcm / "drums_1.wav", dtype="float64", always_2d=True)
    snare, _ = sf.read(pcm / "drums_2.wav", dtype="float64", always_2d=True)
    remaining_drums = drums_full - kick - snare
    sf.write(pcm / "drums_3.wav", remaining_drums, samplerate=rate, subtype="FLOAT")
    drums_path.unlink()

    run = {"target": "drumsep", "pass_number": 2, "seconds": elapsed,
           "chunks": len(drum_chunks), "raw_chunks_finite": True, "tail_rounding_frames": 0}
    return assets, run


def separate_8stem(stage, models, inference_source=None, device_name="auto", emit=None, model_profile="mega-8stem", cancelled=None):
    if emit is None:
        emit = lambda *args, **kwargs: None
    if inference_source is None:
        inference_source = DEFAULT_INFERENCE
    model_utils, settings, device = _load_inference_runtime(inference_source, device_name)
    emit("device", name=str(device))
    reference_path, reference, rate = _decode_reference(stage)
    stereo_bytes = len(reference) * 2 * 4
    model_bytes = math.ceil(len(reference) * 44100 / rate) * 2 * 4
    opus_bytes = math.ceil(len(reference) * OPUS_RATE / rate) * reference.shape[1] * 4
    required = 8 * stereo_bytes + 8 * model_bytes + reference.nbytes + 8 * opus_bytes + 64 * 1024**2
    if shutil.disk_usage(stage).free < required:
        raise ValueError(f"Insufficient working space: need approximately {required} more bytes")
    emit("decoded", sample_rate=rate, frames=len(reference), channels=reference.shape[1])

    started = time.monotonic()
    use_mlx = (
        sys.platform == "darwin"
        and (models / "bs_roformer_sw_config.json").is_file()
        and (models / "bs_roformer_sw.safetensors").is_file()
    )
    use_onnx = (
        (models / "bs_roformer_sw_6stem_fp16.onnx").is_file()
        or (models / "bs_roformer_sw_6stem_fp32.onnx").is_file()
    ) and (str(device_name).lower().startswith("dml") or model_profile == "onnx-8stem")
    if use_mlx:
        from .mlx_audio import run_mlx_6stem_pcm
        assets_1, elapsed_1, chunks_1 = run_mlx_6stem_pcm(stage=stage, models=models, reference=reference, rate=rate,
                                                          emit=emit, cancelled=cancelled, total_passes=2)
        n_chunks_1 = chunks_1
    elif use_onnx:
        from .onnx_audio import run_onnx_6stem_pcm
        assets_1, elapsed_1, chunks_1, _ = run_onnx_6stem_pcm(stage=stage, models=models, reference=reference, rate=rate,
                                                              emit=emit, cancelled=cancelled, total_passes=2, device=device_name)
        n_chunks_1 = chunks_1
    else:
        assets_1, elapsed_1, chunks_1 = run_torch_6stem_pcm(
            stage=stage, models=models, reference=reference, rate=rate,
            settings=settings, model_utils=model_utils, device=device,
            emit=emit, cancelled=cancelled, total_passes=2,
        )
        n_chunks_1 = len(chunks_1)
    emit("preparing_pass", pass_number=2, total_passes=2, detail="Preparing drum refinement…")
    assets_2, run_2 = separate_drumsep_pcm(
        stage=stage, models=models, reference=reference, rate=rate,
        settings=settings, model_utils=model_utils, device=device,
        emit=emit, cancelled=cancelled,
    )

    combined_assets = {**assets_1, **assets_2}
    run_1 = {
        "target": "6stem", "pass_number": 1, "seconds": elapsed_1,
        "chunks": n_chunks_1, "raw_chunks_finite": True, "tail_rounding_frames": 0,
    }
    runs = [run_1, run_2]

    pcm = stage / "pcm"
    pcm_validation = reconstruct(reference=reference, outputs=pcm, rate=rate, filenames=WAV_OUTPUTS)
    emit("validated", pass_number=2, codec="pcm_f32le", **pcm_validation)
    outputs = stage / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    validation = export_opus(reference_path=reference_path, pcm=pcm, outputs=outputs,
                             emit=emit, filenames=OUTPUTS)
    emit("validated", **validation)

    return {
        "sample_rate": rate, "frames": len(reference), "channels": reference.shape[1],
        "device": str(device), "model_type": "bs_roformer_and_mdx23c", "model_profile": model_profile,
        "model_assets_sha256": combined_assets, "inference_source": str(inference_source),
        "model_runs": runs, "pcm_validation": pcm_validation,
        "validation": validation,
        "seconds": time.monotonic() - started, "python": sys.version.split()[0],
        "torch": torch.__version__,
    }



def separate(stage, models, inference_source=None, device_name="auto", emit=None,
             model_profile="mega-8stem", cancelled=None):
    if model_profile == "mlx-6stem":
        from .mlx_audio import separate_mlx
        return separate_mlx(stage=stage, models=models, emit=emit, cancelled=cancelled)
    if model_profile == "torch-6stem":
        return separate_torch_6stem(stage=stage, models=models, inference_source=inference_source,
                                    device_name=device_name, emit=emit, cancelled=cancelled)
    if model_profile == "onnx-6stem":
        from .onnx_audio import separate_onnx
        return separate_onnx(stage=stage, models=models, emit=emit, cancelled=cancelled, device_name=device_name)
    if model_profile in ("mega-8stem", "8stem", "torch-8stem", "mlx-8stem", "onnx-8stem"):
        return separate_8stem(stage=stage, models=models, inference_source=inference_source,
                              device_name=device_name, emit=emit, model_profile=model_profile,
                              cancelled=cancelled)
    raise ValueError(f"Unknown model profile: {model_profile}")
