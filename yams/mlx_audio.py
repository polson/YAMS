import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx.core as mx
from mlx_audio_separator.separator.models.roformer.loader import (
    create_bs_roformer_mlx,
    convert_torch_to_mlx_weights,
    _override_depth_from_weights,
)

from .audio import _decode_reference, convert, reconstruct, export_opus, restore_timeline
from .folders import OUTPUTS_6STEM, WAV_OUTPUTS_6STEM, digest


MLX_INSTRUMENTS = ("bass", "drums", "other", "vocals", "guitar", "piano")
MLX_STEM_MAPPING = {
    "drums": "drums.wav",
    "vocals": "vocals.wav",
    "guitar": "guitar.wav",
    "bass": "bass.wav",
    "piano": "keys.wav",
}


def load_mlx_model(weights_path, config):
    raw_weights = mx.load(str(weights_path))
    if any(k.startswith("layers.") or "mask_estimators." in k for k in raw_weights.keys()):
        weights = convert_torch_to_mlx_weights(raw_weights)
    else:
        weights = raw_weights
    _override_depth_from_weights(config, set(weights.keys()))
    model = create_bs_roformer_mlx(config)
    model.load_weights(list(weights.items()), strict=True)
    return model


def run_mlx_6stem_pcm(stage, models, reference, rate, emit=None, cancelled=None, total_passes=1):
    if emit is None:
        emit = lambda *args, **kwargs: None
    config_path = models / "bs_roformer_sw_config.json"
    weights_path = models / "bs_roformer_sw.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"Missing MLX model files in {models}")
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

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    model = load_mlx_model(weights_path, config)

    raw_audio = mx.array(model_input.T, dtype=mx.float32)
    chunk_len = int(round(model.chunk_seconds * 44100))
    emit("model_started", target="6stem", pass_number=1, detail="Separating 6 stems",
         total_passes=total_passes, chunk_size=chunk_len)
    model_started = time.monotonic()

    overlap_len = int(round(model.overlap_seconds * 44100))
    hop_len = chunk_len - overlap_len
    num_samples = raw_audio.shape[1]
    n_hops = int(math.ceil(max(num_samples - overlap_len, 1) / hop_len))
    total_len = (n_hops - 1) * hop_len + chunk_len
    pad_len = total_len - num_samples
    padded = mx.pad(raw_audio, [(0, 0), (0, pad_len)]) if pad_len > 0 else raw_audio

    w_hann = mx.array(np.hanning(chunk_len).astype(np.float32), dtype=mx.float32)
    w_view = w_hann.reshape(1, 1, 1, -1)

    out_acc = mx.zeros((1, 6, 2, total_len), dtype=mx.float32)
    w_acc = mx.zeros((1, 1, 1, total_len), dtype=mx.float32)

    for hop in range(n_hops):
        if cancelled is not None and cancelled.is_set():
            raise KeyboardInterrupt
        start = hop * hop_len
        chunk = padded[:, start:start + chunk_len]
        chunk_batch = chunk[None, ...]
        batch_out = model(chunk_batch)
        out_update = batch_out * w_view
        out_acc = out_acc.at[..., start:start + chunk_len].add(out_update)
        w_acc = w_acc.at[..., start:start + chunk_len].add(w_view)
        mx.eval(out_acc, w_acc)
        emit("model_progress", target="6stem", pass_number=1, chunk=hop + 1, fraction=(hop + 1) / n_hops,
             detail="Separating 6 stems", total_passes=total_passes, chunk_size=chunk_len)

    out_acc = out_acc / mx.maximum(w_acc, 1e-8)
    out_acc = out_acc[..., :num_samples]
    mx.eval(out_acc)
    stems_np = np.array(out_acc[0])
    elapsed = time.monotonic() - model_started
    emit("model_finished", target="6stem", pass_number=1, seconds=elapsed, detail="Separating 6 stems",
         total_passes=total_passes, chunk_size=chunk_len)
    if total_passes == 1:
        emit("finalizing", detail="Finalizing stems…")

    pcm = stage / "pcm"
    pcm.mkdir(exist_ok=True)
    for inst_index, inst_name in enumerate(config["instruments"]):
        if inst_name in MLX_STEM_MAPPING:
            stem_audio = stems_np[inst_index].T
            sf.write(file=work / f"{inst_name}-model.wav", data=stem_audio, samplerate=44100, subtype="FLOAT")
            path = work / f"{inst_name}.wav"
            convert(work / f"{inst_name}-model.wav", path, rate=rate)
            restored, restored_rate = sf.read(file=path, dtype="float32", always_2d=True)
            restored, _ = restore_timeline(restored, restored_rate, reference, rate, inst_name)
            out_file = MLX_STEM_MAPPING[inst_name]
            sf.write(file=pcm / out_file, data=restored, samplerate=rate, subtype="FLOAT")

    combined = np.zeros(reference.shape, dtype=np.float64)
    for out_name in MLX_STEM_MAPPING.values():
        audio, _ = sf.read(file=pcm / out_name, dtype="float64", always_2d=True)
        combined += audio
    sf.write(file=pcm / "song.wav", data=reference.astype(np.float64) - combined, samplerate=rate, subtype="FLOAT")
    return assets, elapsed, n_hops


def separate_mlx(stage, models, emit=None, cancelled=None):
    if emit is None:
        emit = lambda *args, **kwargs: None
    emit("device", name="mlx:gpu")
    reference_path, reference, rate = _decode_reference(stage)
    emit("decoded", sample_rate=rate, frames=len(reference), channels=reference.shape[1])
    stereo_bytes = len(reference) * 2 * 4
    model_bytes = math.ceil(len(reference) * 44100 / rate) * 2 * 4
    opus_bytes = math.ceil(len(reference) * 48000 / rate) * reference.shape[1] * 4
    required = 6 * stereo_bytes + 6 * model_bytes + reference.nbytes + 8 * opus_bytes + 64 * 1024**2
    if shutil.disk_usage(stage).free < required:
        raise ValueError(f"Insufficient working space: need approximately {required} more bytes")

    started = time.monotonic()
    assets, elapsed, n_hops = run_mlx_6stem_pcm(stage=stage, models=models, reference=reference, rate=rate,
                                                emit=emit, cancelled=cancelled, total_passes=1)
    pcm = stage / "pcm"
    pcm_validation = reconstruct(reference=reference, outputs=pcm, rate=rate, filenames=WAV_OUTPUTS_6STEM)
    emit("validated", pass_number=1, codec="pcm_f32le", **pcm_validation)
    outputs = stage / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    validation = export_opus(reference_path=reference_path, pcm=pcm, outputs=outputs,
                             emit=emit, filenames=OUTPUTS_6STEM)
    emit("validated", **validation)

    runs = [{"target": "6stem", "pass_number": 1, "seconds": elapsed,
             "chunks": n_hops, "raw_chunks_finite": True, "tail_rounding_frames": 0}]
    return {
        "sample_rate": rate, "frames": len(reference), "channels": reference.shape[1],
        "device": "mlx:gpu", "model_type": "bs_roformer_mlx", "model_profile": "mlx-6stem",
        "model_assets_sha256": assets, "inference_source": "mlx_audio_separator",
        "model_runs": runs, "pcm_validation": pcm_validation,
        "validation": validation,
        "seconds": time.monotonic() - started, "python": sys.version.split()[0],
        "mlx": mx.__version__,
    }
