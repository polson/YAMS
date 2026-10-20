import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from .audio import (
    OPUS_RATE,
    OUTPUTS_6STEM,
    WAV_OUTPUTS_6STEM,
    _decode_reference,
    convert,
    export_opus,
    reconstruct,
    restore_timeline,
)
from .folders import digest


ONNX_STEM_ORDER = ("bass", "drums", "other", "vocals", "guitar", "piano")

ONNX_STEM_MAPPING = {
    "drums": "drums.wav",
    "bass": "bass.wav",
    "guitar": "guitar.wav",
    "piano": "keys.wav",
    "vocals": "vocals.wav",
}


def _is_out_of_memory(error: Exception) -> bool:
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


def _resolve_providers(device: str = "auto") -> list[str | tuple[str, dict]]:
    available = ort.get_available_providers()
    device_lower = str(device).lower()
    providers: list[str | tuple[str, dict]] = []

    if device_lower.startswith("dml") or (device_lower == "auto" and sys.platform == "win32"):
        if "DmlExecutionProvider" in available:
            device_id = 0
            if ":" in device_lower:
                try:
                    device_id = int(device_lower.split(":")[-1])
                except ValueError:
                    device_id = 0
            providers.append(("DmlExecutionProvider", {"device_id": device_id}))

    if device_lower.startswith("cuda") or (device_lower == "auto" and sys.platform != "win32"):
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        elif "ROCMExecutionProvider" in available:
            providers.append("ROCMExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def create_onnx_session(model_path: Path, device: str = "auto") -> tuple[ort.InferenceSession, str]:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = _resolve_providers(device=device)

    try:
        session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
        active_provider = session.get_providers()[0]
        return session, active_provider
    except Exception as error:
        if _is_out_of_memory(error) and "CPUExecutionProvider" in providers:
            session = ort.InferenceSession(
                str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
            )
            return session, "CPUExecutionProvider"
        raise


def run_onnx_6stem_pcm(
    stage: Path,
    models: Path,
    reference: np.ndarray,
    rate: int,
    emit=None,
    cancelled=None,
    total_passes: int = 1,
    device: str = "auto",
):
    if emit is None:
        emit = lambda *args, **kwargs: None

    model_path = models / "bs_roformer_sw_6stem_fp16.onnx"
    if not model_path.is_file():
        model_path = models / "bs_roformer_sw_6stem_fp32.onnx"
    if not model_path.is_file():
        raise ValueError(f"Missing ONNX model files in {models}")

    assets = {model_path.name: digest(model_path)}
    work = stage / "pass-1"
    work.mkdir(exist_ok=True)

    stereo = np.repeat(reference, 2, axis=1) if reference.shape[1] == 1 else reference
    sf.write(file=work / "stereo.wav", data=stereo, samplerate=rate, subtype="FLOAT")
    convert(work / "stereo.wav", work / "model-input.wav", rate=44100)
    model_input, _ = sf.read(file=work / "model-input.wav", dtype="float32", always_2d=True)

    session, active_provider = create_onnx_session(model_path=model_path, device=device)
    raw_audio = model_input.T
    num_samples = raw_audio.shape[1]

    chunk_len = 176400
    overlap_len = 88200
    hop_len = chunk_len - overlap_len
    pad = overlap_len

    emit(
        "model_started",
        target="6stem",
        pass_number=1,
        detail="Separating 6 stems (ONNX)",
        total_passes=total_passes,
        chunk_size=chunk_len,
    )
    model_started = time.monotonic()

    stft_window = torch.hann_window(2048)
    hann_window_np = np.hanning(chunk_len).astype(np.float32)

    padded = np.pad(raw_audio, [(0, 0), (pad, pad)], mode="reflect")
    total_padded = padded.shape[1]
    n_hops = int(math.ceil(max(total_padded - overlap_len, 1) / hop_len))
    total_len = (n_hops - 1) * hop_len + chunk_len
    if total_len > total_padded:
        padded = np.pad(padded, [(0, 0), (0, total_len - total_padded)])

    out_acc = np.zeros((6, 2, total_len), dtype=np.float32)
    w_acc = np.zeros(total_len, dtype=np.float32)

    active_device_tag = active_provider
    hop = 0
    while hop < n_hops:
        if cancelled is not None and cancelled.is_set():
            raise KeyboardInterrupt

        start = hop * hop_len
        chunk = padded[:, start : start + chunk_len]
        chunk_tensor = torch.from_numpy(chunk)

        spec = torch.stft(
            chunk_tensor,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=stft_window,
            center=True,
            return_complex=True,
        )
        spec_real = spec.real.unsqueeze(0).numpy()
        spec_imag = spec.imag.unsqueeze(0).numpy()

        try:
            outputs = session.run(
                ["out_spec_real", "out_spec_imag"],
                {"spec_real": spec_real, "spec_imag": spec_imag},
            )
        except Exception as error:
            if _is_out_of_memory(error) and active_provider != "CPUExecutionProvider":
                emit(
                    "model_retry",
                    target="6stem",
                    detail="GPU memory limit reached; falling back gracefully to CPU",
                    chunk_size=chunk_len,
                )
                del session
                gc.collect()
                session_options = ort.SessionOptions()
                session = ort.InferenceSession(
                    str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
                )
                active_provider = "CPUExecutionProvider"
                active_device_tag = "CPUExecutionProvider (OOM fallback)"
                continue
            raise

        out_real = torch.from_numpy(outputs[0][0])
        out_imag = torch.from_numpy(outputs[1][0])
        out_complex = torch.complex(out_real, out_imag)

        for stem_idx in range(6):
            stem_audio = torch.istft(
                out_complex[stem_idx],
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=stft_window,
                center=True,
                length=chunk_len,
            ).numpy()
            out_acc[stem_idx, :, start : start + chunk_len] += stem_audio * hann_window_np

        w_acc[start : start + chunk_len] += hann_window_np
        hop += 1

        emit(
            "model_progress",
            target="6stem",
            pass_number=1,
            chunk=hop,
            fraction=hop / n_hops,
            detail="Separating 6 stems (ONNX)",
            total_passes=total_passes,
            chunk_size=chunk_len,
        )

    out_acc = out_acc / np.maximum(w_acc, 1e-8)
    stems_np = out_acc[:, :, pad : pad + num_samples]

    elapsed = time.monotonic() - model_started
    emit(
        "model_finished",
        target="6stem",
        pass_number=1,
        seconds=elapsed,
        detail="Separating 6 stems (ONNX)",
        total_passes=total_passes,
        chunk_size=chunk_len,
    )
    if total_passes == 1:
        emit("finalizing", detail="Finalizing stems…")

    pcm = stage / "pcm"
    pcm.mkdir(exist_ok=True)

    for inst_index, inst_name in enumerate(ONNX_STEM_ORDER):
        if inst_name in ONNX_STEM_MAPPING:
            stem_audio = stems_np[inst_index].T
            sf.write(file=work / f"{inst_name}-model.wav", data=stem_audio, samplerate=44100, subtype="FLOAT")
            path = work / f"{inst_name}.wav"
            convert(work / f"{inst_name}-model.wav", path, rate=rate)
            restored, restored_rate = sf.read(file=path, dtype="float32", always_2d=True)
            restored, _ = restore_timeline(restored, restored_rate, reference, rate, inst_name)
            out_file = ONNX_STEM_MAPPING[inst_name]
            sf.write(file=pcm / out_file, data=restored, samplerate=rate, subtype="FLOAT")

    combined = np.zeros(reference.shape, dtype=np.float64)
    for out_name in ONNX_STEM_MAPPING.values():
        audio, _ = sf.read(file=pcm / out_name, dtype="float64", always_2d=True)
        combined += audio
    sf.write(file=pcm / "song.wav", data=reference.astype(np.float64) - combined, samplerate=rate, subtype="FLOAT")

    del session
    gc.collect()
    return assets, elapsed, n_hops, active_device_tag


def separate_onnx(stage: Path, models: Path, emit=None, cancelled=None, device_name: str = "auto"):
    if emit is None:
        emit = lambda *args, **kwargs: None

    reference_path, reference, rate = _decode_reference(stage)
    emit("decoded", sample_rate=rate, frames=len(reference), channels=reference.shape[1])
    stereo_bytes = len(reference) * 2 * 4
    model_bytes = math.ceil(len(reference) * 44100 / rate) * 2 * 4
    opus_bytes = math.ceil(len(reference) * 48000 / rate) * reference.shape[1] * 4
    required = 6 * stereo_bytes + 6 * model_bytes + reference.nbytes + 8 * opus_bytes + 64 * 1024**2
    if shutil.disk_usage(stage).free < required:
        raise ValueError(f"Insufficient working space: need approximately {required} more bytes")

    started = time.monotonic()
    assets, elapsed, n_hops, active_device = run_onnx_6stem_pcm(
        stage=stage,
        models=models,
        reference=reference,
        rate=rate,
        emit=emit,
        cancelled=cancelled,
        total_passes=1,
        device=device_name,
    )
    emit("device", name=active_device)

    pcm = stage / "pcm"
    pcm_validation = reconstruct(reference=reference, outputs=pcm, rate=rate, filenames=WAV_OUTPUTS_6STEM)
    emit("validated", pass_number=1, codec="pcm_f32le", **pcm_validation)

    outputs = stage / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    validation = export_opus(
        reference_path=reference_path, pcm=pcm, outputs=outputs, emit=emit, filenames=OUTPUTS_6STEM
    )
    emit("validated", **validation)

    runs = [
        {
            "target": "6stem",
            "pass_number": 1,
            "seconds": elapsed,
            "chunks": n_hops,
            "raw_chunks_finite": True,
            "tail_rounding_frames": 0,
        }
    ]
    return {
        "sample_rate": rate,
        "frames": len(reference),
        "channels": reference.shape[1],
        "device": active_device,
        "model_type": "bs_roformer_onnx",
        "model_profile": "onnx-6stem",
        "model_assets_sha256": assets,
        "inference_source": "onnxruntime",
        "model_runs": runs,
        "pcm_validation": pcm_validation,
        "validation": validation,
        "seconds": time.monotonic() - started,
        "python": sys.version.split()[0],
        "onnxruntime": ort.__version__,
    }
