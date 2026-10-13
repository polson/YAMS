import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from yams.audio import convert, export_opus, reconstruct, restore_timeline
from yams.folders import OUTPUTS, WAV_OUTPUTS, digest


class AudioValidationTests(unittest.TestCase):
    def fixture(self, channels, rate):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        reference = np.random.default_rng(5).uniform(-0.9, 0.9, (2200, channels)).astype(np.float32)
        total = np.zeros(reference.shape, dtype=np.float64)
        for index, name in enumerate(WAV_OUTPUTS[:-1]):
            audio = reference.astype(np.float64) * (0.05 + index * 0.03)
            if index == 0:
                audio *= 40
            sf.write(file=folder / name, data=audio, samplerate=rate, subtype="FLOAT")
            saved, _ = sf.read(file=folder / name, dtype="float64", always_2d=True)
            total += saved
        sf.write(file=folder / "song.wav", data=reference.astype(np.float64) - total, samplerate=rate, subtype="FLOAT")
        return folder, reference

    def test_preserves_headroom_and_reconstructs_mono_and_stereo(self):
        for channels in (1, 2):
            for rate in (44100, 48000):
                with self.subTest(channels=channels, rate=rate):
                    folder, reference = self.fixture(channels, rate)
                    result = reconstruct(reference, folder, rate)
                    self.assertLessEqual(result["peak_error"], 1e-6)
                    self.assertGreater(result["files"]["song.wav"]["peak"], 1)
                    for name in WAV_OUTPUTS:
                        self.assertEqual(result["files"][name]["sha256"], digest(folder / name))

    def test_restores_resampled_timeline_with_bounded_crop_and_padding(self):
        reference = np.zeros((4, 1), dtype=np.float32)
        audio = np.arange(12, dtype=np.float32).reshape(6, 2)
        cropped, tail = restore_timeline(audio, 48000, reference, 48000, "stem")
        self.assertEqual(tail, 2)
        np.testing.assert_array_equal(cropped, audio[:4].mean(axis=1, keepdims=True))

        padded, tail = restore_timeline(audio[:2], 48000, reference, 48000, "stem")
        self.assertEqual(tail, -2)
        np.testing.assert_array_equal(padded, np.pad(audio[:2].mean(axis=1, keepdims=True), ((0, 2), (0, 0))))

        with self.assertRaisesRegex(ValueError, "Unexpected resampling timeline: stem"):
            restore_timeline(np.zeros((7, 2)), 48000, reference, 48000, "stem")

    def test_rejects_wrong_timing_rate_channels_and_nonfinite_samples(self):
        for invalid in ("length", "rate", "channels", "nan", "clipping", "integer"):
            with self.subTest(invalid=invalid):
                folder, reference = self.fixture(2, 48000)
                audio, rate = sf.read(file=folder / "song.wav", dtype="float64", always_2d=True)
                subtype = "FLOAT"
                if invalid == "length":
                    audio = audio[:-1]
                elif invalid == "rate":
                    rate = 44100
                elif invalid == "channels":
                    audio = audio[:, :1]
                elif invalid == "nan":
                    audio[0, 0] = np.nan
                elif invalid == "clipping":
                    audio = np.clip(audio, -1, 1)
                else:
                    subtype = "PCM_16"
                sf.write(file=folder / "song.wav", data=audio, samplerate=rate, subtype=subtype)
                with self.assertRaises(ValueError):
                    reconstruct(reference, folder, 48000)


class OpusExportTests(unittest.TestCase):
    def fixture(self, channels, rate, frames):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        stage = Path(temporary.name)
        pcm = stage / "pcm"
        outputs = stage / "outputs"
        pcm.mkdir()
        outputs.mkdir()
        timeline = np.arange(frames) / rate
        reference = np.repeat((0.5 * np.sin(2 * np.pi * 440 * timeline))[:, None], channels, axis=1)
        reference[:min(96, frames)] = 0
        reference[-min(96, frames):] = 0
        sf.write(file=stage / "reference.wav", data=reference, samplerate=rate, subtype="FLOAT")
        for name in WAV_OUTPUTS:
            sf.write(file=pcm / name, data=reference / len(OUTPUTS), samplerate=rate, subtype="FLOAT")
        return stage, pcm, outputs

    def test_actual_opus_codec_geometry_hashes_and_lossy_error(self):
        for channels in (1, 2):
            for rate, frames in ((44100, 2200), (48000, 1003), (32000, 3200)):
                with self.subTest(channels=channels, rate=rate, frames=frames):
                    stage, pcm, outputs = self.fixture(channels, rate, frames)
                    events = []
                    result = export_opus(reference_path=stage / "reference.wav", pcm=pcm, outputs=outputs,
                                         emit=lambda event, **fields: events.append((event, fields)))
                    self.assertEqual(result["sample_rate"], 48000)
                    self.assertEqual(result["channels"], channels)
                    self.assertEqual(result["bitrate"], 192000)
                    self.assertTrue(result["lossy"])
                    self.assertGreater(result["rms_error"], 0)
                    self.assertLess(result["rms_error"], 0.025)
                    self.assertEqual(set(path.name for path in outputs.iterdir()), set(OUTPUTS))
                    self.assertEqual([fields["asset"] for event, fields in events], list(OUTPUTS))
                    for name in OUTPUTS:
                        info = sf.info(outputs / name)
                        self.assertEqual((info.format, info.subtype), ("OGG", "OPUS"))
                        self.assertEqual((info.samplerate, info.channels, info.frames), (48000, channels, result["frames"]))
                        self.assertEqual(result["files"][name]["sha256"], digest(outputs / name))
                    self.assertFalse(list(outputs.glob("*.wav")))

    def test_float_headroom_is_not_clipped_or_normalized(self):
        stage, pcm, outputs = self.fixture(2, 48000, 4800)
        audio, rate = sf.read(file=pcm / "guitar.wav", always_2d=True)
        sf.write(file=pcm / "guitar.wav", data=audio * 40, samplerate=rate, subtype="FLOAT")
        result = export_opus(reference_path=stage / "reference.wav", pcm=pcm, outputs=outputs,
                             emit=lambda *args, **kwargs: None)
        self.assertGreater(result["files"]["guitar.opus"]["peak"], 2)
        self.assertGreater(result["files"]["guitar.opus"]["samples_above_full_scale"], 0)

    def test_decoded_geometry_and_nonfinite_samples_are_rejected(self):
        for invalid in ("length", "channels", "rate", "nan"):
            with self.subTest(invalid=invalid):
                stage, pcm, outputs = self.fixture(2, 48000, 1003)

                def invalid_decode(source, destination, rate):
                    convert(source=source, destination=destination, rate=rate)
                    if source.suffix == ".opus":
                        audio, decoded_rate = sf.read(file=destination, always_2d=True)
                        if invalid == "length":
                            audio = audio[:-1]
                        elif invalid == "channels":
                            audio = audio[:, :1]
                        elif invalid == "rate":
                            decoded_rate = 44100
                        else:
                            audio[0, 0] = np.nan
                        sf.write(file=destination, data=audio, samplerate=decoded_rate, subtype="FLOAT")

                with patch("yams.audio.convert", side_effect=invalid_decode), self.assertRaises(ValueError):
                    export_opus(reference_path=stage / "reference.wav", pcm=pcm, outputs=outputs,
                                emit=lambda *args, **kwargs: None)

    def test_cancellation_between_stems_stops_encoding(self):
        stage, pcm, outputs = self.fixture(1, 44100, 2200)

        def stop(event, **fields):
            if fields["asset"] == OUTPUTS[1]:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            export_opus(reference_path=stage / "reference.wav", pcm=pcm, outputs=outputs, emit=stop)
        self.assertEqual([path.name for path in outputs.iterdir()], [OUTPUTS[0]])


class Torch6StemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.stage = Path(self.temporary.name) / "stage"
        self.stage.mkdir()
        self.models = Path(self.temporary.name) / "models"
        self.models.mkdir()
        (self.models / "BS-Rofo-SW-Fixed.yaml").write_text("dummy")
        (self.models / "BS-Rofo-SW-Fixed.ckpt").write_bytes(b"dummy")
        self.reference_audio = np.random.default_rng(42).uniform(-0.5, 0.5, (4800, 2)).astype(np.float32)
        sf.write(file=self.stage / "original", data=self.reference_audio, samplerate=48000, format="WAV", subtype="FLOAT")

    def test_separate_torch_6stem_reconstructs_and_exports_all_stems(self):
        from yams.audio import DEFAULT_INFERENCE, separate_torch_6stem
        from yams.folders import OUTPUTS_6STEM, WAV_OUTPUTS_6STEM

        if str(DEFAULT_INFERENCE) not in sys.path:
            sys.path.insert(0, str(DEFAULT_INFERENCE))

        mock_config = MagicMock()
        mock_config.model.num_stems = 6
        mock_config.audio.sample_rate = 44100
        mock_config.inference.normalize = False
        mock_config.training.use_amp = False
        mock_config.inference.chunk_size = 44100
        mock_config.inference.num_overlap = 2
        mock_config.inference.batch_size = 1

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        def fake_demix(config, model, mix, device, model_type):
            stems = {}
            for inst in ("drums", "vocals", "guitar", "bass", "piano", "other"):
                stems[inst] = (mix * 0.1).astype(np.float32)
            return stems

        with patch("torch.load", return_value={}), \
             patch("utils.settings.get_model_from_config", return_value=(mock_model, mock_config)), \
             patch("utils.model_utils.demix", side_effect=fake_demix):
            result = separate_torch_6stem(stage=self.stage, models=self.models, device_name="cpu")

        self.assertEqual(result["model_profile"], "torch-6stem")
        self.assertEqual(result["model_type"], "bs_roformer_torch")
        self.assertLessEqual(result["pcm_validation"]["peak_error"], 1e-6)
        pcm_files = [p.name for p in (self.stage / "pcm").iterdir()]
        self.assertEqual(set(pcm_files), set(WAV_OUTPUTS_6STEM))
        opus_files = [p.name for p in (self.stage / "outputs").iterdir()]
        self.assertEqual(set(opus_files), set(OUTPUTS_6STEM))


class Torch8StemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.stage = Path(self.temporary.name) / "stage"
        self.stage.mkdir()
        self.models = Path(self.temporary.name) / "models"
        self.models.mkdir()
        (self.models / "BS-Rofo-SW-Fixed.yaml").write_text("dummy")
        (self.models / "BS-Rofo-SW-Fixed.ckpt").write_bytes(b"dummy")
        (self.models / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml").write_text("dummy")
        (self.models / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt").write_bytes(b"dummy")
        self.reference_audio = np.random.default_rng(42).uniform(-0.5, 0.5, (4800, 2)).astype(np.float32)
        sf.write(file=self.stage / "original", data=self.reference_audio, samplerate=48000, format="WAV", subtype="FLOAT")

    def test_separate_8stem_reconstructs_and_exports_all_stems(self):
        from yams.audio import DEFAULT_INFERENCE, separate_8stem
        from yams.folders import OUTPUTS, WAV_OUTPUTS

        if str(DEFAULT_INFERENCE) not in sys.path:
            sys.path.insert(0, str(DEFAULT_INFERENCE))

        mock_config_6stem = MagicMock()
        mock_config_6stem.model.num_stems = 6
        mock_config_6stem.audio.sample_rate = 44100
        mock_config_6stem.inference.normalize = False
        mock_config_6stem.training.use_amp = False
        mock_config_6stem.inference.chunk_size = 44100
        mock_config_6stem.inference.num_overlap = 2
        mock_config_6stem.inference.batch_size = 1

        mock_config_drum = MagicMock()
        mock_config_drum.audio.chunk_size = 44100
        mock_config_drum.inference.num_overlap = 2
        mock_config_drum.inference.batch_size = 1
        mock_config_drum.training.use_amp = False

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        def fake_get_model(model_type, config_path):
            if model_type == "bs_roformer":
                return mock_model, mock_config_6stem
            return mock_model, mock_config_drum

        def fake_demix(config, model, mix, device, model_type):
            if model_type == "bs_roformer":
                return {inst: (mix * 0.1).astype(np.float32) for inst in ("drums", "vocals", "guitar", "bass", "piano", "other")}
            return {inst: (mix * 0.2).astype(np.float32) for inst in ("Kick", "Snare", "Toms", "Hh", "Ride", "Crash")}

        events = []

        with patch("torch.load", return_value={}), \
             patch("utils.settings.get_model_from_config", side_effect=fake_get_model), \
             patch("utils.model_utils.demix", side_effect=fake_demix):
            result = separate_8stem(
                stage=self.stage, models=self.models, device_name="cpu",
                emit=lambda event, **fields: events.append((event, fields)),
            )

        self.assertEqual(result["model_profile"], "mega-8stem")
        self.assertEqual(result["model_type"], "bs_roformer_and_mdx23c")
        self.assertEqual(len(result["model_runs"]), 2)
        self.assertEqual(result["model_runs"][0]["target"], "6stem")
        self.assertEqual(result["model_runs"][1]["target"], "drumsep")
        self.assertLessEqual(result["pcm_validation"]["peak_error"], 1e-6)
        pcm_files = [p.name for p in (self.stage / "pcm").iterdir()]
        self.assertEqual(set(pcm_files), set(WAV_OUTPUTS))
        opus_files = [p.name for p in (self.stage / "outputs").iterdir()]
        self.assertEqual(set(opus_files), set(OUTPUTS))

        drum_start_events = [fields for ev, fields in events if ev == "model_started" and fields.get("target") == "drumsep"]
        self.assertEqual(len(drum_start_events), 1)
        self.assertEqual(drum_start_events[0]["pass_number"], 2)
        self.assertEqual(drum_start_events[0]["total_passes"], 2)


if __name__ == "__main__":
    unittest.main()
