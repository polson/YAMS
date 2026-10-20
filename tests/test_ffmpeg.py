import gzip
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import yams.ffmpeg as ffmpeg


class FfmpegTests(unittest.TestCase):
    def test_current_asset_key_returns_supported_tuple(self):
        key = ffmpeg._current_asset_key()
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 2)
        self.assertIn(key[0], ("linux", "darwin", "win32"))
        self.assertIn(key[1], ("x86_64", "arm64"))

    def test_find_ffmpeg_prefers_cache_then_resources_then_system(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cache_bin = temp / "cache" / "bin"
            resources_bin = temp / "resources" / "bin"
            cache_bin.mkdir(parents=True)
            resources_bin.mkdir(parents=True)

            bin_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
            cache_file = cache_bin / bin_name
            resources_file = resources_bin / bin_name

            cache_file.write_text("fake")
            resources_file.write_text("fake")

            with patch("yams.ffmpeg.test_ffmpeg_binary", return_value=True):
                found = ffmpeg.find_ffmpeg(cache_bin=cache_bin, resources_bin=resources_bin)
                self.assertEqual(found, cache_file)

            with patch("yams.ffmpeg.test_ffmpeg_binary", side_effect=[False, True]):
                found = ffmpeg.find_ffmpeg(cache_bin=cache_bin, resources_bin=resources_bin)
                self.assertEqual(found, resources_file)

    def test_ensure_ffmpeg_returns_existing_without_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cache_dir = temp / "cache"
            cache_bin = cache_dir / "bin"
            cache_bin.mkdir(parents=True)

            bin_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
            target = cache_bin / bin_name
            target.write_text("fake")

            with patch("yams.ffmpeg.test_ffmpeg_binary", return_value=True), \
                 patch("urllib.request.urlopen") as mock_url:
                result = ffmpeg.ensure_ffmpeg(cache_dir=cache_dir)
                self.assertEqual(result, target)
                mock_url.assert_not_called()

    def test_ensure_ffmpeg_aborts_on_corrupt_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cache_dir = temp / "cache"

            mock_response = MagicMock()
            mock_response.read.side_effect = [b"corrupt data", b""]
            mock_response.__enter__.return_value = mock_response

            with patch("yams.ffmpeg.find_ffmpeg", return_value=None), \
                 patch("urllib.request.urlopen", return_value=mock_response), \
                 patch.dict("yams.ffmpeg.FFMPEG_ASSETS", {
                     ffmpeg._current_asset_key(): {
                         "url": "https://example.com/ffmpeg.gz",
                         "bytes": 100,
                         "sha256": "expected_hash",
                         "binary": "ffmpeg",
                     }
                 }):
                with self.assertRaises(ValueError):
                    ffmpeg.ensure_ffmpeg(cache_dir=cache_dir)


if __name__ == "__main__":
    unittest.main()
