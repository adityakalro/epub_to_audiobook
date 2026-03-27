import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.tts_providers.voxtral_tts_provider import (
    VoxtralTTSProvider,
    get_voxtral_supported_voices,
)
from tests.test_utils import get_voxtral_config


class TestVoxtralTtsProvider(unittest.TestCase):

    def test_default_voice(self):
        config = get_voxtral_config()
        config.voice_name = None
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, VoxtralTTSProvider)
        self.assertEqual(tts_provider.config.voice_name, "neutral_male")

    def test_invalid_voice_raises(self):
        config = get_voxtral_config()
        config.voice_name = "robot_voice_9000"
        with self.assertRaises(ValueError):
            get_tts_provider(config)

    def test_supported_voices_contains_expected(self):
        voices = get_voxtral_supported_voices()
        for expected in ["casual_male", "casual_female", "neutral_male", "neutral_female",
                         "cheerful_female", "fr_male", "fr_female", "hi_male"]:
            self.assertIn(expected, voices)

    def test_estimate_cost_is_zero(self):
        config = get_voxtral_config()
        tts_provider = get_tts_provider(config)
        self.assertEqual(tts_provider.estimate_cost(0), 0.0)
        self.assertEqual(tts_provider.estimate_cost(1_000_000), 0.0)

    def test_output_format_is_wav(self):
        config = get_voxtral_config()
        tts_provider = get_tts_provider(config)
        self.assertEqual(tts_provider.get_output_file_extension(), "wav")

    def test_break_string_is_string(self):
        config = get_voxtral_config()
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider.get_break_string(), str)

    @patch("audiobook_generator.tts_providers.voxtral_tts_provider._get_model")
    def test_text_to_speech_writes_wav(self, mock_get_model):
        import numpy as np

        mock_result = MagicMock()
        mock_result.audio = np.zeros(24000, dtype=np.float32)  # 1s of silence at 24kHz
        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result]
        mock_get_model.return_value = mock_model

        config = get_voxtral_config()
        tts_provider = get_tts_provider(config)

        audio_tags = MagicMock()
        audio_tags.idx = 1
        audio_tags.title = "Test Chapter"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tts_provider.text_to_speech("Hello world.", tmp_path, audio_tags)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            os.unlink(tmp_path)

    @patch("audiobook_generator.tts_providers.voxtral_tts_provider._get_model")
    def test_text_to_speech_multiple_chunks(self, mock_get_model):
        import numpy as np

        mock_result = MagicMock()
        mock_result.audio = np.zeros(2400, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_result]
        mock_get_model.return_value = mock_model

        config = get_voxtral_config()
        tts_provider = get_tts_provider(config)

        audio_tags = MagicMock()
        audio_tags.idx = 2
        audio_tags.title = "Long Chapter"

        # Build text longer than VOXTRAL_MAX_CHARS to force multiple chunks
        long_text = ("This is a sentence. " * 30).strip()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tts_provider.text_to_speech(long_text, tmp_path, audio_tags)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
            # generate should have been called more than once (multiple chunks)
            self.assertGreater(mock_model.generate.call_count, 1)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
