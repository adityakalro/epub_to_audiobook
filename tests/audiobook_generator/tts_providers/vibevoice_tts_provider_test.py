import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.tts_providers.vibevoice_tts_provider import (
    VibeVoiceTTSProvider,
    VIBEVOICE_DEFAULT_CFG_PACE,
)
from tests.test_utils import get_vibevoice_config


class TestVibeVoiceTtsProvider(unittest.TestCase):

    def test_missing_ref_audio_raises(self):
        config = get_vibevoice_config(ref_audio=None)
        with self.assertRaises(ValueError):
            get_tts_provider(config)

    def test_nonexistent_ref_audio_raises(self):
        config = get_vibevoice_config(ref_audio="/nonexistent/path/voice.wav")
        with self.assertRaises(ValueError):
            get_tts_provider(config)

    def test_default_cfg_pace_applied(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            config = get_vibevoice_config(ref_audio=tmp_path)
            config.vibevoice_cfg_pace = None
            provider = get_tts_provider(config)
            self.assertEqual(provider.config.vibevoice_cfg_pace, VIBEVOICE_DEFAULT_CFG_PACE)
        finally:
            os.unlink(tmp_path)

    def test_estimate_cost_is_zero(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            config = get_vibevoice_config(ref_audio=tmp_path)
            provider = get_tts_provider(config)
            self.assertEqual(provider.estimate_cost(0), 0.0)
            self.assertEqual(provider.estimate_cost(1_000_000), 0.0)
        finally:
            os.unlink(tmp_path)

    def test_output_format_is_wav(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            config = get_vibevoice_config(ref_audio=tmp_path)
            provider = get_tts_provider(config)
            self.assertEqual(provider.get_output_file_extension(), "wav")
        finally:
            os.unlink(tmp_path)

    def test_break_string_is_string(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            config = get_vibevoice_config(ref_audio=tmp_path)
            provider = get_tts_provider(config)
            self.assertIsInstance(provider.get_break_string(), str)
        finally:
            os.unlink(tmp_path)

    @patch("audiobook_generator.tts_providers.vibevoice_tts_provider._get_model_and_processor")
    def test_text_to_speech_writes_wav(self, mock_get_model):
        import numpy as np

        sample_rate = 24000
        mock_processor = MagicMock()
        mock_processor.audio_processor.sampling_rate = sample_rate

        mock_speech = MagicMock()
        mock_speech.cpu.return_value.float.return_value.numpy.return_value = np.zeros(
            sample_rate, dtype=np.float32
        )

        mock_output = MagicMock()
        mock_output.speech_outputs = [mock_speech]

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_model.generate.return_value = mock_output

        mock_processor.return_value = {
            "input_ids": MagicMock(spec=["to"]),
        }
        mock_processor.__call__ = lambda *a, **kw: {}

        mock_get_model.return_value = (mock_model, mock_processor)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref:
            ref_path = ref.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            out_path = out.name

        try:
            config = get_vibevoice_config(ref_audio=ref_path)
            provider = get_tts_provider(config)

            audio_tags = MagicMock()
            audio_tags.idx = 1
            audio_tags.title = "Test Chapter"

            provider.text_to_speech("Hello world.", out_path, audio_tags)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
        finally:
            os.unlink(ref_path)
            os.unlink(out_path)

    @patch("audiobook_generator.tts_providers.vibevoice_tts_provider._get_model_and_processor")
    def test_text_to_speech_multiple_chunks(self, mock_get_model):
        import numpy as np

        sample_rate = 24000
        mock_processor = MagicMock()
        mock_processor.audio_processor.sampling_rate = sample_rate

        mock_speech = MagicMock()
        mock_speech.cpu.return_value.float.return_value.numpy.return_value = np.zeros(
            2400, dtype=np.float32
        )

        mock_output = MagicMock()
        mock_output.speech_outputs = [mock_speech]

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_model.generate.return_value = mock_output

        mock_get_model.return_value = (mock_model, mock_processor)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref:
            ref_path = ref.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            out_path = out.name

        try:
            config = get_vibevoice_config(ref_audio=ref_path)
            provider = get_tts_provider(config)

            audio_tags = MagicMock()
            audio_tags.idx = 2
            audio_tags.title = "Long Chapter"

            long_text = ("This is a sentence. " * 30).strip()
            provider.text_to_speech(long_text, out_path, audio_tags)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
            self.assertGreater(mock_model.generate.call_count, 1)
        finally:
            os.unlink(ref_path)
            os.unlink(out_path)


if __name__ == "__main__":
    unittest.main()
