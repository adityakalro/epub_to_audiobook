import logging
import wave
import numpy as np

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

CHATTERBOX_MODEL_ID = "mlx-community/chatterbox-turbo-fp16"
CHATTERBOX_SAMPLE_RATE = 24000
CHATTERBOX_MAX_CHARS = 500

_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        from mlx_audio.tts.utils import load
        logger.info(f"Loading Chatterbox model: {CHATTERBOX_MODEL_ID}")
        _model_cache = load(CHATTERBOX_MODEL_ID)
        logger.info("Chatterbox model loaded successfully")
    return _model_cache


class ChatterboxTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        super().__init__(config)

    def validate_config(self):
        pass

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        text_chunks = split_text(text, CHATTERBOX_MAX_CHARS, self.config.language)
        model = _get_model()
        all_audio = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(f"Processing {chunk_id}, length={len(chunk)}")
            logger.debug(f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]")

            kwargs = {"text": chunk}
            if self.config.chatterbox_ref_audio:
                kwargs["ref_audio"] = self.config.chatterbox_ref_audio

            for result in model.generate(**kwargs):
                all_audio.append(np.array(result.audio))

        if not all_audio:
            logger.warning(f"No audio generated for {output_file}")
            return

        audio = np.concatenate(all_audio)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(CHATTERBOX_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        logger.debug(f"Skipping ID3 tags for WAV output: {output_file}")

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return "wav"

    def estimate_cost(self, total_chars):
        return 0.0
