import io
import logging
import wave

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

VOXTRAL_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
VOXTRAL_SAMPLE_RATE = 24000
VOXTRAL_MAX_CHARS = 500

_model_cache = None


def get_voxtral_supported_voices():
    return [
        "casual_female", "casual_male", "cheerful_female", "neutral_female", "neutral_male",
        "fr_male", "fr_female",
        "es_male", "es_female",
        "de_male", "de_female",
        "it_male", "it_female",
        "pt_male", "pt_female",
        "nl_male", "nl_female",
        "ar_male", "ar_female",
        "hi_male", "hi_female",
    ]


def _get_model():
    global _model_cache
    if _model_cache is None:
        from mlx_audio.tts.utils import load
        logger.info(f"Loading Voxtral model: {VOXTRAL_MODEL_ID}")
        _model_cache = load(VOXTRAL_MODEL_ID)
        logger.info("Voxtral model loaded successfully")
    return _model_cache


class VoxtralTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.voice_name = config.voice_name or "neutral_male"
        super().__init__(config)

    def validate_config(self):
        if self.config.voice_name not in get_voxtral_supported_voices():
            raise ValueError(
                f"Voxtral: Unsupported voice: {self.config.voice_name}. "
                f"Supported voices: {get_voxtral_supported_voices()}"
            )

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        import numpy as np

        text_chunks = split_text(text, VOXTRAL_MAX_CHARS, self.config.language)
        model = _get_model()
        all_audio = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(f"Processing {chunk_id}, length={len(chunk)}")
            logger.debug(f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]")

            for result in model.generate(text=chunk, voice=self.config.voice_name):
                all_audio.append(np.array(result.audio))

        if not all_audio:
            logger.warning(f"No audio generated for {output_file}")
            return

        audio = np.concatenate(all_audio)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(VOXTRAL_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        try:
            set_audio_tags(output_file, audio_tags)
        except Exception as e:
            logger.warning(f"Could not set audio tags for WAV file: {e}")

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return "wav"

    def estimate_cost(self, total_chars):
        return 0.0
