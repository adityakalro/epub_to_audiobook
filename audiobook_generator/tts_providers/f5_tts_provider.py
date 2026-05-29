import io
import logging
import wave

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

F5_SAMPLE_RATE = 24000
F5_MAX_CHARS = 500
F5_DEFAULT_MODEL = "lucasnewman/f5-tts-mlx"
F5_DEFAULT_STEPS = 8
F5_DEFAULT_METHOD = "rk4"
F5_DEFAULT_CFG_STRENGTH = 2.0
F5_DEFAULT_SPEED = 1.0
F5_DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."

F5_METHODS = ["euler", "midpoint", "rk4"]

_model_cache = {}


def get_f5_supported_methods():
    return list(F5_METHODS)


def _get_model(model_name=F5_DEFAULT_MODEL, quantization_bits=None):
    global _model_cache
    cache_key = f"{model_name}_q{quantization_bits}" if quantization_bits else model_name
    if cache_key not in _model_cache:
        from f5_tts_mlx.cfm import F5TTS
        logger.info(f"Loading F5-TTS model: {model_name}, quantization: {quantization_bits}")
        _model_cache[cache_key] = F5TTS.from_pretrained(
            model_name, quantization_bits=quantization_bits
        )
        logger.info("F5-TTS model loaded successfully")
    return _model_cache[cache_key]


def _load_ref_audio(ref_audio_path=None, ref_text=None):
    import pkgutil
    import soundfile as sf
    import mlx.core as mx

    if ref_audio_path:
        audio, sr = sf.read(ref_audio_path)
        if sr != F5_SAMPLE_RATE:
            raise ValueError(
                f"F5-TTS: Reference audio must have sample rate {F5_SAMPLE_RATE}Hz, got {sr}Hz"
            )
        audio_text = ref_text or ""
        if not ref_text:
            logger.warning("No reference text provided; voice quality may be degraded")
    else:
        data = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
        audio, sr = sf.read(io.BytesIO(data))
        audio_text = F5_DEFAULT_REF_TEXT

    audio = mx.array(audio)
    rms = mx.sqrt(mx.mean(mx.square(audio)))
    target_rms = 0.1
    if rms < target_rms:
        audio = audio * target_rms / rms

    return audio, audio_text


def get_f5_supported_models():
    return [F5_DEFAULT_MODEL]


class F5TTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or F5_DEFAULT_MODEL
        config.f5_steps = config.f5_steps or F5_DEFAULT_STEPS
        config.f5_method = config.f5_method or F5_DEFAULT_METHOD
        config.f5_cfg_strength = config.f5_cfg_strength or F5_DEFAULT_CFG_STRENGTH
        config.f5_speed = config.f5_speed or F5_DEFAULT_SPEED
        super().__init__(config)

    def validate_config(self):
        if self.config.f5_method not in F5_METHODS:
            raise ValueError(
                f"F5-TTS: Unsupported method '{self.config.f5_method}'. "
                f"Choose from: {F5_METHODS}"
            )
        if self.config.f5_ref_audio and not self.config.f5_ref_text:
            logger.warning(
                "F5-TTS: reference audio provided without reference text; "
                "voice quality may be degraded. Use --f5_ref_text"
            )

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        import mlx.core as mx
        import numpy as np
        from f5_tts_mlx.utils import convert_char_to_pinyin

        text_chunks = split_text(text, F5_MAX_CHARS, self.config.language)
        f5tts = _get_model(self.config.model_name, self.config.f5_quantization_bits)
        ref_audio, ref_text = _load_ref_audio(
            self.config.f5_ref_audio, self.config.f5_ref_text
        )
        all_audio = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(f"Processing {chunk_id}, length={len(chunk)}")
            logger.debug(f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]")

            combined = convert_char_to_pinyin([ref_text + " " + chunk])

            audio_wave, _ = f5tts.sample(
                ref_audio[None, ...],
                text=combined,
                duration=None,
                steps=self.config.f5_steps,
                method=self.config.f5_method,
                speed=self.config.f5_speed,
                cfg_strength=self.config.f5_cfg_strength,
                sway_sampling_coef=-1.0,
                seed=None,
            )
            audio_wave = audio_wave[ref_audio.shape[0]:]
            mx.eval(audio_wave)
            all_audio.append(np.array(audio_wave))

        if not all_audio:
            logger.warning(f"No audio generated for {output_file}")
            return

        audio = np.concatenate(all_audio)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(F5_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        logger.debug(f"Skipping ID3 tags for WAV output: {output_file}")

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return "wav"

    def estimate_cost(self, total_chars):
        return 0.0
