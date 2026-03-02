import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from audiobook_generator.utils.utils import set_audio_tags

logger = logging.getLogger(__name__)

# Default sample rate (24 kHz) for Kokoro and Qwen3-TTS
MLX_DEFAULT_SAMPLE_RATE = 24000

# Map lang_code (shared CLI/UI) to Qwen3-TTS language name
QWEN3_LANG_CODE_TO_NAME = {
    "a": "English",
    "b": "English",
    "j": "Japanese",
    "z": "Chinese",
    "e": "Spanish",
    "f": "French",
}


def get_mlx_default_model() -> str:
    """Default MLX-Audio TTS model (Qwen3-TTS)."""
    return "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


def get_mlx_kokoro_voices():
    """Kokoro voice presets for dropdowns (subset of 54)."""
    return [
        "af_heart",
        "af_bella",
        "af_nova",
        "af_sky",
        "am_adam",
        "am_echo",
        "bf_alice",
        "bf_emma",
        "bm_daniel",
        "bm_george",
        "jf_alpha",
        "jm_kumo",
        "zf_xiaobei",
        "zm_yunxi",
    ]


def get_mlx_qwen3_voices():
    """Qwen3-TTS voice presets (Base model built-in voices)."""
    return [
        "Chelsie",
        "Drew",
        "Emily",
        "Josh",
        "Molly",
        "Ryan",
    ]


def get_mlx_voices_for_model(model_id: str):
    """Return voice list for the given model (Qwen3 vs Kokoro)."""
    if model_id and "Qwen3" in model_id:
        return get_mlx_qwen3_voices()
    return get_mlx_kokoro_voices()


def get_mlx_lang_codes():
    """Shared language codes: a=American EN, b=British EN, j=JA, z=ZH, e=ES, f=FR."""
    return [
        ("a", "American English"),
        ("b", "British English"),
        ("j", "Japanese"),
        ("z", "Mandarin Chinese"),
        ("e", "Spanish"),
        ("f", "French"),
    ]


class MLXAudioTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.output_format = config.output_format or "mp3"
        self.price = 0.0
        self.tts_model = None
        self.sample_rate = MLX_DEFAULT_SAMPLE_RATE
        super().__init__(config)

    def __str__(self) -> str:
        return f"MLXAudioTTSProvider(config={self.config})"

    def validate_config(self):
        """Validate MLX-Audio configuration. Optionally warn if not Apple Silicon."""
        if sys.platform != "darwin":
            logger.warning(
                "MLX-Audio is optimized for Apple Silicon (macOS). "
                "It may not work or may be slow on other platforms."
            )

    def _load_model(self):
        """Lazy-load the MLX-Audio TTS model."""
        if self.tts_model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError:
            raise ImportError(
                "mlx-audio is not installed. "
                "Install with: pip install mlx-audio (Apple Silicon only)"
            )
        model_id = self.config.mlx_model or get_mlx_default_model()
        logger.info("Loading MLX-Audio model: %s", model_id)
        try:
            self.tts_model = load_model(model_id)
            logger.info("MLX-Audio model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX-Audio model: {e}") from e

    def _audio_to_numpy(self, audio):
        """Convert mlx array (or similar) to numpy. Returns None if audio is None."""
        if audio is None:
            return None
        try:
            return np.array(audio)
        except Exception:
            return np.asarray(
                audio.tolist() if hasattr(audio, "tolist") else audio
            )

    def _is_qwen3_model(self) -> bool:
        model_id = (self.config.mlx_model or get_mlx_default_model()).lower()
        return "qwen3" in model_id

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        """
        Convert text to speech using MLX-Audio (Qwen3-TTS or Kokoro).
        """
        self._load_model()

        model_id = self.config.mlx_model or get_mlx_default_model()
        is_qwen3 = self._is_qwen3_model()

        # Ensure no None reaches mlx-audio (it can do path + str internally)
        if is_qwen3:
            voice = (self.config.mlx_voice or "Chelsie").strip() or "Chelsie"
            lang_code = (self.config.mlx_lang_code or "a").strip() or "a"
            language = QWEN3_LANG_CODE_TO_NAME.get(lang_code, "English")
        else:
            voice = (self.config.mlx_voice or "af_heart").strip() or "af_heart"
            speed = float(self.config.mlx_speed or 1.0)
            lang_code = (self.config.mlx_lang_code or "a").strip() or "a"
            language = None
        voice = str(voice)
        lang_code = str(lang_code)

        logger.info(
            "Generating audio for text of length %d characters (voice=%s, lang=%s)",
            len(text),
            voice,
            language or lang_code,
        )

        try:
            audio_chunks = []

            if is_qwen3:
                # Qwen3-TTS: generate with temperature=0 for consistent voice across chunks
                results = list(
                    self.tts_model.generate(
                        text=text,
                        voice=voice,
                        language=language,
                        temperature=0.0,
                    )
                )
                for result in results:
                    arr = self._audio_to_numpy(getattr(result, "audio", None))
                    if arr is not None:
                        audio_chunks.append(arr)
            else:
                # Kokoro: generate(text=..., voice=..., speed=..., lang_code=...) -> iterator
                speed = float(self.config.mlx_speed or 1.0)
                for result in self.tts_model.generate(
                    text=text,
                    voice=voice,
                    speed=speed,
                    lang_code=lang_code,
                ):
                    arr = self._audio_to_numpy(getattr(result, "audio", None))
                    if arr is not None:
                        audio_chunks.append(arr)

            if not audio_chunks:
                raise RuntimeError("MLX-Audio generated no audio chunks")

            audio_data = np.concatenate(audio_chunks)

            # Normalize to int16 if float
            if audio_data.dtype in (np.float32, np.float64):
                audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / "mlx_audio.wav"
                segment = AudioSegment(
                    data=audio_data.tobytes(),
                    sample_width=2,
                    frame_rate=self.sample_rate,
                    channels=1,
                )
                segment.export(tmp_path, format="wav")
                if audio_tags:
                    set_audio_tags(tmp_path, audio_tags)
                AudioSegment.from_wav(tmp_path).export(
                    output_file, format=self.config.output_format
                )
            logger.info("Conversion completed, output file: %s", output_file)
        except TypeError as e:
            if "NoneType" in str(e) and "operand" in str(e):
                raise RuntimeError(
                    "MLX-Audio failed with a NoneType error. This often happens when "
                    "espeak/espeak-ng is not installed (needed by Kokoro for some languages). "
                    "Try: brew install espeak-ng (macOS) or sudo apt install espeak-ng (Linux). "
                    "Original error: %s" % e
                ) from e
            raise
        except Exception as e:
            logger.error("Error generating speech with MLX-Audio: %s", e)
            raise

    def estimate_cost(self, total_chars):
        return 0

    def get_break_string(self):
        return "."

    def get_output_file_extension(self):
        return self.config.output_format
