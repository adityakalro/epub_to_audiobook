import glob
import logging
import os
import wave

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

VIBEVOICE_GGUF_REPO = "gguf-org/vibevoice-gguf"
VIBEVOICE_MODELS = {
    "q4_k_m": "vibevoice-1.5b-q4_k_m.gguf",
    "iq4_nl":  "vibevoice-1.5b-iq4_nl.gguf",
    "q4_0":    "vibevoice-1.5b-q4_0.gguf",
    "iq4_xs":  "vibevoice-1.5b-iq4_xs.gguf",
    "q4_1":    "vibevoice-1.5b-q4_1.gguf",
}
VIBEVOICE_DEFAULT_MODEL = "q4_k_m"
VIBEVOICE_BASE_REPO_OWNER = "callgg"
VIBEVOICE_BASE_REPO_NAME = "vibevoice-bf16"
VIBEVOICE_BASE_GHASH = "53a915ae1a937cde20531290877f23aee39a7cc21786ff3a783158ac443ae74d"
VIBEVOICE_MAX_CHARS = 500
VIBEVOICE_DEFAULT_CFG_PACE = 1.3


def get_vibevoice_supported_models():
    return list(VIBEVOICE_MODELS.keys())

_model_cache = {}
_processor_cache = {}


def _get_model_and_processor(model_key=VIBEVOICE_DEFAULT_MODEL):
    global _model_cache, _processor_cache
    if model_key not in _model_cache:
        import torch
        from huggingface_hub import hf_hub_download
        from gguf_connector.tph import get_hf_cache_hub_path
        from gguf_connector.quant3 import convert_gguf_to_safetensors
        from gguf_connector.quant4 import add_metadata_to_safetensors
        from yvoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from yvoice.processor.vibevoice_processor import VibeVoiceProcessor

        gguf_file = VIBEVOICE_MODELS[model_key]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        logger.info(f"Downloading VibeVoice 4-bit GGUF ({model_key}): {gguf_file}")
        gguf_path = hf_hub_download(repo_id=VIBEVOICE_GGUF_REPO, filename=gguf_file)

        model_path = get_hf_cache_hub_path(
            VIBEVOICE_BASE_REPO_OWNER, VIBEVOICE_BASE_REPO_NAME, VIBEVOICE_BASE_GHASH
        )

        if not glob.glob(os.path.join(model_path, "*.safetensors")):
            logger.info(f"Dequantizing GGUF to SafeTensors at {model_path}")
            convert_gguf_to_safetensors(gguf_path, model_path, use_bf16)
            add_metadata_to_safetensors(model_path, {"format": "pt"})
        else:
            logger.info(f"Using cached dequantized model at {model_path}")

        logger.info(f"Loading VibeVoice model ({model_key})")
        _model_cache[model_key] = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path, dtype=dtype, device_map=device
        )
        _processor_cache[model_key] = VibeVoiceProcessor.from_pretrained(model_path)
        logger.info(f"VibeVoice model ({model_key}) loaded successfully")

    return _model_cache[model_key], _processor_cache[model_key]


class VibeVoiceTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.vibevoice_cfg_pace = config.vibevoice_cfg_pace or VIBEVOICE_DEFAULT_CFG_PACE
        config.vibevoice_model = config.vibevoice_model or VIBEVOICE_DEFAULT_MODEL
        super().__init__(config)

    def validate_config(self):
        if not self.config.vibevoice_ref_audio:
            raise ValueError(
                "VibeVoice requires a reference audio file for voice cloning. "
                "Provide a WAV file path via --vibevoice_ref_audio."
            )
        if not os.path.isfile(self.config.vibevoice_ref_audio):
            raise ValueError(
                f"VibeVoice reference audio file not found: {self.config.vibevoice_ref_audio}"
            )
        if self.config.vibevoice_model not in VIBEVOICE_MODELS:
            raise ValueError(
                f"Unsupported VibeVoice model '{self.config.vibevoice_model}'. "
                f"Choose from: {list(VIBEVOICE_MODELS.keys())}"
            )

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        import numpy as np
        import torch

        text_chunks = split_text(text, VIBEVOICE_MAX_CHARS, self.config.language)
        model, processor = _get_model_and_processor(self.config.vibevoice_model)
        device = next(model.parameters()).device
        all_audio = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(f"Processing {chunk_id}, length={len(chunk)}")
            logger.debug(f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]")

            inputs = processor(
                text=[chunk],
                voice_samples=[[self.config.vibevoice_ref_audio]],
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in inputs.items()
            }
            output = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                cfg_scale=float(self.config.vibevoice_cfg_pace),
                max_new_tokens=None,
            )
            speech_tensor = output.speech_outputs[0]
            all_audio.append(speech_tensor.cpu().float().numpy())

        if not all_audio:
            logger.warning(f"No audio generated for {output_file}")
            return

        audio = np.concatenate(all_audio)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        sample_rate = processor.audio_processor.sampling_rate

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
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
