# AGENTS.md

## Commands

```bash
# Tests need PYTHONPATH (no pyproject.toml / setup.py exists)
PYTHONPATH=. python3 -m unittest discover tests/

# Single test file
PYTHONPATH=. python3 -m unittest tests/audiobook_generator/tts_providers/base_tts_provider_test.py

# Single test case
PYTHONPATH=. python3 -m unittest tests.audiobook_generator.tts_providers.azure_tts_provider_test.AzureTTSProviderTest.test_name

# CLI & WebUI
python3 main.py <input.epub> <output_folder> --tts <azure|openai|edge|piper|voxtral|vibevoice|f5>
python3 main_ui.py --host 127.0.0.1 --port 7860

# Install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## Architecture

**7 TTS providers** (all extend `BaseTTSProvider` in `audiobook_generator/tts_providers/`):
- `azure`, `openai`, `edge`, `piper`, `voxtral`, `vibevoice`, `f5`

**Provider registration** — NOT in `__init__.py` (it's empty). Done in `base_tts_provider.py:get_tts_provider()` and `get_supported_tts_providers()`. To add a new provider: create file, add import + branch in `get_tts_provider()`, add constant in `base_tts_provider.py`, register CLI args in `main.py`, add config fields in `general_config.py`, add Gradio tab in `web_ui.py`.

**Config** — `GeneralConfig` built from `argparse` namespace via `getattr(args, field, None)`. Tests build configs via `unittest.mock.MagicMock` attribute stubs in `tests/test_utils.py`.

**Core flow**: `EpubBookParser.get_chapters()` → `AudiobookGenerator` splits chapters across `multiprocessing.Pool(processes=worker_count, initializer=setup_logging)` → each worker creates fresh `TTSProvider` instance → `text_to_speech()` splits text via `split_text()` (sentencex, language-aware), streams audio segments, merges via pydub or direct write, tags MP3s.

## Key Deviations & Gotchas

- **No pyproject.toml/setup.py** — package isn't installable. Always use `PYTHONPATH=.` or run from project root.
- **Tests patch env vars** — cloud providers (`azure`, `openai`) require `@patch.dict('os.environ', ...)` for their API keys.
- **CLI `--newline_mode` default is `double`**, not `single`.
- **CLI `--break_duration`** is shared by azure+edge, not azure-only.
- **`--no_prompt`** skips the cost confirmation prompt (needed for scripting/WebUI).
- **`--use_pydub_merge`** — slower but handles mixed audio formats; requires `ffmpeg`.
- **`--remove_reference_numbers`** (e.g. `[3]`, `[12.1]`) is separate from `--remove_endnotes`.
- **Logs** go to `logs/EtA_<timestamp>.log` automatically. Worker processes use `[Worker-PID]` prefix.
- **Docker**: app code at `/app_src`, user files at `/app` (important: mounting to `/app` won't shadow app code). The `entrypoint.sh` auto-detects CLI vs WebUI.
- **`--model_name tts-1` is required for Kokoro** (OpenAI-compatible local TTS), otherwise Kokoro breaks.
- **Voxtral** uses `mlx-audio` (Apple Silicon only), model auto-downloads from HF on first use.
- **VibeVoice** downloads GGUF model + base weights from HF on first use; requires `torch`.
- **F5** uses `f5-tts-mlx` (Apple Silicon via MLX), model auto-downloads from HF on first use. Default ref voice is built-in; optionally provide custom `.wav` (mono, 24kHz) for zero-shot cloning.

## Env Vars

| Provider | Env Vars |
|----------|----------|
| Azure | `MS_TTS_KEY`, `MS_TTS_REGION` |
| OpenAI | `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL` |
| Edge | none |
| Piper | none (binary path via `--piper_path`) |
| Voxtral | none |
| VibeVoice | none |
| F5 | none |
