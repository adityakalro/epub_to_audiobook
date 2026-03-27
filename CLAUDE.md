# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run CLI
python3 main.py <input.epub> <output_folder> --tts <azure|openai|edge|piper>

# Run WebUI
python3 main_ui.py --host 127.0.0.1 --port 7860

# Run all tests
python3 -m unittest discover tests/

# Run a single test file
python3 -m unittest tests/audiobook_generator/book_parsers/epub_book_parser_test.py

# Run a single test case
python3 -m unittest tests.audiobook_generator.tts_providers.azure_tts_provider_test.AzureTTSProviderTest.test_name
```

## Architecture

The app converts EPUB books to MP3 audiobooks via a **layered plugin architecture**:

```
CLI (main.py) / WebUI (main_ui.py)
         ↓
  GeneralConfig / UiConfig       (audiobook_generator/config/)
         ↓
  EpubBookParser                 (audiobook_generator/book_parsers/)
  — extracts chapters, cleans text, handles newlines/endnotes/regex replacements
         ↓
  AudiobookGenerator             (audiobook_generator/core/)
  — multiprocessing pool across chapters
         ↓
  TTS Provider                   (audiobook_generator/tts_providers/)
  — splits text via split_text() → calls TTS API → merges audio → tags MP3
         ↓
  Output: numbered MP3 files + optional .txt files
```

### Key Layers

**TTS Providers** (`audiobook_generator/tts_providers/`) — all extend `BaseTTSProvider`:
- `AzureTTSProvider` — Microsoft Azure Cognitive Services
- `OpenAITTSProvider` — OpenAI TTS API
- `EdgeTTSProvider` — Microsoft Edge read-aloud (free, no key required)
- `PiperTTSProvider` — local open-source TTS

Each provider implements: `validate_config()`, `text_to_speech()`, `estimate_cost()`, `get_break_string()`, `get_output_file_extension()`.

**Book Parser** (`audiobook_generator/book_parsers/EpubBookParser`) — extracts chapters using ebooklib + BeautifulSoup; supports `--search_and_replace_file` for regex-based text correction.

**Core Engine** (`audiobook_generator/core/audiobook_generator.py`) — `AudiobookGenerator` orchestrates multiprocessing, cost estimation, preview mode, and failure tracking.

**Utilities** (`audiobook_generator/utils/`):
- `split_text()` — language-aware chunking via sentencex
- `set_audio_tags()` — MP3 ID3 metadata injection
- `merge_audio_segments()` — pydub-based audio merging
- `filename_sanitizer.py` — cross-platform safe filenames
- `docker_helper.py` — Piper Docker container management

**WebUI** (`audiobook_generator/ui/web_ui.py`) — Gradio interface with per-provider tabs, real-time log viewer, and subprocess isolation for conversion.

### Adding a New TTS Provider

1. Create `audiobook_generator/tts_providers/<name>_tts_provider.py` extending `BaseTTSProvider`
2. Register it in `audiobook_generator/tts_providers/__init__.py`
3. Add CLI args in `audiobook_generator/config/general_config.py`
4. Add a tab in `audiobook_generator/ui/web_ui.py`

## Testing

Tests use `unittest` with `unittest.mock`. The file `tests/test_utils.py` provides helper functions (`get_azure_config()`, `get_openai_config()`) for building test configs. Sample EPUB: `examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub`.

## Environment Variables

Cloud TTS providers use environment variables for credentials:
- Azure: `MS_TTS_KEY`, `MS_TTS_REGION`
- OpenAI: `OPENAI_API_KEY`

## Docker

```bash
# CLI via Docker
docker run -v ./:/app -e MS_TTS_KEY=$KEY -e MS_TTS_REGION=$REGION \
  ghcr.io/p0n1/epub_to_audiobook book.epub output --tts azure

# WebUI via Docker Compose
docker compose -f docker-compose.webui.yml up
```

Dockerfile uses `python:3.11-slim-trixie`, separates app code (`/app_src`) from user files (`/app`).
