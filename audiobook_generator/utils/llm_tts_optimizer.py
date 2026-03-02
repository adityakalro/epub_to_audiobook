"""
LLM TTS optimizer: run chapter text through a local LLM (Ollama) to improve
punctuation, emotion, and intonation for natural TTS. Long chapters are split
at paragraph or sentence boundaries.
"""

import json
import logging
import os

import requests

from sentencex import segment

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_MAX_CHARS = 4000
OLLAMA_TIMEOUT_SECONDS = 120

#TTS_OPTIMIZATION_SYSTEM = """You are a text editor. Output ONLY the rewritten text, no preamble, explanation, or quotes around it. 
#Preserve paragraph breaks as they are. Rewrite the following text so it is optimized for text-to-speech: 
#improve punctuation, preserve or add cues for emotion and intonation, and keep the exact same content so it reads naturally when spoken. """

TTS_OPTIMIZATION_SYSTEM = """You are a script writer.  Create an input for a TTS model from the following text segment. Introduce the appropriate line breaks to that the TTS will pause and add the right intonations. Only output the script, no preamble or suffix. Add any notes that you want the tts system to account for in square brackets."""

def _split_chapter_for_llm(text: str, max_chars: int, language: str) -> list[str]:
    """
    Split chapter text into chunks for the LLM, at paragraph or sentence boundaries.
    Never cuts mid-paragraph or mid-sentence.
    """
    if not text or max_chars <= 0:
        return [text] if text else []

    lang = language or "en"
    # Normalize to sentencex-style code if needed (e.g. "en-US" -> "en" for segment)
    if lang and "-" in lang:
        lang = lang.split("-")[0]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")

    current: list[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            if current:
                current.append("")
                current_len += 2  # \n\n
            continue

        need_join = 2 if current else 0  # \n\n between paragraphs
        if current_len + need_join + len(para_stripped) <= max_chars:
            if current:
                current.append(para_stripped)
                current_len += need_join + len(para_stripped)
            else:
                current.append(para_stripped)
                current_len = len(para_stripped)
            continue

        # Adding this paragraph would exceed limit: flush current first
        flush()

        if len(para_stripped) <= max_chars:
            current.append(para_stripped)
            current_len = len(para_stripped)
            continue

        # Paragraph itself is too long: split by sentences
        try:
            sentences = list(segment(lang, para_stripped))
        except Exception:
            sentences = [para_stripped]

        for sent in sentences:
            s = sent.strip() if isinstance(sent, str) else str(sent).strip()
            if not s:
                continue
            need_join = 1 if current else 0
            if current_len + need_join + len(s) <= max_chars:
                if current:
                    current.append(s)
                    current_len += need_join + len(s)
                else:
                    current.append(s)
                    current_len = len(s)
            else:
                flush()
                if len(s) <= max_chars:
                    current.append(s)
                    current_len = len(s)
                else:
                    # Single sentence longer than max_chars: add as one chunk
                    chunks.append(s)
                    current = []
                    current_len = 0

    flush()
    return chunks if chunks else [text]


def _call_ollama(prompt: str, ollama_url: str, ollama_model: str) -> str | None:
    """Call Ollama /api/generate. Returns response text or None on failure."""
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        out = data.get("response")
        return out.strip() if isinstance(out, str) else None
    except requests.exceptions.RequestException as e:
        logger.warning("Ollama request failed: %s", e)
        return None
    except (KeyError, TypeError) as e:
        logger.warning("Ollama response invalid: %s", e)
        return None


def optimize_chapter_for_tts(
    text: str,
    ollama_url: str,
    ollama_model: str,
    language: str | None = None,
    max_chars: int = 4000,
) -> str:
    """
    Run chapter text through Ollama to optimize for TTS. Long text is split
    at paragraph/sentence boundaries; each chunk is sent separately and
    results are concatenated. On failure for a chunk, the original chunk
    is kept.
    """
    if not text or not text.strip():
        return text

    lang = language or "en-US"
    chunks = _split_chapter_for_llm(text, max_chars, lang)
    results: list[str] = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            results.append(chunk)
            continue
        prompt = f"{TTS_OPTIMIZATION_SYSTEM}\n\n---\n\n{chunk}"
        if len(chunks) > 1:
            prompt = (
                "This is a segment of a longer chapter; preserve meaning and style. "
                + prompt
            )
        optimized = _call_ollama(prompt, ollama_url, ollama_model)
        if optimized is not None:
            results.append(optimized)
        else:
            logger.warning(
                "LLM optimization failed for chunk %s/%s, using original text",
                i + 1,
                len(chunks),
            )
            results.append(chunk)

    return "\n\n".join(results)


def load_ollama_config(config_path: str | None = None) -> dict:
    """
    Load Ollama TTS config from JSON file. Returns dict with ollama_url,
    ollama_model, ollama_max_chars. Uses defaults for missing keys.
    """
    defaults = {
        "ollama_url": DEFAULT_OLLAMA_URL,
        "ollama_model": DEFAULT_OLLAMA_MODEL,
        "ollama_max_chars": DEFAULT_OLLAMA_MAX_CHARS,
    }

    path = config_path
    if path is None:
        path = os.path.join(os.getcwd(), "ollama_tts_config.json")

    if not path or not os.path.isfile(path):
        return defaults.copy()

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load Ollama config from %s: %s", path, e)
        return defaults.copy()

    return {
        "ollama_url": data.get("ollama_url") or defaults["ollama_url"],
        "ollama_model": data.get("ollama_model") or defaults["ollama_model"],
        "ollama_max_chars": int(data.get("ollama_max_chars", defaults["ollama_max_chars"])),
    }
