import logging
import os
import re
from typing import List, Tuple

from audiobook_generator.book_parsers.base_book_parser import BaseBookParser
from audiobook_generator.config.general_config import GeneralConfig

logger = logging.getLogger(__name__)


class MarkdownBookParser(BaseBookParser):
    def __init__(self, config: GeneralConfig):
        super().__init__(config)
        with open(self.config.input_file, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

    def validate_config(self):
        if self.config.input_file is None:
            raise ValueError("Markdown Parser: Input file cannot be empty")
        if not (self.config.input_file.endswith(".md") or self.config.input_file.endswith(".markdown")):
            raise ValueError(f"Markdown Parser: Unsupported file format: {self.config.input_file}")

    def get_book(self):
        return self.raw_text

    def get_book_title(self) -> str:
        for line in self.raw_text.splitlines():
            if line.startswith("# "):
                return line.lstrip("# ").strip()
        return os.path.splitext(os.path.basename(self.config.input_file))[0]

    def get_book_author(self) -> str:
        return "Unknown"

    def get_chapters(self, break_string) -> List[Tuple[str, str]]:
        chapters = []
        search_and_replaces = self.get_search_and_replaces()
        lines = self.raw_text.splitlines()

        heading_indices = []
        for i, line in enumerate(lines):
            if line.startswith("# "):
                heading_indices.append(i)

        if not heading_indices:
            raw = "\n".join(lines)
            cleaned_text = self._clean_text(raw, break_string, search_and_replaces)
            title = self._sanitize_title(
                os.path.splitext(os.path.basename(self.config.input_file))[0],
                break_string
            )
            if cleaned_text.strip():
                chapters.append((title, cleaned_text))
        else:
            if heading_indices[0] > 0:
                raw = "\n".join(lines[:heading_indices[0]])
                cleaned_text = self._clean_text(raw, break_string, search_and_replaces)
                if cleaned_text.strip():
                    title = self._sanitize_title("Introduction", break_string)
                    chapters.append((title, cleaned_text))

            for idx, h_idx in enumerate(heading_indices):
                title_raw = lines[h_idx].lstrip("# ").strip()
                start = h_idx + 1
                end = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(lines)
                raw = "\n".join(lines[start:end])
                cleaned_text = self._clean_text(raw, break_string, search_and_replaces)
                title = self._sanitize_title(title_raw, break_string)
                chapters.append((title, cleaned_text))

        return chapters

    def _clean_text(self, raw: str, break_string: str, search_and_replaces: list) -> str:
        if self.config.newline_mode == "single":
            cleaned = re.sub(r"[\n]+", break_string, raw.strip())
        elif self.config.newline_mode == "double":
            cleaned = re.sub(r"[\n]{2,}", break_string, raw.strip())
        elif self.config.newline_mode == "none":
            cleaned = re.sub(r"[\n]+", " ", raw.strip())
        else:
            raise ValueError(f"Invalid newline mode: {self.config.newline_mode}")

        cleaned = re.sub(r"\s+", " ", cleaned)

        if self.config.remove_endnotes:
            cleaned = re.sub(r'(?<=[a-zA-Z.,!?;")])\d+', "", cleaned)

        if self.config.remove_reference_numbers:
            cleaned = re.sub(r'\[\d+(\.\d+)?\]', '', cleaned)

        for sar in search_and_replaces:
            cleaned = re.sub(sar['search'], sar['replace'], cleaned)

        return cleaned.strip()
