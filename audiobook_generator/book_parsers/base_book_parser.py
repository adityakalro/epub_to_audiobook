import re
from typing import List, Tuple

from audiobook_generator.config.general_config import GeneralConfig

EPUB = "epub"
MARKDOWN = "markdown"


class BaseBookParser:  # Base interface for books parsers
    # Base Book Parser interface
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.validate_config()

    def __str__(self) -> str:
        return f"{self.config}"

    def validate_config(self):
        raise NotImplementedError

    def get_book(self):
        raise NotImplementedError

    def get_book_title(self) -> str:
        raise NotImplementedError

    def get_book_author(self) -> str:
        raise NotImplementedError

    def get_chapters(self, break_string) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def get_search_and_replaces(self):
        search_and_replaces = []
        if self.config.search_and_replace_file:
            with open(self.config.search_and_replace_file) as fp:
                for line in fp.readlines():
                    if '==' in line and not line.startswith('==') and not line.endswith('==') and not line.startswith('#'):
                        search_and_replaces.append({
                            'search': r"{}".format(line.split('==')[0]),
                            'replace': r"{}".format(line.split('==')[1][:-1])
                        })
        return search_and_replaces

    @staticmethod
    def _sanitize_title(title, break_string) -> str:
        title = title.replace(break_string, " ")
        sanitized_title = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
        sanitized_title = re.sub(r"\s+", "_", sanitized_title.strip())
        return sanitized_title


# Common support methods for all book parsers

def get_supported_book_parsers() -> List[str]:
    return [EPUB, MARKDOWN]


def get_book_parser(config) -> BaseBookParser:
    input_format = getattr(config, 'input_format', None)
    if input_format is None:
        if config.input_file.endswith(".epub"):
            input_format = EPUB
        elif config.input_file.endswith(".md") or config.input_file.endswith(".markdown"):
            input_format = MARKDOWN

    if input_format == EPUB:
        from audiobook_generator.book_parsers.epub_book_parser import EpubBookParser
        return EpubBookParser(config)
    elif input_format == MARKDOWN:
        from audiobook_generator.book_parsers.markdown_book_parser import MarkdownBookParser
        return MarkdownBookParser(config)
    else:
        raise NotImplementedError(f"Unsupported file format: {config.input_file}")
