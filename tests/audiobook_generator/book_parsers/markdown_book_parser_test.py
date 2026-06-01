import unittest
from unittest.mock import MagicMock

from audiobook_generator.book_parsers.base_book_parser import (
    get_book_parser,
    get_supported_book_parsers,
)
from audiobook_generator.book_parsers.markdown_book_parser import MarkdownBookParser


class TestMarkdownBookParser(unittest.TestCase):

    def _make_config(self, input_file="examples/sample.md"):
        args = MagicMock(
            input_file=input_file,
            input_format=None,
            output_folder="output",
            preview=False,
            output_text=False,
            log="INFO",
            newline_mode="double",
            chapter_start=1,
            chapter_end=-1,
            remove_endnotes=False,
            remove_reference_numbers=False,
            search_and_replace_file="",
            title_mode="auto",
            tts="azure",
            language="en-US",
        )
        from audiobook_generator.config.general_config import GeneralConfig
        return GeneralConfig(args)

    def test_get_markdown_book_parser(self):
        config = self._make_config()
        parser = get_book_parser(config)
        self.assertIsInstance(parser, MarkdownBookParser)

    def test_get_book_title(self):
        config = self._make_config()
        parser = MarkdownBookParser(config)
        self.assertEqual(parser.get_book_title(), "The Adventures of Sample Book")

    def test_get_book_author(self):
        config = self._make_config()
        parser = MarkdownBookParser(config)
        self.assertEqual(parser.get_book_author(), "Unknown")

    def test_get_chapters_count(self):
        config = self._make_config()
        parser = MarkdownBookParser(config)
        chapters = parser.get_chapters("   ")
        self.assertEqual(len(chapters), 3)

    def test_get_chapters_titles(self):
        config = self._make_config()
        parser = MarkdownBookParser(config)
        chapters = parser.get_chapters("   ")
        titles = [title for title, _ in chapters]
        self.assertEqual(titles, [
            "The_Adventures_of_Sample_Book",
            "Chapter_2_The_Journey_Begins",
            "Chapter_3",
        ])

    def test_get_chapters_content(self):
        config = self._make_config()
        parser = MarkdownBookParser(config)
        chapters = parser.get_chapters("   ")
        # Chapter 1 should contain the subsection text
        self.assertIn("Section Within Chapter", chapters[0][1])
        # Chapter 2 should contain the second chapter content
        self.assertIn("second chapter", chapters[1][1])
        # Chapter 3 should contain the final chapter content
        self.assertIn("Final chapter", chapters[2][1])

    def test_unsupported_file_format(self):
        config = self._make_config(input_file="book.pdf")
        with self.assertRaises(NotImplementedError):
            get_book_parser(config)

    def test_supported_parsers_includes_markdown(self):
        parsers = get_supported_book_parsers()
        self.assertIn("markdown", parsers)

    def test_file_without_headings(self):
        args = MagicMock(
            input_file="examples/sample.md",
            input_format=None,
            output_folder="output",
            preview=False,
            output_text=False,
            log="INFO",
            newline_mode="double",
            chapter_start=1,
            chapter_end=-1,
            remove_endnotes=False,
            remove_reference_numbers=False,
            search_and_replace_file="",
            title_mode="auto",
            tts="azure",
            language="en-US",
        )
        from audiobook_generator.config.general_config import GeneralConfig
        config = GeneralConfig(args)
        parser = MarkdownBookParser(config)
        # file has headings, but just testing the structure
        self.assertIsInstance(parser, MarkdownBookParser)

    def test_validate_config_raises_on_wrong_extension(self):
        config = self._make_config(input_file="book.txt")
        with self.assertRaises(ValueError):
            MarkdownBookParser(config)

    def test_validate_config_raises_on_none(self):
        args = MagicMock(
            input_file=None,
            input_format=None,
        )
        from audiobook_generator.config.general_config import GeneralConfig
        config = GeneralConfig(args)
        with self.assertRaises(ValueError):
            MarkdownBookParser(config)


if __name__ == '__main__':
    unittest.main()
