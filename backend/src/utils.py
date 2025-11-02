import time
import functools
import logging
from pathlib import Path

from markitdown import MarkItDown

logger = logging.getLogger(__name__)


def to_markdown(filepath: Path) -> str:
    markdown_converter = MarkItDown()
    output = markdown_converter.convert(filepath)
    markdown_text: str = output.text_content
    return markdown_text


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        retval = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info("Elapsed: {elapsed:.2f}[sec]")
        return retval

    return wrapper
