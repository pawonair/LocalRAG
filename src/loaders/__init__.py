"""
Document Loaders Module
Provides unified document loading for various file formats including media.
"""

from .router import DocumentRouter, LoaderResult
from .base import Document, BaseLoader
from .pdf import PDFLoader
from .text import TextLoader, MarkdownLoader, JSONLoader, CSVLoader, XMLLoader
from .office import WordLoader, ExcelLoader, PowerPointLoader
from .code import CodeLoader
from .web import WebLoader
from .media import ImageLoader, AudioLoader, VideoLoader, get_media_loaders

__all__ = [
    # Core
    "DocumentRouter",
    "LoaderResult",
    "Document",
    "BaseLoader",
    # Document loaders
    "PDFLoader",
    "TextLoader",
    "MarkdownLoader",
    "JSONLoader",
    "CSVLoader",
    "XMLLoader",
    "WordLoader",
    "ExcelLoader",
    "PowerPointLoader",
    "CodeLoader",
    "WebLoader",
    # Media loaders
    "ImageLoader",
    "AudioLoader",
    "VideoLoader",
    "get_media_loaders",
]
