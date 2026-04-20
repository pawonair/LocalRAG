"""
Document Router Module
Detects file types and routes to appropriate loaders.
"""

import mimetypes
from pathlib import Path
from typing import Optional, Dict, Type, List
from dataclasses import dataclass, field

from .base import BaseLoader, LoaderResult, Document


# File extension to MIME type mapping for common types
EXTENSION_MIME_MAP = {
    # Documents
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    # Data formats
    ".json": "application/json",
    ".csv": "text/csv",
    ".xml": "application/xml",
    # Office documents
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    # Code files
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".jsx": "text/javascript",
    ".tsx": "text/typescript",
    ".java": "text/x-java",
    ".cpp": "text/x-c++",
    ".c": "text/x-c",
    ".h": "text/x-c",
    ".hpp": "text/x-c++",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".php": "text/x-php",
    ".swift": "text/x-swift",
    ".kt": "text/x-kotlin",
    ".scala": "text/x-scala",
    ".r": "text/x-r",
    ".sql": "text/x-sql",
    ".sh": "text/x-shellscript",
    ".bash": "text/x-shellscript",
    ".zsh": "text/x-shellscript",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".conf": "text/plain",
    # Web
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".wma": "audio/x-ms-wma",
    ".aac": "audio/aac",
    # Video
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
}

# Categories for file types
FILE_CATEGORIES = {
    "pdf": [".pdf"],
    "text": [".txt"],
    "markdown": [".md", ".markdown"],
    "json": [".json"],
    "csv": [".csv"],
    "xml": [".xml"],
    "word": [".docx", ".doc"],
    "excel": [".xlsx", ".xls"],
    "powerpoint": [".pptx", ".ppt"],
    "code": [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r", ".sql",
        ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".html", ".htm", ".css"
    ],
    # Media types
    "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"],
    "video": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv"],
}


@dataclass
class FileInfo:
    """Information about a detected file."""
    path: str
    filename: str
    extension: str
    mime_type: str
    category: str
    size_bytes: int = 0


class DocumentRouter:
    """
    Routes documents to appropriate loaders based on file type.
    """

    def __init__(self):
        self._loaders: Dict[str, BaseLoader] = {}
        self._register_default_loaders()

    def _register_default_loaders(self):
        """Register all default loaders."""
        # Import loaders here to avoid circular imports
        from .pdf import PDFLoader
        from .text import TextLoader, MarkdownLoader, JSONLoader, CSVLoader, XMLLoader
        from .office import WordLoader, ExcelLoader, PowerPointLoader
        from .code import CodeLoader
        from .media import ImageLoader, AudioLoader, VideoLoader

        # Register loaders by category
        self._loaders["pdf"] = PDFLoader()
        self._loaders["text"] = TextLoader()
        self._loaders["markdown"] = MarkdownLoader()
        self._loaders["json"] = JSONLoader()
        self._loaders["csv"] = CSVLoader()
        self._loaders["xml"] = XMLLoader()
        self._loaders["word"] = WordLoader()
        self._loaders["excel"] = ExcelLoader()
        self._loaders["powerpoint"] = PowerPointLoader()
        self._loaders["code"] = CodeLoader()

        # Media loaders
        self._loaders["image"] = ImageLoader()
        self._loaders["audio"] = AudioLoader()
        self._loaders["video"] = VideoLoader()

    def detect_file_type(self, file_path: str) -> FileInfo:
        """
        Detect the type of a file.

        Args:
            file_path: Path to the file

        Returns:
            FileInfo with detected type information
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Get MIME type
        mime_type = EXTENSION_MIME_MAP.get(extension)
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"

        # Determine category
        category = "unknown"
        for cat, extensions in FILE_CATEGORIES.items():
            if extension in extensions:
                category = cat
                break

        # Get file size
        size_bytes = 0
        if path.exists():
            size_bytes = path.stat().st_size

        return FileInfo(
            path=str(path),
            filename=path.name,
            extension=extension,
            mime_type=mime_type,
            category=category,
            size_bytes=size_bytes,
        )

    def get_loader(self, file_path: str) -> Optional[BaseLoader]:
        """
        Get the appropriate loader for a file.

        Args:
            file_path: Path to the file

        Returns:
            Loader instance or None if no suitable loader
        """
        file_info = self.detect_file_type(file_path)
        return self._loaders.get(file_info.category)

    def load(self, file_path: str) -> LoaderResult:
        """
        Load a document using the appropriate loader.

        Args:
            file_path: Path to the file

        Returns:
            LoaderResult with documents or error
        """
        file_info = self.detect_file_type(file_path)
        loader = self._loaders.get(file_info.category)

        if not loader:
            return LoaderResult(
                success=False,
                error=f"No loader available for file type: {file_info.extension}",
                file_type=file_info.category,
            )

        result = loader.load(file_path)
        result.file_type = file_info.category
        result.metadata["file_info"] = {
            "filename": file_info.filename,
            "extension": file_info.extension,
            "mime_type": file_info.mime_type,
            "category": file_info.category,
            "size_bytes": file_info.size_bytes,
        }

        return result

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """
        Load a document from bytes content.

        Args:
            content: File content as bytes
            filename: Original filename

        Returns:
            LoaderResult with documents or error
        """
        file_info = self.detect_file_type(filename)
        loader = self._loaders.get(file_info.category)

        if not loader:
            return LoaderResult(
                success=False,
                error=f"No loader available for file type: {file_info.extension}",
                file_type=file_info.category,
            )

        result = loader.load_from_bytes(content, filename)
        result.file_type = file_info.category
        result.metadata["file_info"] = {
            "filename": file_info.filename,
            "extension": file_info.extension,
            "mime_type": file_info.mime_type,
            "category": file_info.category,
            "size_bytes": len(content),
        }

        return result

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        extensions = []
        for exts in FILE_CATEGORIES.values():
            extensions.extend(exts)
        return sorted(set(extensions))

    def is_supported(self, file_path: str) -> bool:
        """Check if a file type is supported."""
        file_info = self.detect_file_type(file_path)
        return file_info.category != "unknown"
