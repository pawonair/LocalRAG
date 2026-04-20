"""
Base Loader Module
Abstract base class for all document loaders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Document:
    """Represents a document chunk with content and metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LoaderResult:
    """Result from a document loader."""
    success: bool
    documents: List[Document] = field(default_factory=list)
    error: Optional[str] = None
    file_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    # Supported file extensions for this loader
    SUPPORTED_EXTENSIONS: List[str] = []

    @classmethod
    def can_handle(cls, file_path: str) -> bool:
        """Check if this loader can handle the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @abstractmethod
    def load(self, file_path: str) -> LoaderResult:
        """
        Load a document from the given file path.

        Args:
            file_path: Path to the file to load

        Returns:
            LoaderResult with documents or error
        """
        pass

    @abstractmethod
    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """
        Load a document from bytes content.

        Args:
            content: File content as bytes
            filename: Original filename for metadata

        Returns:
            LoaderResult with documents or error
        """
        pass

    def _create_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Create standard metadata for a document."""
        path = Path(file_path)
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
        }
        metadata.update(kwargs)
        return metadata
