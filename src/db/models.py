"""
Database Models
Data classes for documents and collections.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


@dataclass
class CollectionRecord:
    """Represents a document collection."""
    id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    document_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "document_count": self.document_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.now()),
            document_count=data.get("document_count", 0),
        )


@dataclass
class DocumentRecord:
    """Represents a document in the database."""
    id: str
    filename: str
    file_type: str
    file_path: str
    collection_id: Optional[str] = None
    uploaded_at: datetime = field(default_factory=datetime.now)
    chunk_count: int = 0
    file_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "collection_id": self.collection_id,
            "uploaded_at": self.uploaded_at.isoformat(),
            "chunk_count": self.chunk_count,
            "file_size": self.file_size,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """Create from dictionary."""
        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return cls(
            id=data["id"],
            filename=data["filename"],
            file_type=data["file_type"],
            file_path=data["file_path"],
            collection_id=data.get("collection_id"),
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]) if isinstance(data.get("uploaded_at"), str) else data.get("uploaded_at", datetime.now()),
            chunk_count=data.get("chunk_count", 0),
            file_size=data.get("file_size", 0),
            metadata=metadata,
        )
