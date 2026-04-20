"""
Database Module
SQLite database for document and collection management.
"""

import sqlite3
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
from contextlib import contextmanager

from .models import DocumentRecord, CollectionRecord


# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class Database:
    """
    SQLite database for persistent document storage.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize database.

        Args:
            data_dir: Directory for data storage (default: project/data)
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.db_path = self.data_dir / "localrag.db"
        self.documents_dir = self.data_dir / "documents"
        self.vectors_dir = self.data_dir / "vectors"

        # Ensure directories exist
        self._ensure_directories()

        # Initialize database
        self._init_db()

    def _ensure_directories(self):
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Collections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    collection_id TEXT,
                    uploaded_at TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (collection_id) REFERENCES collections(id)
                )
            """)

            # Create default collection if not exists
            cursor.execute("SELECT COUNT(*) FROM collections WHERE id = 'default'")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO collections (id, name, description, created_at, updated_at)
                    VALUES ('default', 'Default', 'Default document collection', ?, ?)
                """, (datetime.now().isoformat(), datetime.now().isoformat()))

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection as context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== Collection Operations ====================

    def create_collection(self, name: str, description: str = "") -> CollectionRecord:
        """Create a new collection."""
        collection_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO collections (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (collection_id, name, description, now.isoformat(), now.isoformat()))
            conn.commit()

        return CollectionRecord(
            id=collection_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )

    def get_collection(self, collection_id: str) -> Optional[CollectionRecord]:
        """Get a collection by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
            row = cursor.fetchone()

            if row:
                # Get document count
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection_id = ?",
                    (collection_id,)
                )
                doc_count = cursor.fetchone()[0]

                return CollectionRecord(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    document_count=doc_count,
                )

        return None

    def list_collections(self) -> List[CollectionRecord]:
        """List all collections."""
        collections = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM collections ORDER BY created_at DESC")

            for row in cursor.fetchall():
                # Get document count
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection_id = ?",
                    (row["id"],)
                )
                doc_count = cursor.fetchone()[0]

                collections.append(CollectionRecord(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    document_count=doc_count,
                ))

        return collections

    def update_collection(self, collection_id: str, name: str = None, description: str = None) -> bool:
        """Update a collection."""
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(collection_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE collections SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_collection(self, collection_id: str, delete_documents: bool = False) -> bool:
        """
        Delete a collection.

        Args:
            collection_id: Collection ID
            delete_documents: If True, also delete all documents in collection
        """
        if collection_id == "default":
            return False  # Cannot delete default collection

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if delete_documents:
                # Get documents in collection
                cursor.execute(
                    "SELECT id, file_path FROM documents WHERE collection_id = ?",
                    (collection_id,)
                )
                for row in cursor.fetchall():
                    # Delete file
                    file_path = Path(row["file_path"])
                    if file_path.exists():
                        file_path.unlink()

                # Delete documents from database
                cursor.execute(
                    "DELETE FROM documents WHERE collection_id = ?",
                    (collection_id,)
                )
            else:
                # Move documents to default collection
                cursor.execute(
                    "UPDATE documents SET collection_id = 'default' WHERE collection_id = ?",
                    (collection_id,)
                )

            # Delete collection
            cursor.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            conn.commit()

            return cursor.rowcount > 0

    # ==================== Document Operations ====================

    def add_document(
        self,
        filename: str,
        file_type: str,
        content: bytes,
        collection_id: str = "default",
        chunk_count: int = 0,
        metadata: Dict[str, Any] = None
    ) -> DocumentRecord:
        """
        Add a document to the database.

        Args:
            filename: Original filename
            file_type: Document type (pdf, image, etc.)
            content: File content as bytes
            collection_id: Collection to add to
            chunk_count: Number of chunks created
            metadata: Additional metadata
        """
        doc_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        # Save file
        file_ext = Path(filename).suffix
        stored_filename = f"{doc_id}{file_ext}"
        file_path = self.documents_dir / stored_filename

        with open(file_path, "wb") as f:
            f.write(content)

        # Create database record
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents
                (id, filename, file_type, file_path, collection_id, uploaded_at, chunk_count, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                filename,
                file_type,
                str(file_path),
                collection_id,
                now.isoformat(),
                chunk_count,
                len(content),
                json.dumps(metadata or {}),
            ))
            conn.commit()

        return DocumentRecord(
            id=doc_id,
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            collection_id=collection_id,
            uploaded_at=now,
            chunk_count=chunk_count,
            file_size=len(content),
            metadata=metadata or {},
        )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get a document by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()

            if row:
                return DocumentRecord.from_dict(dict(row))

        return None

    def list_documents(
        self,
        collection_id: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentRecord]:
        """
        List documents with optional filtering.

        Args:
            collection_id: Filter by collection
            file_type: Filter by file type
            limit: Maximum documents to return
            offset: Offset for pagination
        """
        documents = []
        conditions = []
        params = []

        if collection_id:
            conditions.append("collection_id = ?")
            params.append(collection_id)
        if file_type:
            conditions.append("file_type = ?")
            params.append(file_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM documents
                {where_clause}
                ORDER BY uploaded_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])

            for row in cursor.fetchall():
                documents.append(DocumentRecord.from_dict(dict(row)))

        return documents

    def update_document(
        self,
        doc_id: str,
        collection_id: str = None,
        chunk_count: int = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Update a document."""
        updates = []
        params = []

        if collection_id is not None:
            updates.append("collection_id = ?")
            params.append(collection_id)
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        params.append(doc_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get file path
            cursor.execute("SELECT file_path FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()

            if not row:
                return False

            # Delete file
            file_path = Path(row["file_path"])
            if file_path.exists():
                file_path.unlink()

            # Delete from database
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

            return cursor.rowcount > 0

    def search_documents(self, query: str) -> List[DocumentRecord]:
        """Search documents by filename."""
        documents = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM documents
                WHERE filename LIKE ?
                ORDER BY uploaded_at DESC
            """, (f"%{query}%",))

            for row in cursor.fetchall():
                documents.append(DocumentRecord.from_dict(dict(row)))

        return documents

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            # Total collections
            cursor.execute("SELECT COUNT(*) FROM collections")
            total_collections = cursor.fetchone()[0]

            # Documents by type
            cursor.execute("""
                SELECT file_type, COUNT(*) as count
                FROM documents
                GROUP BY file_type
            """)
            by_type = {row["file_type"]: row["count"] for row in cursor.fetchall()}

            # Total size
            cursor.execute("SELECT SUM(file_size) FROM documents")
            total_size = cursor.fetchone()[0] or 0

            return {
                "total_documents": total_docs,
                "total_collections": total_collections,
                "documents_by_type": by_type,
                "total_size_bytes": total_size,
            }


# Singleton instance
_database_instance: Optional[Database] = None


def get_database(data_dir: Optional[Path] = None) -> Database:
    """Get or create the database singleton."""
    global _database_instance

    if _database_instance is None:
        _database_instance = Database(data_dir)

    return _database_instance
