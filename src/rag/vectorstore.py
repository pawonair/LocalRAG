"""
Vector Store Manager
Handles FAISS vector store persistence and operations.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pickle
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_VECTORS_DIR = DEFAULT_DATA_DIR / "vectors"


class VectorStoreManager:
    """
    Manages FAISS vector stores with persistence support.
    Supports multiple collections with separate indices.
    """

    def __init__(
        self,
        vectors_dir: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store manager.

        Args:
            vectors_dir: Directory for vector store files
            embedding_model: HuggingFace embedding model name
        """
        self.vectors_dir = Path(vectors_dir) if vectors_dir else DEFAULT_VECTORS_DIR
        self.embedding_model = embedding_model
        self._embedder = None
        self._stores: Dict[str, FAISS] = {}

        # Ensure directory exists
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedder(self) -> HuggingFaceEmbeddings:
        """Lazy load embedder."""
        if self._embedder is None:
            self._embedder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return self._embedder

    def _get_index_path(self, collection_id: str) -> Path:
        """Get path for a collection's index files."""
        return self.vectors_dir / collection_id

    def _get_metadata_path(self, collection_id: str) -> Path:
        """Get path for collection metadata."""
        return self.vectors_dir / f"{collection_id}_metadata.json"

    # ==================== Core Operations ====================

    def create_store(
        self,
        collection_id: str,
        documents: List[Document]
    ) -> FAISS:
        """
        Create a new vector store from documents.

        Args:
            collection_id: Collection identifier
            documents: List of LangChain documents

        Returns:
            FAISS vector store
        """
        store = FAISS.from_documents(documents, self.embedder)
        self._stores[collection_id] = store

        # Save immediately
        self.save_store(collection_id)

        return store

    def add_documents(
        self,
        collection_id: str,
        documents: List[Document]
    ) -> bool:
        """
        Add documents to an existing store.

        Args:
            collection_id: Collection identifier
            documents: Documents to add

        Returns:
            True if successful
        """
        # Load store if not in memory
        if collection_id not in self._stores:
            store = self.load_store(collection_id)
            if store is None:
                # Create new store
                self.create_store(collection_id, documents)
                return True
        else:
            store = self._stores[collection_id]

        # Add documents
        store.add_documents(documents)
        self._stores[collection_id] = store

        # Save
        self.save_store(collection_id)

        return True

    def get_store(self, collection_id: str) -> Optional[FAISS]:
        """
        Get a vector store, loading from disk if necessary.

        Args:
            collection_id: Collection identifier

        Returns:
            FAISS store or None
        """
        if collection_id in self._stores:
            return self._stores[collection_id]

        return self.load_store(collection_id)

    def save_store(self, collection_id: str) -> bool:
        """
        Save a vector store to disk.

        Args:
            collection_id: Collection identifier

        Returns:
            True if successful
        """
        if collection_id not in self._stores:
            return False

        store = self._stores[collection_id]
        index_path = self._get_index_path(collection_id)

        try:
            store.save_local(str(index_path))

            # Save metadata
            metadata = {
                "collection_id": collection_id,
                "embedding_model": self.embedding_model,
                "document_count": len(store.docstore._dict) if hasattr(store.docstore, '_dict') else 0,
            }

            with open(self._get_metadata_path(collection_id), "w") as f:
                json.dump(metadata, f)

            return True

        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False

    def load_store(self, collection_id: str) -> Optional[FAISS]:
        """
        Load a vector store from disk.

        Args:
            collection_id: Collection identifier

        Returns:
            FAISS store or None if not found
        """
        index_path = self._get_index_path(collection_id)

        if not index_path.exists():
            return None

        try:
            store = FAISS.load_local(
                str(index_path),
                self.embedder,
                allow_dangerous_deserialization=True
            )
            self._stores[collection_id] = store
            return store

        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None

    def delete_store(self, collection_id: str) -> bool:
        """
        Delete a vector store.

        Args:
            collection_id: Collection identifier

        Returns:
            True if successful
        """
        # Remove from memory
        if collection_id in self._stores:
            del self._stores[collection_id]

        # Remove from disk
        index_path = self._get_index_path(collection_id)
        metadata_path = self._get_metadata_path(collection_id)

        try:
            if index_path.exists():
                # FAISS saves as directory
                import shutil
                shutil.rmtree(index_path)

            if metadata_path.exists():
                metadata_path.unlink()

            return True

        except Exception as e:
            print(f"Error deleting vector store: {e}")
            return False

    # ==================== Search Operations ====================

    def search(
        self,
        collection_id: str,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            collection_id: Collection to search
            query: Search query
            k: Number of results
            filter_dict: Metadata filter (optional)

        Returns:
            List of matching documents
        """
        store = self.get_store(collection_id)

        if store is None:
            return []

        if filter_dict:
            return store.similarity_search(query, k=k, filter=filter_dict)
        else:
            return store.similarity_search(query, k=k)

    def search_with_scores(
        self,
        collection_id: str,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        Search with relevance scores.

        Args:
            collection_id: Collection to search
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        store = self.get_store(collection_id)

        if store is None:
            return []

        return store.similarity_search_with_score(query, k=k)

    def search_all_collections(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        """
        Search across all loaded collections.

        Args:
            query: Search query
            k: Number of results per collection

        Returns:
            Combined list of matching documents
        """
        all_results = []

        # Search loaded stores
        for collection_id in self._stores:
            results = self.search(collection_id, query, k=k)
            all_results.extend(results)

        # Also check for unloaded stores on disk
        for path in self.vectors_dir.iterdir():
            if path.is_dir() and path.name not in self._stores:
                store = self.load_store(path.name)
                if store:
                    results = self.search(path.name, query, k=k)
                    all_results.extend(results)

        return all_results

    # ==================== Utility Methods ====================

    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = set(self._stores.keys())

        # Add collections on disk
        for path in self.vectors_dir.iterdir():
            if path.is_dir():
                collections.add(path.name)

        return sorted(collections)

    def get_collection_info(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""
        metadata_path = self._get_metadata_path(collection_id)

        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)

        return None

    def merge_stores(
        self,
        target_collection_id: str,
        source_collection_ids: List[str]
    ) -> bool:
        """
        Merge multiple stores into one.

        Args:
            target_collection_id: Target collection
            source_collection_ids: Source collections to merge

        Returns:
            True if successful
        """
        target_store = self.get_store(target_collection_id)

        for source_id in source_collection_ids:
            if source_id == target_collection_id:
                continue

            source_store = self.get_store(source_id)
            if source_store:
                target_store.merge_from(source_store)

        self._stores[target_collection_id] = target_store
        self.save_store(target_collection_id)

        return True

    def clear_memory(self):
        """Clear all stores from memory (doesn't delete from disk)."""
        self._stores.clear()

    def get_document_count(self, collection_id: str) -> int:
        """Get document count for a collection."""
        store = self.get_store(collection_id)
        if store and hasattr(store.docstore, '_dict'):
            return len(store.docstore._dict)
        return 0
