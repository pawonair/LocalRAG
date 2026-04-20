"""
Database Module
Provides persistent storage for documents and collections.
"""

from .database import Database, get_database
from .models import DocumentRecord, CollectionRecord

__all__ = [
    "Database",
    "get_database",
    "DocumentRecord",
    "CollectionRecord",
]
