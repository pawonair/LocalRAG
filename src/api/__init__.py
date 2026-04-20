"""
API Module
REST API for LocalRAG.
"""

from .routes import create_app, router
from .models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    CollectionResponse,
    ChatMessage,
)

__all__ = [
    "create_app",
    "router",
    "QueryRequest",
    "QueryResponse",
    "DocumentUploadResponse",
    "CollectionResponse",
    "ChatMessage",
]
