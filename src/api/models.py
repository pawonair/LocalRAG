"""
API Models
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ChatRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message model."""
    role: ChatRole
    content: str
    timestamp: Optional[datetime] = None
    thinking: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    """Request for querying documents."""
    query: str = Field(..., min_length=1, description="The question to ask")
    collection_id: Optional[str] = Field(None, description="Collection to search")
    k: int = Field(3, ge=1, le=20, description="Number of results")
    include_sources: bool = Field(True, description="Include source documents")
    stream: bool = Field(False, description="Stream the response")


class QueryResponse(BaseModel):
    """Response from query."""
    answer: str
    thinking: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    confidence: Optional[Dict[str, Any]] = None
    query: str
    collection_id: Optional[str] = None
    processing_time: float = 0.0


class DocumentUploadRequest(BaseModel):
    """Request for document upload metadata."""
    collection_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """Response from document upload."""
    success: bool
    document_id: Optional[str] = None
    filename: str
    file_type: str
    chunk_count: int = 0
    message: str = ""


class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    filename: str
    file_type: str
    chunk_count: int
    collection_id: str
    uploaded_at: Optional[datetime] = None


class CollectionRequest(BaseModel):
    """Request for creating a collection."""
    name: str = Field(..., min_length=1)
    description: Optional[str] = None


class CollectionResponse(BaseModel):
    """Collection information."""
    id: str
    name: str
    description: Optional[str] = None
    document_count: int = 0
    created_at: Optional[datetime] = None


class CollectionListResponse(BaseModel):
    """List of collections."""
    collections: List[CollectionResponse]
    total: int


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentInfo]
    total: int
    collection_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for chat."""
    message: str = Field(..., min_length=1)
    collection_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None
    stream: bool = False


class ChatResponse(BaseModel):
    """Response from chat."""
    message: ChatMessage
    sources: List[Dict[str, Any]] = []
    confidence: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    display_name: str
    category: str
    parameter_size: str
    is_available: bool = False
    capabilities: List[str] = []


class ModelListResponse(BaseModel):
    """List of models."""
    models: List[ModelInfo]
    current_model: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    version: str = "1.0.0"
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
