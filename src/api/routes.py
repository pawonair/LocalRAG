"""
API Routes
FastAPI routes for LocalRAG REST API.
"""

import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QueryRequest, QueryResponse,
    DocumentUploadResponse, DocumentInfo, DocumentListResponse,
    CollectionRequest, CollectionResponse, CollectionListResponse,
    ChatRequest, ChatResponse, ChatMessage, ChatRole,
    ModelInfo, ModelListResponse,
    HealthResponse,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_database
from rag.vectorstore import VectorStoreManager
from rag.pipeline import AdvancedRAGPipeline, RAGConfig
from llm.ollama import OllamaClient
from llm.models import ModelRegistry, ModelCapability
from loaders.router import DocumentRouter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document as LangchainDocument

router = APIRouter(prefix="/api/v1", tags=["LocalRAG API"])

_db = None
_vector_manager = None
_rag_pipeline = None
_ollama_client = None
_model_registry = None
_document_router = None


def get_db():
    global _db
    if _db is None:
        _db = get_database()
    return _db


def get_vector_manager():
    global _vector_manager
    if _vector_manager is None:
        _vector_manager = VectorStoreManager()
    return _vector_manager


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        vm = get_vector_manager()
        client = get_ollama_client()
        _rag_pipeline = AdvancedRAGPipeline(
            vector_store_manager=vm,
            llm_func=lambda p: client.generate(p),
            config=RAGConfig()
        )
    return _rag_pipeline


def get_model_registry():
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_document_router():
    global _document_router
    if _document_router is None:
        _document_router = DocumentRouter()
    return _document_router


@router.get("/health", response_model=HealthResponse)
async def health_check():
    client = get_ollama_client()
    return HealthResponse(
        status="healthy",
        ollama_connected=client.is_connected(),
        version="1.0.0",
        timestamp=datetime.now()
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    start_time = time.time()
    pipeline = get_rag_pipeline()
    client = get_ollama_client()
    collection_id = request.collection_id or "default"

    rag_result = pipeline.retrieve(
        collection_id=collection_id,
        query=request.query,
        retrieval_k=request.k
    )

    if not rag_result.final_documents:
        raise HTTPException(status_code=404, detail="No documents found")

    context = rag_result.get_context()
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {request.query}\n\nANSWER:"

    response = client.generate(prompt)

    thinking = None
    answer = response
    if "<think>" in response and "</think>" in response:
        import re
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    sources = []
    if request.include_sources:
        for doc in rag_result.final_documents:
            sources.append({
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", "Unknown"),
            })

    return QueryResponse(
        answer=answer,
        thinking=thinking,
        sources=sources,
        confidence=rag_result.confidence_metrics,
        query=request.query,
        collection_id=collection_id,
        processing_time=time.time() - start_time
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Query(None)
):
    db = get_db()
    vm = get_vector_manager()
    doc_router = get_document_router()
    pipeline = get_rag_pipeline()
    collection = collection_id or "default"

    try:
        content = await file.read()
        filename = file.filename
        result = doc_router.load_from_bytes(content, filename)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        langchain_docs = [
            LangchainDocument(page_content=d.page_content, metadata=d.metadata)
            for d in result.documents
        ]

        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        splitter = SemanticChunker(embedder)
        documents = splitter.split_documents(langchain_docs)

        vm.add_documents(collection, documents)
        pipeline.add_to_bm25_index(collection, documents)

        doc_record = db.add_document(
            filename=filename,
            file_type=result.file_type,
            content=content,
            collection_id=collection,
            chunk_count=len(documents),
            metadata=result.metadata
        )

        return DocumentUploadResponse(
            success=True,
            document_id=doc_record.id,
            filename=filename,
            file_type=result.file_type,
            chunk_count=len(documents),
            message=f"Processed {filename}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(collection_id: Optional[str] = Query(None)):
    db = get_db()
    docs = db.list_documents(collection_id=collection_id)
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d.id, filename=d.filename, file_type=d.file_type,
                chunk_count=d.chunk_count, collection_id=d.collection_id,
                uploaded_at=d.uploaded_at
            ) for d in docs
        ],
        total=len(docs),
        collection_id=collection_id
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    db = get_db()
    vm = get_vector_manager()
    pipeline = get_rag_pipeline()

    docs = db.list_documents()
    doc = next((d for d in docs if d.id == document_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    collection_id = doc.collection_id
    db.delete_document(document_id)

    remaining = db.list_documents(collection_id=collection_id)
    if not remaining:
        vm.delete_store(collection_id)
        pipeline.clear_bm25_index(collection_id)

    return {"success": True, "message": f"Deleted {doc.filename}"}


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    db = get_db()
    collections = db.list_collections()
    return CollectionListResponse(
        collections=[
            CollectionResponse(
                id=c.id, name=c.name, description=c.description,
                document_count=len(db.list_documents(collection_id=c.id)),
                created_at=c.created_at
            ) for c in collections
        ],
        total=len(collections)
    )


@router.post("/collections", response_model=CollectionResponse)
async def create_collection(request: CollectionRequest):
    db = get_db()
    c = db.create_collection(request.name, request.description)
    return CollectionResponse(
        id=c.id, name=c.name, description=c.description,
        document_count=0, created_at=c.created_at
    )


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    db = get_db()
    vm = get_vector_manager()
    pipeline = get_rag_pipeline()

    docs = db.list_documents(collection_id=collection_id)
    for d in docs:
        db.delete_document(d.id)

    db.delete_collection(collection_id)
    vm.delete_store(collection_id)
    pipeline.clear_bm25_index(collection_id)

    return {"success": True, "message": f"Deleted collection"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    pipeline = get_rag_pipeline()
    client = get_ollama_client()
    collection_id = request.collection_id or "default"

    rag_result = pipeline.retrieve(
        collection_id=collection_id,
        query=request.message,
        retrieval_k=3
    )

    context = rag_result.get_context() if rag_result.final_documents else ""

    history_text = ""
    if request.history:
        for msg in request.history[-6:]:
            role = "User" if msg.role == ChatRole.USER else "Assistant"
            history_text += f"{role}: {msg.content}\n\n"

    if context:
        prompt = f"CONTEXT:\n{context}\n\nHISTORY:\n{history_text}\n\nUser: {request.message}\n\nAssistant:"
    else:
        prompt = f"HISTORY:\n{history_text}\n\nUser: {request.message}\n\nAssistant:"

    response = client.generate(prompt)

    thinking = None
    answer = response
    if "<think>" in response and "</think>" in response:
        import re
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    sources = []
    for doc in rag_result.final_documents[:3]:
        sources.append({
            "content": doc.page_content[:300],
            "source": doc.metadata.get("source", "Unknown"),
        })

    return ChatResponse(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content=answer,
            timestamp=datetime.now(),
            thinking=thinking,
            sources=sources
        ),
        sources=sources,
        confidence=rag_result.confidence_metrics
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    client = get_ollama_client()
    registry = get_model_registry()

    available = client.list_models()
    available_names = [m.name for m in available]

    models = []
    for m in available:
        info = registry.get_model(m.name)
        caps = []
        if info:
            caps = [c.name.lower() for c in info.capabilities]
        models.append(ModelInfo(
            name=m.name,
            display_name=info.display_name if info else m.name,
            category=info.category.value if info else "unknown",
            parameter_size=info.parameter_size if info else m.parameter_size,
            is_available=True,
            capabilities=caps
        ))

    return ModelListResponse(
        models=models,
        current_model=client.config.model
    )


@router.post("/models/{model_name}/pull")
async def pull_model(model_name: str):
    client = get_ollama_client()
    success = client.pull_model(model_name)
    if success:
        return {"success": True, "message": f"Pulled {model_name}"}
    raise HTTPException(status_code=500, detail="Failed to pull model")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LocalRAG API",
        description="REST API for LocalRAG - Local RAG System",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/")
    async def root():
        return {"message": "LocalRAG API", "docs": "/docs"}

    return app
