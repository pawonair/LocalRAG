# LocalRAG User Guide

> Complete guide for installing, configuring, and using LocalRAG - your local RAG system powered by Ollama and Deepseek R1.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Using the Web Interface](#using-the-web-interface)
6. [Using the REST API](#using-the-rest-api)
7. [Using the CLI](#using-the-cli)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

LocalRAG is a fully local Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents using AI. All processing happens on your machine - no data is sent to external servers.

### Key Features

- **Multi-format document support**: PDF, Word, Excel, PowerPoint, text, code, images, audio, video, and web pages
- **Advanced RAG pipeline**: Hybrid search, re-ranking, query expansion, and confidence scoring
- **Thinking model visualization**: See the AI's reasoning process with Deepseek R1
- **Persistent storage**: Documents and chat history survive restarts
- **Collection management**: Organize documents into separate collections
- **Multiple interfaces**: Web UI, REST API, and CLI for batch processing
- **Export capabilities**: Export chats to Markdown, PDF, JSON, HTML, or plain text

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime environment |
| Ollama | Latest | Local LLM inference |
| Tesseract | Latest | OCR for images (optional) |
| FFmpeg | Latest | Audio/video processing (optional) |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| Storage | 10 GB | 50+ GB (for models) |
| GPU | None (CPU works) | NVIDIA with 8+ GB VRAM |

---

## Installation

### Step 1: Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com](https://ollama.com/download)

### Step 2: Pull Required Models

```bash
# Start Ollama service
ollama serve

# In another terminal, pull the default model
ollama pull deepseek-r1:1.5b

# Optional: Pull additional models
ollama pull llama3.2:3b        # Fast chat model
ollama pull llava:7b           # Vision model for images
ollama pull nomic-embed-text   # Embedding model
```

### Step 3: Clone and Setup LocalRAG

```bash
# Clone the repository
git clone <repository-url>
cd LocalRAG

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
```

### Step 4: Install Optional Dependencies

**For OCR (image text extraction):**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

**For Audio/Video Processing:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

---

## Running the Application

### Web Interface (Streamlit)

The primary way to use LocalRAG is through the Streamlit web interface.

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start LocalRAG
cd LocalRAG
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
streamlit run src/home.py
```

The application will open in your browser at `http://localhost:8501`.

### REST API Server

Run the FastAPI server for programmatic access:

```bash
# Start the API server
cd LocalRAG
source venv/bin/activate
uvicorn src.api.routes:create_app --factory --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn src.api.routes:create_app --factory --reload --port 8000
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

### CLI Batch Processing

Process multiple documents from the command line:

```bash
cd LocalRAG
source venv/bin/activate
python3 -m src.cli.batch /path/to/documents --collection my-docs
```

---

## Using the Web Interface

### Document Upload

1. **Click the sidebar** to expand document options
2. **Drag and drop** files into the upload area, or click to browse
3. Wait for processing (chunking and embedding)
4. A success message confirms the upload

**Supported formats:**
- Documents: PDF, DOCX, XLSX, PPTX
- Text: TXT, MD, JSON, CSV, XML
- Code: PY, JS, TS, JAVA, CPP, GO, RS, and 20+ more
- Media: PNG, JPG, MP3, WAV, MP4, MOV (requires Tesseract/Whisper)
- Web: Enter a URL to fetch and process web content

### Collections

Organize documents into separate collections:

1. **Select a collection** from the dropdown in the sidebar
2. **Create new collections** using the "New Collection" expander
3. Each collection has its own vector store and search space

### Chatting with Documents

1. **Type your question** in the chat input at the bottom
2. **View the AI's reasoning** in the expandable "Thinking" section
3. **Check sources** in the "Sources" expander to verify answers
4. **Confidence indicators** show answer reliability:
   - 🟢 High confidence
   - 🟡 Medium confidence
   - 🔴 Low confidence

### Model Configuration

Access model settings in the sidebar under "Model":

**Model Selection:**
- Choose from installed Ollama models
- Models marked with ✓ are installed
- Models marked with ↓ can be pulled

**Generation Settings:**
- **Temperature**: 0.0 (precise) to 2.0 (creative)
- **Context Window**: 2K to 32K tokens
- **Presets**: Creative, Balanced, Precise

### RAG Settings

Fine-tune retrieval under "RAG Settings":

**Retrieval Mode:**
- **Hybrid** (recommended): Combines semantic + keyword search
- **Semantic**: Meaning-based search using embeddings
- **Keyword**: Exact term matching with BM25

**Query Expansion:**
- **None**: Use query as-is
- **Rewrite**: Generate alternative phrasings
- **HyDE**: Create hypothetical answer for better matching
- **Decompose**: Break complex questions into sub-queries
- **Step-back**: Generate broader context queries

**Display Settings:**
- **Show Confidence**: Display retrieval confidence scores
- **Enable Citations**: Show source references
- **Results Count**: Number of chunks to retrieve (1-10)

### Exporting Chats

1. Scroll to "Chat" section in sidebar
2. Click "Export Chat" expander
3. Select format: Markdown, JSON, HTML, Text, or PDF
4. Click "Export" then "Download"

---

## Using the REST API

### Base URL

```
http://localhost:8000/api/v1
```

### Authentication

Currently, the API does not require authentication. For production use, implement authentication in `src/api/routes.py`.

### Endpoints

#### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "version": "1.0.0",
  "timestamp": "2026-02-01T12:00:00"
}
```

#### Query Documents

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "collection_id": "default",
    "k": 3,
    "include_sources": true
  }'
```

Response:
```json
{
  "answer": "The document discusses...",
  "thinking": "Let me analyze the context...",
  "sources": [
    {
      "content": "Relevant excerpt...",
      "source": "document.pdf"
    }
  ],
  "confidence": {
    "overall_score": 0.85,
    "confidence_level": "high"
  },
  "processing_time": 1.23
}
```

#### Upload Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/document.pdf" \
  -F "collection_id=default"
```

Response:
```json
{
  "success": true,
  "document_id": "abc123",
  "filename": "document.pdf",
  "file_type": "pdf",
  "chunk_count": 42,
  "message": "Processed document.pdf"
}
```

#### List Documents

```bash
curl http://localhost:8000/api/v1/documents?collection_id=default
```

#### Delete Document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/{document_id}
```

#### List Collections

```bash
curl http://localhost:8000/api/v1/collections
```

#### Create Collection

```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Papers",
    "description": "Academic papers for research"
  }'
```

#### Chat with History

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain more about that?",
    "collection_id": "default",
    "history": [
      {"role": "user", "content": "What is RAG?"},
      {"role": "assistant", "content": "RAG stands for..."}
    ]
  }'
```

#### List Models

```bash
curl http://localhost:8000/api/v1/models
```

#### Pull Model

```bash
curl -X POST http://localhost:8000/api/v1/models/llama3.2:3b/pull
```

### API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Using the CLI

### Batch Document Processing

Process entire directories of documents from the command line.

#### Basic Usage

```bash
python3 -m src.cli.batch /path/to/documents
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--collection` | `-c` | Collection ID (default: "default") |
| `--recursive` | `-r` | Process subdirectories |
| `--types` | `-t` | File types to process |
| `--workers` | `-w` | Parallel workers (default: 4) |
| `--verbose` | `-v` | Verbose output |
| `--dry-run` | `-n` | List files without processing |
| `--output` | `-o` | Save report to JSON file |

#### Examples

**Process a directory recursively:**
```bash
python3 -m src.cli.batch /path/to/docs --recursive --collection research
```

**Preview files without processing:**
```bash
python3 -m src.cli.batch /path/to/docs --dry-run --verbose
```

**Process specific file types:**
```bash
python3 -m src.cli.batch /path/to/docs --types .pdf .docx .txt
```

**Parallel processing with report:**
```bash
python3 -m src.cli.batch /path/to/docs --workers 8 --output report.json
```

#### Output

```
Found 15 files to process
✓ document1.pdf (24 chunks, 1.23s)
✓ document2.docx (12 chunks, 0.89s)
✗ broken.pdf: Failed to parse PDF
✓ notes.txt (3 chunks, 0.12s)
...

==================================================
BATCH PROCESSING SUMMARY
==================================================
Collection: research
Total files: 15
Successful: 14
Failed: 1
Total chunks: 156
Total time: 12.45s

Failed files:
  - broken.pdf: Failed to parse PDF

Report saved to: report.json
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:1.5b

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database
DATABASE_PATH=data/localrag.db

# Vector Store
VECTOR_STORE_PATH=data/vectors

# Document Storage
DOCUMENT_STORAGE_PATH=data/documents
```

### Model Configuration

Edit default model settings in `src/home.py`:

```python
# Configuration
DEFAULT_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3
DEFAULT_COLLECTION = "default"
```

### RAG Pipeline Configuration

Adjust RAG settings in `src/rag/pipeline.py`:

```python
@dataclass
class RAGConfig:
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    semantic_weight: float = 0.7
    enable_reranking: bool = False
    rerank_top_k: int = 10
    query_expansion_mode: QueryExpansionMode = QueryExpansionMode.NONE
    enable_citations: bool = True
    show_confidence: bool = True
    retrieval_k: int = 5
```

### Webhook Configuration

Configure webhooks programmatically:

```python
from src.api.webhooks import get_webhook_manager, WebhookConfig, WebhookEvent

manager = get_webhook_manager()
manager.register(WebhookConfig(
    url="https://your-server.com/webhook",
    events=[
        WebhookEvent.DOCUMENT_UPLOADED,
        WebhookEvent.DOCUMENT_PROCESSED
    ],
    secret="your-secret-key",
    retry_count=3,
    timeout=10.0
))
```

---

## Troubleshooting

### Ollama Connection Issues

**Problem:** "Ollama Disconnected" status

**Solutions:**
1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```
2. Check Ollama is accessible:
   ```bash
   curl http://localhost:11434/api/tags
   ```
3. Verify the model is installed:
   ```bash
   ollama list
   ```

### Model Not Found

**Problem:** Model fails to load or generate

**Solutions:**
1. Pull the model:
   ```bash
   ollama pull deepseek-r1:1.5b
   ```
2. Check available models:
   ```bash
   ollama list
   ```
3. Try a smaller model if running low on memory

### Document Processing Failures

**Problem:** Documents fail to upload or process

**Solutions:**
1. Check file format is supported
2. Ensure file is not corrupted
3. For PDFs, ensure they are not password-protected
4. For images, ensure Tesseract is installed
5. Check the console for detailed error messages

### Memory Issues

**Problem:** Application crashes or runs slowly

**Solutions:**
1. Use a smaller model:
   ```bash
   ollama pull deepseek-r1:1.5b  # Instead of larger variants
   ```
2. Reduce context window in settings (2048 or 4096)
3. Process fewer documents at once
4. Increase system swap space

### OCR Not Working

**Problem:** Images upload but no text is extracted

**Solutions:**
1. Install Tesseract:
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu
   sudo apt install tesseract-ocr
   ```
2. Verify installation:
   ```bash
   tesseract --version
   ```

### Audio/Video Processing Fails

**Problem:** Media files fail to process

**Solutions:**
1. Install FFmpeg:
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu
   sudo apt install ffmpeg
   ```
2. Install Whisper model (downloads automatically on first use)
3. Ensure sufficient disk space for temporary files

### Vector Store Errors

**Problem:** Search returns no results or errors

**Solutions:**
1. Delete and rebuild the vector store:
   ```bash
   rm -rf data/vectors/*
   ```
2. Re-upload documents
3. Check embedding model is available

### API Server Issues

**Problem:** API endpoints return errors

**Solutions:**
1. Ensure Ollama is running
2. Check the server logs for detailed errors
3. Verify request format matches API documentation
4. Test with the Swagger UI at `/docs`

---

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check TODO.md for implementation details
- **Logs**: Check console output for detailed error messages

---

## Quick Reference

### Start Everything

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Web UI
cd LocalRAG && source venv/bin/activate
streamlit run src/home.py

# Terminal 3 (optional): Start API Server
cd LocalRAG && source venv/bin/activate
uvicorn src.api.routes:create_app --factory --port 8000
```

### Key URLs

| Service | URL |
|---------|-----|
| Web Interface | http://localhost:8501 |
| API Server | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |

### Default Credentials

| Setting | Default Value |
|---------|---------------|
| Model | deepseek-r1:1.5b |
| Embedding | all-MiniLM-L6-v2 |
| Collection | default |
| Temperature | 0.7 |
| Context Window | 4096 |

---

*Last Updated: February 1, 2026*
