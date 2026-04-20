# LocalRAG Development Roadmap

> Local RAG System with Ollama + Deepseek | Interactive Chat Interface | Multi-Media Support

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| UI Framework | ✅ Done | Streamlit-based web interface |
| LLM Integration | ✅ Done | Ollama + Deepseek R1 1.5B |
| PDF Processing | ✅ Done | PDFPlumber extraction |
| Vector Store | ✅ Done | FAISS with HuggingFace embeddings |
| RAG Chain | ✅ Done | LangChain retrieval chain |
| Chat Interface | ✅ Done | Full chat with memory & streaming |
| Thinking Display | ✅ Done | Deepseek reasoning visualization |
| Multi-Format Support | ✅ Done | PDF, Office, text, code, web |
| Media Support | ✅ Done | Images (OCR+Vision), Audio (Whisper), Video |
| Document Management | ✅ Done | SQLite + FAISS persistence with collections |
| Advanced RAG | ✅ Done | Hybrid search, re-ranking, query expansion, citations |
| Model Management | ✅ Done | Model selector, parameter tuning, auto-pull |
| Export & Integration | ✅ Done | Chat export, REST API, webhooks, batch CLI |

---

## Phase 1: Core Chat Enhancement
**Priority:** 🔴 HIGH | **Status:** ✅ COMPLETED

### Tasks
- [x] **1.1 Conversation Memory**
  - [x] Implement session-based chat history storage
  - [x] Add context window management (token limiting)
  - [x] Create `src/memory/chat_memory.py`

- [x] **1.2 Chat UI Redesign**
  - [x] Create chat message components with bubbles
  - [x] Add user/assistant message differentiation
  - [x] Implement auto-scroll to latest message
  - [x] Create `src/components/chat.py`

- [x] **1.3 Thinking Model Display**
  - [x] Parse `<think>` tags from Deepseek responses
  - [x] Create collapsible thinking section in UI
  - [x] Add toggle for showing/hiding reasoning
  - [x] Create `src/components/thinking_display.py`

- [x] **1.4 Response Streaming**
  - [x] Implement Ollama streaming responses
  - [x] Add typing indicator during generation
  - [x] Handle stream interruption gracefully

### Files Created/Modified
```
src/
├── memory/
│   ├── __init__.py          ✅ Created
│   └── chat_memory.py       ✅ Created
├── components/
│   ├── __init__.py          ✅ Created
│   ├── chat.py              ✅ Created
│   └── thinking_display.py  ✅ Created
├── styles.py                ✅ Updated
└── home.py                  ✅ Refactored
```

---

## Phase 2: Multi-Format Document Support
**Priority:** 🔴 HIGH | **Status:** ✅ COMPLETED

### Tasks
- [x] **2.1 Document Router**
  - [x] Create unified document ingestion interface
  - [x] Implement file type detection (MIME + extension)
  - [x] Create `src/loaders/router.py`

- [x] **2.2 Text Files Support**
  - [x] Add `.txt` file loader
  - [x] Add `.md` (Markdown) loader
  - [x] Add `.json` loader with structure preservation
  - [x] Add `.csv` loader with table awareness
  - [x] Add `.xml` loader
  - [x] Create `src/loaders/text.py`

- [x] **2.3 Office Documents**
  - [x] Add `.docx` (Word) support
  - [x] Add `.xlsx` (Excel) support with sheet handling
  - [x] Add `.pptx` (PowerPoint) support with slide extraction
  - [x] Create `src/loaders/office.py`

- [x] **2.4 Code Files**
  - [x] Add syntax-aware code file processing
  - [x] Support: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`
  - [x] Preserve code structure and comments
  - [x] Create `src/loaders/code.py`

- [x] **2.5 Web Content**
  - [x] Add URL ingestion capability
  - [x] Implement web page scraping
  - [x] Create `src/loaders/web.py`

### New Dependencies Added
```txt
python-docx>=1.1.0
openpyxl>=3.1.0
python-pptx>=0.6.23
beautifulsoup4>=4.12.0
lxml>=5.0.0
```

### Files Created
```
src/loaders/
├── __init__.py       ✅ Created
├── base.py           ✅ Created (base loader class)
├── router.py         ✅ Created
├── pdf.py            ✅ Created
├── text.py           ✅ Created (txt, md, json, csv, xml)
├── office.py         ✅ Created (docx, xlsx, pptx)
├── code.py           ✅ Created (20+ languages)
└── web.py            ✅ Created
```

---

## Phase 3: Media Processing Pipeline
**Priority:** 🔴 HIGH | **Status:** ✅ COMPLETED

### Tasks
- [x] **3.1 Image Processing**
  - [x] Implement OCR with Tesseract
  - [x] Add vision model support (Ollama LLaVA)
  - [x] Handle image formats: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF
  - [x] Extract text from screenshots and scanned documents

- [x] **3.2 Audio Transcription**
  - [x] Integrate Whisper for local transcription
  - [x] Support formats: MP3, WAV, M4A, FLAC, OGG, WMA, AAC
  - [x] Handle long audio with segment chunking
  - [x] Include timestamps in transcriptions

- [x] **3.3 Video Processing**
  - [x] Implement frame extraction with OpenCV
  - [x] Extract and transcribe audio track
  - [x] Support formats: MP4, MOV, AVI, MKV, WEBM, WMV, FLV
  - [x] Generate video summaries from keyframes using vision model

- [ ] **3.4 Multimodal Embeddings** (Future enhancement)
  - [ ] Add CLIP-based image embeddings
  - [ ] Enable cross-modal search (text → image)
  - [ ] Create unified multimodal vector store

### New Dependencies Added
```txt
pytesseract>=0.3.10
opencv-python>=4.8.0
openai-whisper>=20231117
```

### Files Created
```
src/loaders/
└── media.py          ✅ Created (ImageLoader, AudioLoader, VideoLoader)
```

### Media Processing Features
| Media Type | Processing Method | Output |
|------------|-------------------|--------|
| **Images** | OCR (Tesseract) + Vision Model (LLaVA) | Text extraction + AI description |
| **Audio** | Whisper transcription | Full transcript with timestamps |
| **Video** | Frame extraction + Audio transcription | Keyframe descriptions + transcript |

---

## Phase 4: Document Management System
**Priority:** 🟡 MEDIUM | **Status:** ✅ COMPLETED

### Tasks
- [x] **4.1 Persistent Storage**
  - [x] Set up SQLite database
  - [x] Create document metadata schema
  - [x] Implement file storage organization
  - [x] Create `src/db/database.py`

- [x] **4.2 Collection Management**
  - [x] Create collection CRUD operations
  - [x] Allow grouping documents into collections
  - [x] Enable collection-scoped searches
  - [x] Add collection UI in sidebar

- [x] **4.3 Vector Store Persistence**
  - [x] Save FAISS indices to disk
  - [x] Load indices on startup
  - [x] Implement incremental index updates
  - [x] Create `src/rag/vectorstore.py`

- [x] **4.4 Document Viewer**
  - [x] Preview uploaded documents in UI
  - [x] Show document metadata
  - [x] Display chunk breakdown

- [x] **4.5 Search & Filter**
  - [x] Browse uploaded documents
  - [x] Filter by type, date, collection
  - [x] Delete documents and update indices

### Database Schema
```sql
-- Documents
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    collection_id TEXT,
    uploaded_at TIMESTAMP,
    chunk_count INTEGER
);

-- Collections
CREATE TABLE collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP
);
```

### Files Created
```
src/db/
├── __init__.py       ✅ Created
├── database.py       ✅ Created
└── models.py         ✅ Created

src/rag/
├── __init__.py       ✅ Created
└── vectorstore.py    ✅ Created

data/
├── documents/.gitkeep  ✅ Created
├── vectors/.gitkeep    ✅ Created
├── cache/.gitkeep      ✅ Created
└── localrag.db         (created at runtime)
```

---

## Phase 5: Advanced RAG Features
**Priority:** 🟡 MEDIUM | **Status:** ✅ COMPLETED

### Tasks
- [x] **5.1 Hybrid Search**
  - [x] Implement BM25 keyword search
  - [x] Combine with semantic search (FAISS)
  - [x] Add Reciprocal Rank Fusion (RRF)

- [x] **5.2 Re-ranking**
  - [x] Add cross-encoder re-ranking
  - [x] Implement FlashRank support
  - [x] Configure re-rank top-k
  - [x] LLM-based reranking fallback

- [x] **5.3 Query Expansion**
  - [x] LLM-based query rewriting
  - [x] Generate multiple sub-queries
  - [x] Implement HyDE (Hypothetical Document Embeddings)
  - [x] Step-back prompting
  - [x] Query decomposition

- [x] **5.4 Citation System**
  - [x] Show source chunks with responses
  - [x] Add numbered references
  - [x] Relevance indicators per source

- [x] **5.5 Confidence Scoring**
  - [x] Display retrieval confidence scores
  - [x] Show relevance indicators (🟢 High, 🟡 Medium, 🔴 Low)
  - [x] Warn on low-confidence answers

- [x] **5.6 Multi-Query RAG**
  - [x] Decompose complex questions
  - [x] Aggregate results from sub-queries
  - [x] Score boosting for multi-query matches

### New Dependencies Added
```txt
rank-bm25>=0.2.2
flashrank>=0.2.0
```

### Files Created
```
src/rag/
├── __init__.py       ✅ Updated (exports all components)
├── retriever.py      ✅ Created (HybridRetriever, BM25Index, MultiQueryRetriever)
├── reranker.py       ✅ Created (FlashRankReranker, LLMReranker, CohereReranker)
├── query_expansion.py ✅ Created (HyDE, QueryDecomposer, StepBackExpander)
├── citations.py      ✅ Created (CitationManager, ConfidenceScorer)
└── pipeline.py       ✅ Created (AdvancedRAGPipeline, RAGConfig)
```

### RAG Settings UI
| Setting | Options | Description |
|---------|---------|-------------|
| Retrieval Mode | Hybrid, Semantic, Keyword | Search strategy |
| Semantic Weight | 0.0 - 1.0 | Balance for hybrid mode |
| Query Expansion | None, Rewrite, HyDE, Decompose, Step-back | Query enhancement |
| Show Confidence | On/Off | Display confidence scores |
| Enable Citations | On/Off | Show source references |
| Results Count | 1-10 | Number of chunks to retrieve |

---

## Phase 6: Model Configuration & Management
**Priority:** 🟡 MEDIUM | **Status:** ✅ COMPLETED

### Tasks
- [x] **6.1 Model Selector UI**
  - [x] Dropdown to switch Ollama models
  - [x] Show model capabilities (chat/vision/embed)
  - [x] Display model size and requirements
  - [x] Model category filtering (Thinking, Chat, Vision, Code, Embedding)

- [x] **6.2 Embedding Model Config**
  - [x] Model registry with embedding models
  - [x] Support: nomic-embed-text, mxbai-embed-large, all-minilm
  - [x] Model capability metadata

- [x] **6.3 Parameter Tuning**
  - [x] Temperature slider
  - [x] Top-p / Top-k controls
  - [x] Context window size selector
  - [x] Max tokens setting
  - [x] Repeat penalty control
  - [x] Quick presets (Creative, Balanced, Precise)

- [x] **6.4 Model Status**
  - [x] Show Ollama connection status
  - [x] Display loaded/running models
  - [x] Generation statistics (tokens/sec, duration)

- [x] **6.5 Auto Model Pull**
  - [x] Detect missing models
  - [x] Prompt to pull required models
  - [x] Progress callback support

### Supported Models Registry
```python
# Thinking Models
deepseek-r1:1.5b, deepseek-r1:7b, deepseek-r1:14b, deepseek-r1:32b

# Chat Models
llama3.2:1b, llama3.2:3b, llama3.1:8b, mistral:7b
qwen2.5:7b, qwen2.5:14b, gemma2:9b, phi3:medium

# Vision Models
llava:7b, llava:13b, llava-llama3:8b, moondream:1.8b

# Code Models
codellama:7b, codellama:13b, deepseek-coder:6.7b, starcoder2:7b

# Embedding Models
nomic-embed-text, mxbai-embed-large, all-minilm, snowflake-arctic-embed
```

### Files Created
```
src/llm/
├── __init__.py       ✅ Created (module exports)
├── ollama.py         ✅ Created (OllamaClient, OllamaConfig)
├── models.py         ✅ Created (ModelRegistry, ModelInfo, capabilities)
└── prompts.py        ✅ Created (PromptTemplate, PromptManager)

src/components/
└── settings.py       ✅ Created (UI components for model settings)
```

### Model Settings UI
| Setting | Options | Description |
|---------|---------|-------------|
| Model Select | Available + Registry | Switch between installed models |
| Temperature | 0.0 - 2.0 | Creativity level |
| Context Window | 2K - 128K tokens | Maximum context length |
| Preset | Creative, Balanced, Precise | Quick parameter presets |

---

## Phase 7: Export & Integration
**Priority:** 🟢 LOW | **Status:** ✅ COMPLETED

### Tasks
- [x] **7.1 Chat Export**
  - [x] Export conversations as Markdown
  - [x] Export as PDF (using ReportLab)
  - [x] Export as JSON, HTML, plain text
  - [x] Include timestamps and metadata
  - [x] Preserve thinking process and sources

- [x] **7.2 REST API**
  - [x] Create FastAPI endpoints
  - [x] Document API with OpenAPI/Swagger
  - [x] CORS middleware support
  - [x] Endpoints: /health, /query, /documents, /collections, /chat, /models

- [x] **7.3 Webhook Support**
  - [x] Send notifications on document upload
  - [x] Notify on processing completion
  - [x] Custom webhook configuration
  - [x] Async delivery with worker thread
  - [x] HMAC signature authentication
  - [x] Retry logic with configurable attempts

- [x] **7.4 Batch Processing**
  - [x] CLI for bulk document ingestion
  - [x] Recursive directory support
  - [x] Progress reporting
  - [x] Parallel processing with configurable workers
  - [x] JSON report generation
  - [x] Dry-run mode

### New Dependencies Added
```txt
fastapi>=0.115.0
uvicorn>=0.34.0
reportlab>=4.2.0
```

### Files Created
```
src/export/
├── __init__.py       ✅ Created (module exports)
└── chat_export.py    ✅ Created (ChatExporter with 5 formats)

src/api/
├── __init__.py       ✅ Created (module exports)
├── models.py         ✅ Created (Pydantic request/response models)
├── routes.py         ✅ Created (FastAPI endpoints)
└── webhooks.py       ✅ Created (WebhookManager, async delivery)

src/cli/
├── __init__.py       ✅ Created (module exports)
└── batch.py          ✅ Created (BatchProcessor, CLI interface)
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check with Ollama status |
| `/api/v1/query` | POST | Query documents with RAG |
| `/api/v1/documents/upload` | POST | Upload and process documents |
| `/api/v1/documents` | GET | List documents in collection |
| `/api/v1/documents/{id}` | DELETE | Delete a document |
| `/api/v1/collections` | GET/POST | List or create collections |
| `/api/v1/collections/{id}` | DELETE | Delete a collection |
| `/api/v1/chat` | POST | Chat with context and history |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/{name}/pull` | POST | Pull a model from Ollama |

### Webhook Events
| Event | Description |
|-------|-------------|
| `document.uploaded` | Document received and queued |
| `document.processed` | Document chunked and indexed |
| `document.deleted` | Document removed |
| `collection.created` | New collection created |
| `collection.deleted` | Collection removed |
| `query.completed` | Query finished with results |
| `error` | Processing error occurred |

### CLI Usage
```bash
# Basic batch processing
python -m src.cli.batch /path/to/documents

# With options
python -m src.cli.batch /path/to/documents \
    --collection my-docs \
    --recursive \
    --workers 8 \
    --output report.json

# Dry run to preview files
python -m src.cli.batch /path/to/documents --dry-run --verbose
```

---

## Quick Wins (Immediate Improvements)

- [x] Add chat history using `st.session_state`
- [x] Implement response streaming with Ollama
- [x] Parse and display `<think>` tags from Deepseek
- [x] Allow multiple file uploads at once
- [x] Add file type icons in upload area
- [x] Save FAISS index to disk for persistence
- [ ] Show upload progress indicator

---

## Project Structure (Final)

```
LocalRAG/
├── src/
│   ├── __init__.py
│   ├── home.py              # Main Streamlit application
│   ├── styles.py            # UI styling
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── thinking_display.py
│   │   └── settings.py
│   │
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── router.py
│   │   ├── pdf.py
│   │   ├── text.py
│   │   ├── office.py
│   │   ├── code.py
│   │   ├── web.py
│   │   └── media.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── query_expansion.py
│   │   ├── citations.py
│   │   └── pipeline.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama.py
│   │   ├── models.py
│   │   └── prompts.py
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── models.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── chat_memory.py
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   └── chat_export.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── routes.py
│   │   └── webhooks.py
│   │
│   └── cli/
│       ├── __init__.py
│       └── batch.py
│
├── data/
│   ├── documents/
│   ├── vectors/
│   └── cache/
│
├── tests/
├── requirements.txt
├── TODO.md
└── README.md
```

---

## Notes

- **Ollama must be running** locally for the application to work
- **Deepseek R1** is the default thinking model
- **HuggingFace embeddings** are used for vector search
- **FAISS** provides fast similarity search

---

*Last Updated: January 31, 2026*
