"""
LocalRAG - Local RAG System with Ollama & Deepseek R1
Main application with interactive chat interface.
Supports multiple document formats with persistent storage.
"""

import sys
import os
import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Generator
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document as LangchainDocument

from memory.chat_memory import ChatMemory
from components.thinking_display import parse_thinking
from loaders.router import DocumentRouter

from db.database import get_database

from rag.vectorstore import VectorStoreManager
from rag.pipeline import  (
    AdvancedRAGPipeline,
    RAGConfig,
    RetrievalMode,
    QueryExpansionMode
)

from llm.ollama import OllamaClient, OllamaConfig, ConnectionStatus
from llm.models import ModelRegistry, ModelCategory

from export.chat_export import ChatExporter, ExportFormat

# Configuration
DEFAULT_MODEL = "deepseek-r1:latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3
DEFAULT_COLLECTION = "default"

# Supported file types
SUPPORTED_EXTENSIONS = [
    "pdf",
    # Text files
    "txt", "md", "markdown",
    # Data files
    "json", "csv", "xml",
    # Office documents
    "docx", "xlsx", "pptx",
    # Code files
    "py", "js", "ts", "jsx", "tsx", "java", "cpp", "c", "h", "hpp",
    "go", "rs", "rb", "php", "swift", "kt", "scala", "sql",
    "sh", "bash", "yaml", "yml", "toml", "html", "css",
    # Images
    "png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif",
    # Audio
    "mp3", "wav", "m4a", "flac", "ogg", "wma", "aac",
    # Video
    "mp4", "mov", "avi", "mkv", "webm", "wmv", "flv",
]

# File type icons
FILE_ICONS = {
    "pdf": "📄",
    "text": "📝",
    "markdown": "📑",
    "json": "🔧",
    "csv": "📊",
    "xml": "📋",
    "word": "📘",
    "excel": "📗",
    "powerpoint": "📙",
    "code": "💻",
    "web": "🌐",
    "image": "🖼️",
    "audio": "🎵",
    "video": "🎬",
    "unknown": "📎",
}


def initialize_session_state():
    """Initialize all session state variables."""
    # Database and vector store managers
    if "db" not in st.session_state:
        st.session_state.db = get_database()

    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager(
            embedding_model=EMBEDDING_MODEL
        )

    # Current collection
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = DEFAULT_COLLECTION

    # Document router
    if "document_router" not in st.session_state:
        st.session_state.document_router = DocumentRouter()

    # Processing state
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Rotating uploader key clears st.file_uploader selection after successful ingest.
    if "document_uploader_key" not in st.session_state:
        st.session_state.document_uploader_key = 0

    # Ollama client and model configuration
    if "ollama_config" not in st.session_state:
        st.session_state.ollama_config = OllamaConfig(
            model=DEFAULT_MODEL,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            num_ctx=4096
        )

    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient(
            config=st.session_state.ollama_config
        )

    if "model_registry" not in st.session_state:
        st.session_state.model_registry = ModelRegistry()

    # RAG configuration
    if "rag_config" not in st.session_state:
        st.session_state.rag_config = RAGConfig(
            retrieval_mode=RetrievalMode.HYBRID,
            enable_reranking=False,  # Disabled by default (requires FlashRank)
            query_expansion_mode=QueryExpansionMode.NONE,
            enable_citations=True,
            show_confidence=True
        )

    # Advanced RAG pipeline
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = AdvancedRAGPipeline(
            vector_store_manager=st.session_state.vector_manager,
            llm_func=_llm_query_func,
            config=st.session_state.rag_config
        )

    # Initialize chat memory
    ChatMemory.initialize()

    # Load existing documents from database
    _load_persisted_state()


def _llm_query_func(prompt: str) -> str:
    """Simple LLM query function for query expansion."""
    try:
        client = st.session_state.ollama_client
        return client.generate(prompt)
    except Exception:
        return ""


def _load_persisted_state():
    """Load persisted documents and vector store on startup."""
    if "persisted_loaded" not in st.session_state:
        st.session_state.persisted_loaded = True

        # Try to load existing vector store
        collection_id = st.session_state.current_collection
        store = st.session_state.vector_manager.get_store(collection_id)

        if store is not None:
            st.session_state.documents_loaded = True

            # Build BM25 index from existing documents
            if hasattr(store.docstore, '_dict'):
                documents = list(store.docstore._dict.values())
                if documents and "rag_pipeline" in st.session_state:
                    st.session_state.rag_pipeline.build_bm25_index(collection_id, documents)
        else:
            st.session_state.documents_loaded = False


def get_current_documents():
    """Get list of documents in current collection."""
    db = st.session_state.db
    return db.list_documents(collection_id=st.session_state.current_collection)


def process_document(uploaded_file) -> bool:
    """
    Process uploaded document, save to database, and update vector store.
    Returns True if successful, False otherwise.
    """
    try:
        router = st.session_state.document_router
        db = st.session_state.db
        vector_manager = st.session_state.vector_manager
        collection_id = st.session_state.current_collection

        filename = uploaded_file.name
        content = uploaded_file.getvalue()

        # Check if document already exists
        existing_docs = db.list_documents(collection_id=collection_id)
        for doc in existing_docs:
            if doc.filename == filename:
                st.warning(f"Document '{filename}' already exists in this collection.")
                return False

        # Load document using router
        with st.spinner(f"Loading {filename}..."):
            result = router.load_from_bytes(content, filename)

        if not result.success:
            st.error(f"Failed to load {filename}: {result.error}")
            return False

        if not result.documents:
            st.error(f"No content extracted from {filename}")
            return False

        # Convert to LangChain documents
        langchain_docs = [
            LangchainDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in result.documents
        ]

        # Split the documents into chunks
        with st.spinner("Creating semantic chunks..."):
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            text_splitter = SemanticChunker(embedder)
            documents = text_splitter.split_documents(langchain_docs)

        # Add to vector store
        with st.spinner("Creating embeddings..."):
            vector_manager.add_documents(collection_id, documents)

        # Update BM25 index for hybrid search
        rag_pipeline = st.session_state.rag_pipeline
        rag_pipeline.add_to_bm25_index(collection_id, documents)

        # Save to database
        db.add_document(
            filename=filename,
            file_type=result.file_type,
            content=content,
            collection_id=collection_id,
            chunk_count=len(documents),
            metadata=result.metadata
        )

        st.session_state.documents_loaded = True
        return True

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return False


def process_url(url: str) -> bool:
    """
    Process a URL and add to vector store and database.

    Returns True if successful, False otherwise.
    """
    try:
        from loaders.web import WebLoader

        db = st.session_state.db
        vector_manager = st.session_state.vector_manager
        collection_id = st.session_state.current_collection

        web_loader = WebLoader()

        if not web_loader.is_valid_url(url):
            st.error("Invalid URL format")
            return False

        with st.spinner(f"Fetching {url}..."):
            result = web_loader.load(url)

        if not result.success:
            st.error(f"Failed to fetch URL: {result.error}")
            return False

        if not result.documents:
            st.error("No content extracted from URL")
            return False

        # Convert to LangChain documents
        langchain_docs = [
            LangchainDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in result.documents
        ]

        # Split and embed
        with st.spinner("Processing content..."):
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            text_splitter = SemanticChunker(embedder)
            documents = text_splitter.split_documents(langchain_docs)

        with st.spinner("Creating embeddings..."):
            vector_manager.add_documents(collection_id, documents)

        # Update BM25 index for hybrid search
        rag_pipeline = st.session_state.rag_pipeline
        rag_pipeline.add_to_bm25_index(collection_id, documents)

        # Save to database
        title = result.metadata.get("title", url)
        content_bytes = "\n".join(doc.page_content for doc in result.documents).encode("utf-8")

        db.add_document(
            filename=title,
            file_type="web",
            content=content_bytes,
            collection_id=collection_id,
            chunk_count=len(documents),
            metadata=result.metadata
        )

        st.session_state.documents_loaded = True

        return True

    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return False


def delete_document(doc_id: str) -> bool:
    """Delete a document from database and rebuild vector store."""
    try:
        db = st.session_state.db
        vector_manager = st.session_state.vector_manager
        rag_pipeline = st.session_state.rag_pipeline
        collection_id = st.session_state.current_collection

        # Delete from database
        db.delete_document(doc_id)

        # Rebuild vector store from remaining documents
        remaining_docs = db.list_documents(collection_id=collection_id)

        if not remaining_docs:
            # No documents left, delete vector store and BM25 index
            vector_manager.delete_store(collection_id)
            rag_pipeline.clear_bm25_index(collection_id)
            st.session_state.documents_loaded = False
        else:
            # Rebuild vector store and BM25 index
            vector_manager.delete_store(collection_id)
            rag_pipeline.clear_bm25_index(collection_id)

            all_langchain_docs = []
            for doc_record in remaining_docs:
                file_path = Path(doc_record.file_path)
                if file_path.exists():
                    router = st.session_state.document_router
                    content = file_path.read_bytes()
                    result = router.load_from_bytes(content, doc_record.filename)

                    if result.success:
                        for doc in result.documents:
                            all_langchain_docs.append(LangchainDocument(
                                page_content=doc.page_content,
                                metadata=doc.metadata
                            ))

            if all_langchain_docs:
                embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                text_splitter = SemanticChunker(embedder)
                documents = text_splitter.split_documents(all_langchain_docs)
                vector_manager.create_store(collection_id, documents)
                rag_pipeline.build_bm25_index(collection_id, documents)

        return True

    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False


def get_relevant_context(query: str, k: int = TOP_K_RESULTS) -> tuple[str, list[dict], dict]:
    """
    Retrieve relevant document chunks for the query using advanced RAG pipeline.

    Returns tuple of (context_string, source_documents, confidence_metrics).
    """
    rag_pipeline = st.session_state.rag_pipeline
    collection_id = st.session_state.current_collection

    # Use advanced RAG pipeline
    rag_result = rag_pipeline.retrieve(
        collection_id=collection_id,
        query=query,
        retrieval_k=k
    )

    if not rag_result.final_documents:
        return "", [], {}

    # Format context
    context_parts = []
    sources = []

    for i, doc in enumerate(rag_result.final_documents):
        context_parts.append(f"[Document {i+1}]\n{doc.page_content}")

        # Find matching citation for relevance score
        relevance = 0.0
        for citation in rag_result.citations:
            if citation.document.page_content[:100] == doc.page_content[:100]:
                relevance = citation.relevance_score
                break

        sources.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", doc.metadata.get("filename", "Unknown")),
            "page": doc.metadata.get("page", doc.metadata.get("section", "N/A")),
            "relevance": relevance,
            "confidence": rag_result.confidence_metrics.get("confidence_level", "unknown")
        })

    context = "\n\n".join(context_parts)
    return context, sources, rag_result.confidence_metrics


def build_prompt(query: str, context: str, chat_history: str = "") -> str:
    """Build the prompt for the LLM."""
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Use the following context to answer the question.
2. If you don't know the answer based on the context, say "I don't have enough information to answer that."
3. Keep your answers clear and concise.
4. If relevant, mention which part of the document your answer comes from.

CONTEXT:
{context}

"""
    if chat_history:
        prompt += f"""PREVIOUS CONVERSATION:
{chat_history}

"""

    prompt += f"""QUESTION: {query}

ANSWER:"""

    return prompt


def stream_response(prompt: str) -> Generator[str, None, None]:
    """Stream response from Ollama using configured client."""
    client = st.session_state.ollama_client
    return client.generate_stream(prompt)


def handle_user_input(user_input: str):
    """Process user input and generate response."""
    ChatMemory.add_message("user", user_input)

    context, sources, confidence_metrics = get_relevant_context(user_input)

    if not context:
        response = "Please upload a document first so I can answer questions about it."
        ChatMemory.add_message("assistant", response)
        return

    chat_history = ChatMemory.get_context_string(max_pairs=3)
    prompt = build_prompt(user_input, context, chat_history)

    thinking_content = None

    with st.chat_message("assistant", avatar="🤖"):
        # Show confidence indicator if enabled
        if st.session_state.rag_config.show_confidence and confidence_metrics:
            confidence_level = confidence_metrics.get("confidence_level", "unknown")
            confidence_score = confidence_metrics.get("overall_score", 0)
            emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence_level, "⚪")
            st.caption(f"{emoji} Confidence: {confidence_level.upper()} ({confidence_score:.0%})")

        message_placeholder = st.empty()
        thinking_placeholder = st.empty()

        response_text = ""
        for chunk in stream_response(prompt):
            response_text += chunk

            if "<think>" in response_text:
                if "</think>" in response_text:
                    thinking_content, partial_answer = parse_thinking(response_text)
                    with thinking_placeholder.expander("💭 Reasoning Process", expanded=False):
                        st.markdown(thinking_content)
                    message_placeholder.markdown(partial_answer + "▌")
                else:
                    think_start = response_text.find("<think>") + 7
                    partial_thinking = response_text[think_start:]
                    with thinking_placeholder.expander("💭 Thinking...", expanded=True):
                        st.markdown(partial_thinking + "▌")
            else:
                message_placeholder.markdown(response_text + "▌")

        thinking_content, final_answer = parse_thinking(response_text)

        thinking_placeholder.empty()
        if thinking_content:
            with thinking_placeholder.expander("💭 Reasoning Process", expanded=False):
                st.markdown(thinking_content)

        message_placeholder.markdown(final_answer)

    ChatMemory.add_message(
        "assistant",
        final_answer,
        thinking=thinking_content,
        sources=sources,
        # metadata={"confidence": confidence_metrics}
    )


def render_chat_history():
    """Render all messages from chat history."""
    messages = ChatMemory.get_messages()

    for message in messages:
        if message.role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Show confidence if available
                if hasattr(message, 'metadata') and message.metadata:
                    confidence = message.metadata.get("confidence", {})
                    if confidence:
                        level = confidence.get("confidence_level", "unknown")
                        score = confidence.get("overall_score", 0)
                        emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "⚪")
                        st.caption(f"{emoji} Confidence: {level.upper()} ({score:.0%})")

                if message.thinking:
                    with st.expander("💭 Reasoning Process", expanded=False):
                        st.markdown(message.thinking)
                st.markdown(message.content)
                if message.sources:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message.sources, 1):
                            source_name = source.get("source", "Unknown")
                            page = source.get("page", "N/A")
                            relevance = source.get("relevance", 0)

                            # Confidence indicator for source
                            if relevance >= 0.7:
                                rel_emoji = "🟢"
                            elif relevance >= 0.4:
                                rel_emoji = "🟡"
                            else:
                                rel_emoji = "🔴"

                            st.markdown(f"**[{i}] {source_name}** (Page: {page}) {rel_emoji}")
                            content = source.get("content", "")[:300]
                            st.markdown(f"> {content}...")
                            st.divider()


def render_sidebar():
    """Render the sidebar with document upload, collections, and settings."""
    with st.sidebar:
        db = st.session_state.db
        collections = db.list_collections()
        collection_names = {c.id: c.name for c in collections}

        # Status
        stats = db.get_stats()
        client = st.session_state.ollama_client
        current_model = st.session_state.ollama_config.model

        # Connection status
        if client.is_connected():
            if st.session_state.documents_loaded:
                st.markdown(f"**Status:** 🟢 Ready")
            else:
                st.markdown(f"**Status:** 🟡 No documents")
            st.caption(f"Collection: {collection_names.get(st.session_state.current_collection, 'Default')}")
            st.caption(f"Docs: {stats['total_documents']} | Model: `{current_model}`")
        else:
            st.markdown("**Status:** 🔴 Ollama Disconnected")
            st.caption("Make sure local ollama server is running.")

        st.divider()

        st.title("📁 Documents")

        # Collection selector
        st.subheader("🗂️ Collection")
        selected_collection = st.selectbox(
            "Select a collection",
            options=list(collection_names.keys()),
            format_func=lambda x: collection_names.get(x, x),
            index=list(collection_names.keys()).index(st.session_state.current_collection)
            if st.session_state.current_collection in collection_names else 0
        )

        if selected_collection != st.session_state.current_collection:
            st.session_state.current_collection = selected_collection
            # Reload vector store for new collection
            store = st.session_state.vector_manager.get_store(selected_collection)
            st.session_state.documents_loaded = store is not None
            ChatMemory.clear()
            st.rerun()

        st.html("<p style='text-align: center; margin: 0;'>-- Or --</p>")

        # New collection button
        with st.expander("Add Collection"):
            new_name = st.text_input("Name", key="new_collection_name")
            new_desc = st.text_input("Description (optional)", key="new_collection_desc")
            if st.button("Create Collection", use_container_width=True):
                if new_name:
                    new_collection = db.create_collection(new_name, new_desc)
                    st.session_state.current_collection = new_collection.id
                    st.success(f"Collection created: {new_name}")
                    st.rerun()

        # Documents in collection
        documents = get_current_documents()

        if documents:
            st.divider()
            st.subheader(f"📚 Sources ({len(documents)})")

            for doc in documents:
                icon = FILE_ICONS.get(doc.file_type, FILE_ICONS["unknown"])
                col1, col2 = st.columns([4, 1], vertical_alignment="center")

                with col1:
                    display_name = doc.filename[:30] + "..." if len(doc.filename) > 30 else doc.filename
                    st.markdown(f"{icon} **{display_name}**")
                    st.caption(f"{doc.file_type} • {doc.chunk_count} chunks")

                with col2:
                    if st.button("🗑️", key=f"del_{doc.id}", help="Delete document"):
                        if delete_document(doc.id):
                            st.success("Deleted")
                            st.rerun()


            # Clear all in collection
            if st.button("🧹️ Clear All Sources", use_container_width=True):
                for doc in documents:
                    delete_document(doc.id)
                ChatMemory.clear()
                st.rerun()

        st.divider()

        # File uploader
        st.subheader("➕ Add Source")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            # type=SUPPORTED_EXTENSIONS,
            type=["pdf", "txt", "md", "markdown", "json", "csv", "xml", "docx", "xlsx", "pptx"],
            accept_multiple_files=True,
            help="Supports PDF, Word, Excel, PowerPoint, text, and data files",
            key=f"document_uploader_{st.session_state.document_uploader_key}"
        )

        if uploaded_files:
            uploaded_any = False
            for uploaded_file in uploaded_files:
                if process_document(uploaded_file):
                    st.success(f"✅ {uploaded_file.name}")
                    uploaded_any = True

            if uploaded_any:
                st.session_state.document_uploader_key += 1
                st.rerun()

        st.html("<p style='text-align: center; margin: 0;'>-- Or --</p>")

        # URL input
        url_input = st.text_input(
            "Enter Source URL",
            placeholder="https://example.com/article",
            help="Fetch and process content from a webpage"
        )

        if st.button("Fetch URL", use_container_width=True, disabled=not url_input):
            if process_url(url_input):
                st.success("✅ URL content loaded")
                st.rerun()

        # Chat controls
        st.divider()
        st.subheader("💬 Chat")

        col1, col2 = st.columns(2, vertical_alignment="center")

        with col1:
            message_count = ChatMemory.count()
            st.metric("Messages", message_count)

        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                ChatMemory.clear()
                st.rerun()

        # Export options
        if message_count > 0:
            with st.expander("📤 Export Chat", expanded=False):
                export_format = st.selectbox(
                    "Format",
                    options=["Markdown", "JSON", "HTML", "Text", "PDF"],
                    key="export_format"
                )

                format_map = {
                    "Markdown": ExportFormat.MARKDOWN,
                    "JSON": ExportFormat.JSON,
                    "HTML": ExportFormat.HTML,
                    "Text": ExportFormat.TXT,
                    "PDF": ExportFormat.PDF
                }

                if st.button("Export", use_container_width=True, key="export_btn"):
                    try:
                        exporter = ChatExporter()
                        messages = ChatMemory.get_messages()

                        # Convert messages to export format
                        export_messages = []
                        for msg in messages:
                            export_messages.append({
                                "role": msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                                "thinking": msg.thinking,
                                "sources": msg.sources
                            })

                        selected_format = format_map[export_format]
                        content = exporter.export(export_messages, selected_format)

                        # File extension mapping
                        ext_map = {
                            ExportFormat.MARKDOWN: "md",
                            ExportFormat.JSON: "json",
                            ExportFormat.HTML: "html",
                            ExportFormat.TXT: "txt",
                            ExportFormat.PDF: "pdf"
                        }

                        filename = f"localrag_chat.{ext_map[selected_format]}"
                        mime_map = {
                            "md": "text/markdown",
                            "json": "application/json",
                            "html": "text/html",
                            "txt": "text/plain",
                            "pdf": "application/pdf"
                        }

                        if isinstance(content, bytes):
                            data = content
                        else:
                            data = content.encode("utf-8")

                        st.download_button(
                            label=f"⬇️ Download {export_format}",
                            data=data,
                            file_name=filename,
                            mime=mime_map[ext_map[selected_format]],
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Export failed: {e}")

        # Model Settings
        st.divider()
        st.title("⚙️ RAG Settings")

        with st.expander("Model Selection", expanded=False):
            # Get available models
            available_models = client.list_models()
            available_names = [m.name for m in available_models]

            # Build model options
            registry = st.session_state.model_registry
            thinking_models = registry.get_thinking_models()
            chat_models = registry.get_chat_models()

            # Combine available and recommended
            model_options = []
            for m in available_models:
                model_options.append(m.name)

            # Add recommended if not available
            for info in thinking_models[:3]:
                if info.name not in model_options:
                    model_options.append(info.name)

            # Current model selection
            try:
                current_idx = model_options.index(current_model)
            except ValueError:
                model_options.insert(0, current_model)
                current_idx = 0

            selected_model = st.selectbox(
                "Select Model",
                options=model_options,
                index=current_idx,
                format_func=lambda x: f"{'✓' if x in available_names else '↓'} {x}",
                key="model_select"
            )

            if selected_model != current_model:
                if selected_model not in available_names:
                    st.warning(f"Model not installed")
                    if st.button(f"Pull {selected_model}", key="pull_model"):
                        with st.spinner(f"Pulling {selected_model}..."):
                            if client.pull_model(selected_model):
                                st.success("Model pulled!")
                                st.session_state.ollama_config.model = selected_model
                                st.session_state.ollama_client.set_model(selected_model)
                                st.rerun()
                else:
                    st.session_state.ollama_config.model = selected_model
                    st.session_state.ollama_client.set_model(selected_model)
                    st.rerun()

            # Model info
            model_info = registry.get_model(selected_model)
            if model_info:
                caps = []
                if model_info.has_thinking:
                    caps.append("🧠 Thinking")
                if model_info.has_vision:
                    caps.append("👁️ Vision")
                if caps:
                    st.caption(" ".join(caps))
                st.caption(model_info.description)

        with st.expander("Generation Settings", expanded=False):
            config = st.session_state.ollama_config

            # Temperature
            temp = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=config.temperature,
                step=0.1,
                help="Higher = more creative",
                key="temp_slider"
            )
            if temp != config.temperature:
                config.temperature = temp

            # Context window
            ctx_options = [2048, 4096, 8192, 16384, 32768]
            try:
                ctx_idx = ctx_options.index(config.num_ctx)
            except ValueError:
                ctx_idx = 1

            ctx = st.selectbox(
                "Context Window",
                options=ctx_options,
                index=ctx_idx,
                format_func=lambda x: f"{x:,} tokens",
                key="ctx_select"
            )
            if ctx != config.num_ctx:
                config.num_ctx = ctx

            # Presets
            preset = st.selectbox(
                "Preset",
                ["Custom", "Creative", "Balanced", "Precise"],
                key="preset_select"
            )

            if preset == "Creative":
                config.temperature = 1.2
                config.top_p = 0.95
            elif preset == "Balanced":
                config.temperature = 0.7
                config.top_p = 0.9
            elif preset == "Precise":
                config.temperature = 0.3
                config.top_p = 0.5

        with st.expander("Retrieval Settings", expanded=False):
            # Retrieval mode
            retrieval_mode = st.selectbox(
                "Retrieval Mode",
                options=["hybrid", "semantic", "keyword"],
                index=["hybrid", "semantic", "keyword"].index(
                    st.session_state.rag_config.retrieval_mode.value
                ),
                help="Hybrid combines semantic + keyword search"
            )

            if retrieval_mode != st.session_state.rag_config.retrieval_mode.value:
                st.session_state.rag_config.retrieval_mode = RetrievalMode(retrieval_mode)
                st.session_state.rag_pipeline.update_config(
                    retrieval_mode=RetrievalMode(retrieval_mode)
                )

            # Semantic weight (for hybrid mode)
            if retrieval_mode == "hybrid":
                semantic_weight = st.slider(
                    "Semantic Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.rag_config.semantic_weight,
                    step=0.1,
                    help="Balance between semantic (1.0) and keyword (0.0) search"
                )

                if semantic_weight != st.session_state.rag_config.semantic_weight:
                    st.session_state.rag_config.semantic_weight = semantic_weight
                    st.session_state.rag_pipeline.update_config(
                        semantic_weight=semantic_weight
                    )

            # Query expansion
            expansion_options = {
                "None": "none",
                "Query Rewrite": "rewrite",
                "HyDE": "hyde",
                "Decompose": "decompose",
                "Step-back": "step_back"
            }

            current_expansion = st.session_state.rag_config.query_expansion_mode.value
            expansion_mode = st.selectbox(
                "Query Expansion",
                options=list(expansion_options.keys()),
                index=list(expansion_options.values()).index(current_expansion)
                if current_expansion in expansion_options.values() else 0,
                help="Improve retrieval by expanding queries"
            )

            new_expansion = expansion_options[expansion_mode]
            if new_expansion != current_expansion:
                st.session_state.rag_config.query_expansion_mode = QueryExpansionMode(new_expansion)
                st.session_state.rag_pipeline.update_config(
                    query_expansion_mode=QueryExpansionMode(new_expansion)
                )

        with st.expander("Display Settings", expanded=False):
            # Show confidence
            show_confidence = st.checkbox(
                "Show Confidence Scores",
                value=st.session_state.rag_config.show_confidence,
                help="Display retrieval confidence indicators"
            )

            if show_confidence != st.session_state.rag_config.show_confidence:
                st.session_state.rag_config.show_confidence = show_confidence
                st.session_state.rag_pipeline.update_config(show_confidence=show_confidence)

            # Enable citations
            enable_citations = st.checkbox(
                "Enable Citations",
                value=st.session_state.rag_config.enable_citations,
                help="Show source citations with responses"
            )

            if enable_citations != st.session_state.rag_config.enable_citations:
                st.session_state.rag_config.enable_citations = enable_citations
                st.session_state.rag_pipeline.update_config(enable_citations=enable_citations)

            # Number of results
            num_results = st.slider(
                "Results to Retrieve",
                min_value=1,
                max_value=10,
                value=st.session_state.rag_config.retrieval_k,
                help="Number of document chunks to retrieve"
            )

            if num_results != st.session_state.rag_config.retrieval_k:
                st.session_state.rag_config.retrieval_k = num_results
                st.session_state.rag_pipeline.update_config(retrieval_k=num_results)

        # Help section
        st.divider()
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Supported formats:**
            - 📄 PDF documents
            - 📘 Word (.docx)
            - 📗 Excel (.xlsx)
            - 📙 PowerPoint (.pptx)
            - 📝 Text files (.txt, .md)
            - 📊 Data files (.json, .csv, .xml)
            - 🌐 Web pages (via URL)

            **RAG Modes:**
            - **Hybrid**: Combines semantic + keyword search (recommended)
            - **Semantic**: Uses embeddings for meaning-based search
            - **Keyword**: Uses BM25 for exact term matching

            **Query Expansion:**
            - **Rewrite**: Generates alternative phrasings
            - **HyDE**: Creates hypothetical answer for better matching
            - **Decompose**: Breaks complex questions into sub-queries
            - **Step-back**: Generates broader context queries

            **Tips:**
            - Use hybrid mode for best results
            - Enable query expansion for complex questions
            - Check confidence scores for answer reliability
            """)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="LocalRAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # apply_styles()
    st.markdown(
        """
            <style>
            /* hr-margin */
            [data-testid=stMarkdownContainer] hr {
                margin: 1em 0;
            }
            </style>
        """,
        unsafe_allow_html=True
    )

    initialize_session_state()
    render_sidebar()

    st.title("🤖 LocalRAG")
    st.caption("Chat with your documents using local AI (Persistent Storage)")

    chat_container = st.container()

    with chat_container:
        if not st.session_state.documents_loaded:
            st.markdown(
                """
                <div style="text-align: center; padding: 60px 20px; color: #666;">
                    <h3 style="color: #333;">Welcome to LocalRAG</h3>
                    <p>Upload documents or enter a URL in the sidebar to start chatting.</p>
                    <p style="font-size: 14px; margin-top: 10px; color: #888;">
                        Your documents are saved locally and persist across sessions.
                    </p>
                    <p style="font-size: 12px; margin-top: 10px; color: #999;">
                        📄 PDF  📘 Word  📗 Excel  📙 PowerPoint  📝 Text  💻 Code 🌐 Web
                    </p>
                    <p style="font-size: 14px; margin-top: 20px;">
                        Powered by <strong>Ollama</strong> + <strong>Deepseek R1</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            render_chat_history()

    if st.session_state.documents_loaded:
        if user_input := st.chat_input("Ask a question about your documents..."):
            handle_user_input(user_input)
            st.rerun()


if __name__ == "__main__":
    main()
