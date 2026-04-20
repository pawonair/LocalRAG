"""
RAG Module
Retrieval-Augmented Generation components.
"""

from .vectorstore import VectorStoreManager
from .retriever import (
    HybridRetriever,
    BM25Index,
    MultiQueryRetriever,
    RetrievalResult,
)
from .reranker import (
    BaseReranker,
    FlashRankReranker,
    LLMReranker,
    CohereReranker,
    RerankerPipeline,
    RerankResult,
    create_reranker,
)
from .query_expansion import (
    BaseQueryExpander,
    LLMQueryRewriter,
    HyDEExpander,
    QueryDecomposer,
    StepBackExpander,
    QueryExpansionPipeline,
    ExpandedQuery,
    create_query_expander,
)
from .citations import (
    Citation,
    CitedResponse,
    CitationManager,
    ConfidenceScorer,
)
from .pipeline import (
    AdvancedRAGPipeline,
    RAGConfig,
    RAGResult,
    RetrievalMode,
    QueryExpansionMode,
    create_rag_pipeline,
)

__all__ = [
    # Vector Store
    "VectorStoreManager",

    # Retrieval
    "HybridRetriever",
    "BM25Index",
    "MultiQueryRetriever",
    "RetrievalResult",

    # Re-ranking
    "BaseReranker",
    "FlashRankReranker",
    "LLMReranker",
    "CohereReranker",
    "RerankerPipeline",
    "RerankResult",
    "create_reranker",

    # Query Expansion
    "BaseQueryExpander",
    "LLMQueryRewriter",
    "HyDEExpander",
    "QueryDecomposer",
    "StepBackExpander",
    "QueryExpansionPipeline",
    "ExpandedQuery",
    "create_query_expander",

    # Citations
    "Citation",
    "CitedResponse",
    "CitationManager",
    "ConfidenceScorer",

    # Pipeline
    "AdvancedRAGPipeline",
    "RAGConfig",
    "RAGResult",
    "RetrievalMode",
    "QueryExpansionMode",
    "create_rag_pipeline",
]
