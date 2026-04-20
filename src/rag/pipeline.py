"""
Advanced RAG Pipeline
Integrates hybrid retrieval, re-ranking, query expansion, and citations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from langchain.schema import Document

from .vectorstore import VectorStoreManager
from .retriever import HybridRetriever, RetrievalResult, MultiQueryRetriever
from .reranker import (
    BaseReranker, FlashRankReranker, LLMReranker,
    RerankerPipeline, RerankResult
)
from .query_expansion import (
    BaseQueryExpander, LLMQueryRewriter, HyDEExpander,
    QueryDecomposer, StepBackExpander, QueryExpansionPipeline, ExpandedQuery
)
from .citations import CitationManager, ConfidenceScorer, Citation, CitedResponse


class RetrievalMode(Enum):
    """Available retrieval modes."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QueryExpansionMode(Enum):
    """Available query expansion modes."""
    NONE = "none"
    REWRITE = "rewrite"
    HYDE = "hyde"
    DECOMPOSE = "decompose"
    STEP_BACK = "step_back"
    ALL = "all"


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval settings
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    retrieval_k: int = 5
    semantic_weight: float = 0.5  # Alpha for hybrid search

    # Re-ranking settings
    enable_reranking: bool = True
    rerank_top_k: int = 10  # Retrieve more, then rerank to k

    # Query expansion settings
    query_expansion_mode: QueryExpansionMode = QueryExpansionMode.NONE
    num_query_rewrites: int = 3

    # Citation settings
    enable_citations: bool = True
    max_citations: int = 5
    include_citation_snippets: bool = True
    citation_format: str = "numbered"  # numbered, superscript, linked

    # Confidence settings
    show_confidence: bool = True
    low_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.7


@dataclass
class RAGResult:
    """Complete result from RAG pipeline."""
    query: str
    expanded_query: Optional[ExpandedQuery] = None
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    rerank_results: List[RerankResult] = field(default_factory=list)
    final_documents: List[Document] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    confidence_metrics: Dict[str, Any] = field(default_factory=dict)
    context_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_context(self) -> str:
        """Get combined context from final documents."""
        if self.context_text:
            return self.context_text

        contexts = []
        for i, doc in enumerate(self.final_documents, 1):
            source = doc.metadata.get("source", f"Source {i}")
            contexts.append(f"[{i}] {source}:\n{doc.page_content}")

        return "\n\n".join(contexts)


class AdvancedRAGPipeline:
    """
    Advanced RAG pipeline with configurable components.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_func: Optional[Callable[[str], str]] = None,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize advanced RAG pipeline.

        Args:
            vector_store_manager: VectorStoreManager instance
            llm_func: Function to call LLM for query expansion
            config: Pipeline configuration
        """
        self.vector_store = vector_store_manager
        self.llm_func = llm_func
        self.config = config or RAGConfig()

        # Initialize components
        self._init_retriever()
        self._init_reranker()
        self._init_query_expander()
        self._init_citation_manager()

    def _init_retriever(self) -> None:
        """Initialize retrieval components."""
        self.hybrid_retriever = HybridRetriever(
            self.vector_store,
            alpha=self.config.semantic_weight
        )

        if self.llm_func:
            self.multi_query_retriever = MultiQueryRetriever(
                self.hybrid_retriever,
                llm_func=self.llm_func
            )
        else:
            self.multi_query_retriever = None

    def _init_reranker(self) -> None:
        """Initialize re-ranking components."""
        self.reranker_pipeline = RerankerPipeline()

        if self.config.enable_reranking:
            try:
                # Try FlashRank first
                self.reranker_pipeline.add_reranker(FlashRankReranker())
            except ImportError:
                # Fallback to LLM reranker
                if self.llm_func:
                    self.reranker_pipeline.add_reranker(
                        LLMReranker(llm_func=self.llm_func)
                    )

    def _init_query_expander(self) -> None:
        """Initialize query expansion components."""
        self.query_expander = QueryExpansionPipeline()

        mode = self.config.query_expansion_mode

        if mode == QueryExpansionMode.NONE:
            return

        if self.llm_func:
            if mode in [QueryExpansionMode.REWRITE, QueryExpansionMode.ALL]:
                self.query_expander.add_expander(
                    LLMQueryRewriter(
                        llm_func=self.llm_func,
                        num_rewrites=self.config.num_query_rewrites
                    )
                )

            if mode in [QueryExpansionMode.HYDE, QueryExpansionMode.ALL]:
                self.query_expander.add_expander(
                    HyDEExpander(llm_func=self.llm_func)
                )

            if mode in [QueryExpansionMode.DECOMPOSE, QueryExpansionMode.ALL]:
                self.query_expander.add_expander(
                    QueryDecomposer(llm_func=self.llm_func)
                )

            if mode in [QueryExpansionMode.STEP_BACK, QueryExpansionMode.ALL]:
                self.query_expander.add_expander(
                    StepBackExpander(llm_func=self.llm_func)
                )

    def _init_citation_manager(self) -> None:
        """Initialize citation components."""
        self.citation_manager = CitationManager(
            max_citations=self.config.max_citations
        )
        self.confidence_scorer = ConfidenceScorer(
            low_threshold=self.config.low_confidence_threshold,
            high_threshold=self.config.high_confidence_threshold
        )

    def update_config(self, **kwargs) -> None:
        """
        Update pipeline configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Reinitialize affected components
        if "semantic_weight" in kwargs:
            self.hybrid_retriever.set_alpha(kwargs["semantic_weight"])

        if "query_expansion_mode" in kwargs or "num_query_rewrites" in kwargs:
            self._init_query_expander()

        if "enable_reranking" in kwargs:
            self._init_reranker()

    def retrieve(
        self,
        collection_id: str,
        query: str,
        **kwargs
    ) -> RAGResult:
        """
        Execute full RAG pipeline.

        Args:
            collection_id: Collection to search
            query: User query
            **kwargs: Override config parameters

        Returns:
            Complete RAG result
        """
        # Apply any overrides
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        result = RAGResult(query=query)

        # Step 1: Query Expansion
        if config.query_expansion_mode != QueryExpansionMode.NONE and self.query_expander.expanders:
            expanded = self.query_expander.expand(query)
            result.expanded_query = expanded
            queries_to_search = expanded.expanded_queries
        else:
            queries_to_search = [query]

        # Step 2: Retrieval
        all_retrieval_results: Dict[str, RetrievalResult] = {}
        retrieve_k = config.rerank_top_k if config.enable_reranking else config.retrieval_k

        for q in queries_to_search:
            results = self.hybrid_retriever.retrieve(
                collection_id,
                q,
                k=retrieve_k,
                method=config.retrieval_mode.value
            )

            for r in results:
                doc_key = r.document.page_content[:500]
                if doc_key not in all_retrieval_results:
                    all_retrieval_results[doc_key] = r
                else:
                    # Boost score for documents found by multiple queries
                    existing = all_retrieval_results[doc_key]
                    existing.combined_score += r.combined_score * 0.3

        result.retrieval_results = list(all_retrieval_results.values())
        result.retrieval_results.sort(key=lambda x: x.combined_score, reverse=True)

        # Step 3: Re-ranking
        if config.enable_reranking and result.retrieval_results:
            rerank_results = self.reranker_pipeline.rerank(
                query,
                result.retrieval_results,
                top_k=config.retrieval_k
            )
            result.rerank_results = rerank_results
            result.final_documents = [r.document for r in rerank_results]
        else:
            result.final_documents = [
                r.document for r in result.retrieval_results[:config.retrieval_k]
            ]

        # Step 4: Calculate confidence
        if result.rerank_results:
            scores = [r.rerank_score for r in result.rerank_results]
        else:
            scores = [r.combined_score for r in result.retrieval_results[:config.retrieval_k]]

        if scores and config.show_confidence:
            result.confidence_metrics = self.confidence_scorer.score_retrieval(
                result.final_documents,
                scores
            )

        # Step 5: Create citations
        if config.enable_citations and result.final_documents:
            result.citations = self.citation_manager.create_citations(
                result.final_documents,
                scores
            )

        # Build context text
        result.context_text = result.get_context()

        # Add metadata
        result.metadata = {
            "retrieval_mode": config.retrieval_mode.value,
            "num_results": len(result.final_documents),
            "reranking_applied": config.enable_reranking and bool(result.rerank_results),
            "query_expansion_applied": bool(result.expanded_query),
        }

        return result

    def build_bm25_index(self, collection_id: str, documents: List[Document]) -> None:
        """Build BM25 index for a collection."""
        self.hybrid_retriever.build_bm25_index(collection_id, documents)

    def add_to_bm25_index(self, collection_id: str, documents: List[Document]) -> None:
        """Add documents to BM25 index."""
        self.hybrid_retriever.add_to_bm25_index(collection_id, documents)

    def clear_bm25_index(self, collection_id: str) -> None:
        """Clear BM25 index for a collection."""
        self.hybrid_retriever.clear_bm25_index(collection_id)

    def format_response_with_citations(
        self,
        response: str,
        rag_result: RAGResult
    ) -> str:
        """
        Format response with inline citations and footer.

        Args:
            response: LLM response text
            rag_result: RAG result with citations

        Returns:
            Formatted response with citations
        """
        if not self.config.enable_citations or not rag_result.citations:
            return response

        # Add inline citations
        cited = self.citation_manager.add_inline_citations(
            response,
            rag_result.citations,
            format_type=self.config.citation_format
        )

        # Add citation footer
        footer = self.citation_manager.format_citation_footer(
            rag_result.citations,
            include_snippets=self.config.include_citation_snippets
        )

        # Add confidence banner if enabled
        if self.config.show_confidence and rag_result.confidence_metrics:
            confidence_banner = self.confidence_scorer.format_confidence_banner(
                rag_result.confidence_metrics
            )
            return f"{cited.cited_response}\n\n{confidence_banner}{footer}"

        return f"{cited.cited_response}{footer}"


def create_rag_pipeline(
    vector_store_manager: VectorStoreManager,
    llm_func: Optional[Callable[[str], str]] = None,
    **config_kwargs
) -> AdvancedRAGPipeline:
    """
    Factory function to create RAG pipeline.

    Args:
        vector_store_manager: Vector store manager
        llm_func: LLM function
        **config_kwargs: Configuration options

    Returns:
        Configured RAG pipeline
    """
    config = RAGConfig(**config_kwargs)
    return AdvancedRAGPipeline(
        vector_store_manager,
        llm_func=llm_func,
        config=config
    )
