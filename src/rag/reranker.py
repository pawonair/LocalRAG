"""
Re-ranking Module
Cross-encoder based re-ranking for improved result quality.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

from langchain.schema import Document

from .retriever import RetrievalResult


@dataclass
class RerankResult:
    """Result from re-ranking with updated scores."""
    document: Document
    original_score: float
    rerank_score: float
    final_rank: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseReranker(ABC):
    """Base class for re-rankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        Re-rank documents based on query relevance.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of results to return

        Returns:
            Re-ranked results
        """
        pass


class FlashRankReranker(BaseReranker):
    """
    Re-ranker using FlashRank for fast cross-encoder scoring.
    FlashRank is a lightweight, fast re-ranking library.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize FlashRank reranker.

        Args:
            model_name: Model to use for re-ranking
        """
        self.model_name = model_name
        self._ranker = None

    @property
    def ranker(self):
        """Lazy load the ranker."""
        if self._ranker is None:
            try:
                from flashrank import Ranker, RerankRequest
                self._ranker = Ranker(model_name=self.model_name)
            except ImportError:
                raise ImportError(
                    "FlashRank not installed. Install with: pip install flashrank"
                )
        return self._ranker

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        Re-rank documents using FlashRank.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of results

        Returns:
            Re-ranked results
        """
        if not documents:
            return []

        try:
            from flashrank import RerankRequest

            # Prepare passages for FlashRank
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(documents)
            ]

            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)

            # Get re-ranked results
            results = self.ranker.rerank(rerank_request)

            # Convert to RerankResult objects
            rerank_results = []
            for rank, result in enumerate(results[:top_k], 1):
                doc_idx = result["id"]
                rerank_results.append(RerankResult(
                    document=documents[doc_idx],
                    original_score=0.0,  # Original score not preserved
                    rerank_score=result["score"],
                    final_rank=rank,
                    metadata=result.get("meta", {})
                ))

            return rerank_results

        except Exception as e:
            # Fallback: return documents in original order
            return [
                RerankResult(
                    document=doc,
                    original_score=0.0,
                    rerank_score=1.0 / (i + 1),
                    final_rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]


class LLMReranker(BaseReranker):
    """
    Re-ranker using LLM for relevance scoring.
    Useful when cross-encoder models are not available.
    """

    def __init__(self, llm_func=None):
        """
        Initialize LLM reranker.

        Args:
            llm_func: Function to call LLM
        """
        self.llm_func = llm_func

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        Re-rank documents using LLM scoring.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of results

        Returns:
            Re-ranked results
        """
        if not documents or not self.llm_func:
            return [
                RerankResult(
                    document=doc,
                    original_score=0.0,
                    rerank_score=1.0 / (i + 1),
                    final_rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        scored_docs = []

        for doc in documents:
            prompt = f"""Rate the relevance of the following passage to the query on a scale of 0-10.
Only respond with a single number.

Query: {query}

Passage: {doc.page_content[:1000]}

Relevance score (0-10):"""

            try:
                response = self.llm_func(prompt)
                # Extract number from response
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', response)
                score = float(numbers[0]) / 10.0 if numbers else 0.5
                score = min(1.0, max(0.0, score))
            except Exception:
                score = 0.5

            scored_docs.append((doc, score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for rank, (doc, score) in enumerate(scored_docs[:top_k], 1):
            results.append(RerankResult(
                document=doc,
                original_score=0.0,
                rerank_score=score,
                final_rank=rank
            ))

        return results


class CohereReranker(BaseReranker):
    """
    Re-ranker using Cohere's rerank API.
    Requires Cohere API key.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v2.0"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Rerank model name
        """
        import os
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy load Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError(
                    "Cohere not installed. Install with: pip install cohere"
                )
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        Re-rank documents using Cohere.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of results

        Returns:
            Re-ranked results
        """
        if not documents or not self.api_key:
            return [
                RerankResult(
                    document=doc,
                    original_score=0.0,
                    rerank_score=1.0 / (i + 1),
                    final_rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        try:
            # Prepare documents for Cohere
            doc_texts = [doc.page_content for doc in documents]

            # Call Cohere rerank
            response = self.client.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_k,
                model=self.model
            )

            # Convert results
            results = []
            for rank, result in enumerate(response.results, 1):
                results.append(RerankResult(
                    document=documents[result.index],
                    original_score=0.0,
                    rerank_score=result.relevance_score,
                    final_rank=rank
                ))

            return results

        except Exception as e:
            # Fallback
            return [
                RerankResult(
                    document=doc,
                    original_score=0.0,
                    rerank_score=1.0 / (i + 1),
                    final_rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]


class RerankerPipeline:
    """
    Pipeline for applying multiple re-ranking stages.
    """

    def __init__(self, rerankers: Optional[List[BaseReranker]] = None):
        """
        Initialize reranker pipeline.

        Args:
            rerankers: List of rerankers to apply in sequence
        """
        self.rerankers = rerankers or []

    def add_reranker(self, reranker: BaseReranker) -> None:
        """Add a reranker to the pipeline."""
        self.rerankers.append(reranker)

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        Apply re-ranking pipeline to retrieval results.

        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Number of final results

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        # Extract documents from results
        documents = [r.document for r in results]

        # Track original scores
        original_scores = {
            doc.page_content[:500]: r.combined_score
            for doc, r in zip(documents, results)
        }

        # Apply each reranker in sequence
        current_docs = documents
        final_results = None

        for reranker in self.rerankers:
            rerank_results = reranker.rerank(query, current_docs, top_k=len(current_docs))

            # Update documents order for next stage
            current_docs = [r.document for r in rerank_results]
            final_results = rerank_results

        if not final_results:
            # No rerankers, return original results as RerankResults
            return [
                RerankResult(
                    document=r.document,
                    original_score=r.combined_score,
                    rerank_score=r.combined_score,
                    final_rank=r.rank
                )
                for r in results[:top_k]
            ]

        # Add original scores to final results
        for result in final_results[:top_k]:
            doc_key = result.document.page_content[:500]
            result.original_score = original_scores.get(doc_key, 0.0)

        return final_results[:top_k]


def create_reranker(
    reranker_type: str = "flashrank",
    **kwargs
) -> BaseReranker:
    """
    Factory function to create a reranker.

    Args:
        reranker_type: Type of reranker ("flashrank", "llm", "cohere")
        **kwargs: Additional arguments for the reranker

    Returns:
        Reranker instance
    """
    rerankers = {
        "flashrank": FlashRankReranker,
        "llm": LLMReranker,
        "cohere": CohereReranker,
    }

    if reranker_type not in rerankers:
        raise ValueError(f"Unknown reranker type: {reranker_type}")

    return rerankers[reranker_type](**kwargs)
