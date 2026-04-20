"""
Hybrid Retriever
Combines BM25 keyword search with semantic search using Reciprocal Rank Fusion (RRF).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
from collections import defaultdict

from rank_bm25 import BM25Okapi
from langchain.schema import Document


@dataclass
class RetrievalResult:
    """Result from retrieval with scores and metadata."""
    document: Document
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0
    retrieval_method: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.document.page_content,
            "metadata": self.document.metadata,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
        }


class BM25Index:
    """BM25 index for keyword-based retrieval."""

    def __init__(self, documents: Optional[List[Document]] = None):
        """
        Initialize BM25 index.

        Args:
            documents: Optional list of documents to index
        """
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

        if documents:
            self.add_documents(documents)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        # Remove punctuation and split
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the index.

        Args:
            documents: Documents to add
        """
        self.documents.extend(documents)

        # Tokenize new documents
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self._tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or not self.documents:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        scored_docs = list(zip(range(len(scores)), scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_docs[:k]

        # Return documents with scores
        results = []
        for idx, score in top_k:
            if score > 0:  # Only include documents with positive scores
                results.append((self.documents[idx], score))

        return results

    def clear(self) -> None:
        """Clear the index."""
        self.documents = []
        self.bm25 = None
        self._tokenized_corpus = []


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        vector_store_manager,
        alpha: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store_manager: VectorStoreManager instance
            alpha: Weight for semantic search (1-alpha for keyword)
            rrf_k: RRF constant (typically 60)
        """
        self.vector_store = vector_store_manager
        self.alpha = alpha
        self.rrf_k = rrf_k
        self._bm25_indices: Dict[str, BM25Index] = {}

    def _get_bm25_index(self, collection_id: str) -> Optional[BM25Index]:
        """Get or create BM25 index for a collection."""
        if collection_id not in self._bm25_indices:
            # Need to build index from vector store documents
            store = self.vector_store.get_store(collection_id)
            if store and hasattr(store.docstore, '_dict'):
                documents = list(store.docstore._dict.values())
                self._bm25_indices[collection_id] = BM25Index(documents)
            else:
                return None

        return self._bm25_indices[collection_id]

    def build_bm25_index(self, collection_id: str, documents: List[Document]) -> None:
        """
        Build BM25 index for a collection.

        Args:
            collection_id: Collection identifier
            documents: Documents to index
        """
        self._bm25_indices[collection_id] = BM25Index(documents)

    def add_to_bm25_index(self, collection_id: str, documents: List[Document]) -> None:
        """
        Add documents to existing BM25 index.

        Args:
            collection_id: Collection identifier
            documents: Documents to add
        """
        if collection_id not in self._bm25_indices:
            self._bm25_indices[collection_id] = BM25Index()

        self._bm25_indices[collection_id].add_documents(documents)

    def clear_bm25_index(self, collection_id: str) -> None:
        """Clear BM25 index for a collection."""
        if collection_id in self._bm25_indices:
            del self._bm25_indices[collection_id]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each ranking list

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            k: Number of results to return

        Returns:
            Combined and ranked results
        """
        # Track scores by document content (as unique identifier)
        doc_scores: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "document": None,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "rrf_semantic": 0.0,
            "rrf_keyword": 0.0,
        })

        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_key = doc.page_content[:500]  # Use first 500 chars as key
            doc_scores[doc_key]["document"] = doc
            doc_scores[doc_key]["semantic_score"] = score
            doc_scores[doc_key]["rrf_semantic"] = 1.0 / (self.rrf_k + rank)

        # Process keyword results
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_key = doc.page_content[:500]
            if doc_scores[doc_key]["document"] is None:
                doc_scores[doc_key]["document"] = doc
            doc_scores[doc_key]["keyword_score"] = score
            doc_scores[doc_key]["rrf_keyword"] = 1.0 / (self.rrf_k + rank)

        # Calculate combined RRF scores with weighting
        results = []
        for doc_key, data in doc_scores.items():
            if data["document"] is None:
                continue

            # Weighted RRF combination
            combined_score = (
                self.alpha * data["rrf_semantic"] +
                (1 - self.alpha) * data["rrf_keyword"]
            )

            result = RetrievalResult(
                document=data["document"],
                semantic_score=data["semantic_score"],
                keyword_score=data["keyword_score"],
                combined_score=combined_score,
                retrieval_method="hybrid"
            )
            results.append(result)

        # Sort by combined score and assign ranks
        results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(results[:k], 1):
            result.rank = i

        return results[:k]

    def retrieve(
        self,
        collection_id: str,
        query: str,
        k: int = 5,
        semantic_k: int = 10,
        keyword_k: int = 10,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using specified method.

        Args:
            collection_id: Collection to search
            query: Search query
            k: Number of final results
            semantic_k: Number of semantic search results
            keyword_k: Number of keyword search results
            method: "hybrid", "semantic", or "keyword"

        Returns:
            List of retrieval results
        """
        results = []

        if method in ["hybrid", "semantic"]:
            # Semantic search using FAISS
            semantic_results = self.vector_store.search_with_scores(
                collection_id, query, k=semantic_k
            )

            if method == "semantic":
                # Return semantic results only
                for rank, (doc, score) in enumerate(semantic_results[:k], 1):
                    # Convert FAISS distance to similarity score
                    similarity = 1.0 / (1.0 + score)
                    results.append(RetrievalResult(
                        document=doc,
                        semantic_score=similarity,
                        combined_score=similarity,
                        rank=rank,
                        retrieval_method="semantic"
                    ))
                return results

        if method in ["hybrid", "keyword"]:
            # Keyword search using BM25
            bm25_index = self._get_bm25_index(collection_id)
            keyword_results = []

            if bm25_index:
                keyword_results = bm25_index.search(query, k=keyword_k)

            if method == "keyword":
                # Normalize BM25 scores
                max_score = max([s for _, s in keyword_results], default=1.0)
                for rank, (doc, score) in enumerate(keyword_results[:k], 1):
                    normalized = score / max_score if max_score > 0 else 0
                    results.append(RetrievalResult(
                        document=doc,
                        keyword_score=normalized,
                        combined_score=normalized,
                        rank=rank,
                        retrieval_method="keyword"
                    ))
                return results

        # Hybrid: combine semantic and keyword results
        if method == "hybrid":
            results = self._reciprocal_rank_fusion(
                semantic_results,
                keyword_results if bm25_index else [],
                k
            )

        return results

    def set_alpha(self, alpha: float) -> None:
        """
        Set the weighting between semantic and keyword search.

        Args:
            alpha: Weight for semantic (0.0 = keyword only, 1.0 = semantic only)
        """
        self.alpha = max(0.0, min(1.0, alpha))


class MultiQueryRetriever:
    """
    Retriever that decomposes complex queries into sub-queries.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        llm_func=None
    ):
        """
        Initialize multi-query retriever.

        Args:
            hybrid_retriever: HybridRetriever instance
            llm_func: Function to call LLM for query decomposition
        """
        self.retriever = hybrid_retriever
        self.llm_func = llm_func

    def _generate_sub_queries(self, query: str) -> List[str]:
        """
        Generate sub-queries from a complex query.

        Args:
            query: Original query

        Returns:
            List of sub-queries
        """
        if not self.llm_func:
            # Fallback: return original query
            return [query]

        prompt = f"""Break down the following question into 2-4 simpler, focused sub-questions that together would help answer the original question.

Original question: {query}

Return only the sub-questions, one per line. Do not include numbering or explanations.
Sub-questions:"""

        try:
            response = self.llm_func(prompt)
            # Parse response into sub-queries
            sub_queries = [
                q.strip() for q in response.strip().split('\n')
                if q.strip() and not q.strip().startswith('#')
            ]

            # Always include original query
            if query not in sub_queries:
                sub_queries.insert(0, query)

            return sub_queries[:5]  # Limit to 5 sub-queries

        except Exception:
            return [query]

    def retrieve(
        self,
        collection_id: str,
        query: str,
        k: int = 5,
        per_query_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using multiple sub-queries.

        Args:
            collection_id: Collection to search
            query: Original query
            k: Total number of results
            per_query_k: Results per sub-query

        Returns:
            Deduplicated and ranked results
        """
        sub_queries = self._generate_sub_queries(query)

        # Collect all results
        all_results: Dict[str, RetrievalResult] = {}

        for sub_query in sub_queries:
            results = self.retriever.retrieve(
                collection_id,
                sub_query,
                k=per_query_k
            )

            for result in results:
                doc_key = result.document.page_content[:500]

                if doc_key not in all_results:
                    all_results[doc_key] = result
                else:
                    # Boost score for documents appearing in multiple queries
                    existing = all_results[doc_key]
                    existing.combined_score += result.combined_score * 0.5

        # Sort and rank final results
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.combined_score, reverse=True)

        for i, result in enumerate(final_results[:k], 1):
            result.rank = i
            result.retrieval_method = "multi-query"

        return final_results[:k]
