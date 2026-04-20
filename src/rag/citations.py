"""
Citation System
Display source chunks with responses and confidence indicators.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import hashlib

from langchain.schema import Document


@dataclass
class Citation:
    """Represents a citation/source reference."""
    id: str
    document: Document
    relevance_score: float
    chunk_index: int
    source_name: str
    source_type: str
    page_number: Optional[int] = None
    highlight_text: Optional[str] = None
    confidence: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.document.page_content,
            "relevance_score": self.relevance_score,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "highlight_text": self.highlight_text,
            "confidence": self.confidence,
            "metadata": self.document.metadata,
        }


@dataclass
class CitedResponse:
    """Response with inline citations."""
    original_response: str
    cited_response: str
    citations: List[Citation] = field(default_factory=list)
    overall_confidence: float = 0.0
    citation_coverage: float = 0.0  # % of response covered by citations


class CitationManager:
    """
    Manages citations for RAG responses.
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.3,
        max_citations: int = 5
    ):
        """
        Initialize citation manager.

        Args:
            min_relevance_threshold: Minimum score to include citation
            max_citations: Maximum citations to include
        """
        self.min_relevance_threshold = min_relevance_threshold
        self.max_citations = max_citations
        self._citation_counter = 0

    def _generate_citation_id(self, document: Document) -> str:
        """Generate unique citation ID."""
        content_hash = hashlib.md5(
            document.page_content[:100].encode()
        ).hexdigest()[:6]
        self._citation_counter += 1
        return f"c{self._citation_counter}_{content_hash}"

    def _extract_source_info(self, document: Document) -> Tuple[str, str, Optional[int]]:
        """
        Extract source information from document metadata.

        Returns:
            (source_name, source_type, page_number)
        """
        metadata = document.metadata or {}

        # Get source name
        source_name = (
            metadata.get("source") or
            metadata.get("filename") or
            metadata.get("file_name") or
            metadata.get("title") or
            "Unknown Source"
        )

        # Get source type
        source_type = (
            metadata.get("file_type") or
            metadata.get("type") or
            metadata.get("loader") or
            "document"
        )

        # Get page number if available
        page_number = metadata.get("page") or metadata.get("page_number")
        if page_number is not None:
            try:
                page_number = int(page_number)
            except (ValueError, TypeError):
                page_number = None

        return source_name, source_type, page_number

    def _calculate_confidence(self, relevance_score: float) -> str:
        """
        Calculate confidence level from relevance score.

        Args:
            relevance_score: Relevance score (0-1)

        Returns:
            Confidence level string
        """
        if relevance_score >= 0.7:
            return "high"
        elif relevance_score >= 0.4:
            return "medium"
        else:
            return "low"

    def create_citations(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None
    ) -> List[Citation]:
        """
        Create citations from documents.

        Args:
            documents: Source documents
            scores: Relevance scores (optional)

        Returns:
            List of citations
        """
        if not scores:
            scores = [0.5] * len(documents)

        citations = []

        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Skip low-relevance documents
            if score < self.min_relevance_threshold:
                continue

            source_name, source_type, page_number = self._extract_source_info(doc)

            citation = Citation(
                id=self._generate_citation_id(doc),
                document=doc,
                relevance_score=score,
                chunk_index=i,
                source_name=source_name,
                source_type=source_type,
                page_number=page_number,
                confidence=self._calculate_confidence(score)
            )
            citations.append(citation)

            if len(citations) >= self.max_citations:
                break

        return citations

    def add_inline_citations(
        self,
        response: str,
        citations: List[Citation],
        format_type: str = "numbered"
    ) -> CitedResponse:
        """
        Add inline citations to response.

        Args:
            response: Original response text
            citations: List of citations
            format_type: "numbered" [1], "superscript" ¹, or "linked" [source]

        Returns:
            CitedResponse with inline citations
        """
        if not citations:
            return CitedResponse(
                original_response=response,
                cited_response=response,
                citations=[],
                overall_confidence=0.0,
                citation_coverage=0.0
            )

        cited_response = response
        matched_citations = []

        # Find sentences/phrases that match citation content
        sentences = re.split(r'(?<=[.!?])\s+', response)

        for i, citation in enumerate(citations, 1):
            citation_content = citation.document.page_content.lower()

            for sentence in sentences:
                # Check if sentence content appears in citation
                sentence_lower = sentence.lower()
                words = sentence_lower.split()

                # Count matching words
                matching_words = sum(
                    1 for word in words
                    if len(word) > 3 and word in citation_content
                )

                # If enough words match, this sentence is supported by citation
                if matching_words >= 3 or (len(words) > 0 and matching_words / len(words) > 0.3):
                    # Add citation reference
                    if format_type == "numbered":
                        ref = f" [{i}]"
                    elif format_type == "superscript":
                        ref = f"^{i}"
                    else:
                        ref = f" [{citation.source_name}]"

                    # Add reference after the sentence (if not already added)
                    if ref not in cited_response:
                        old_sentence = sentence
                        if sentence.endswith(('.', '!', '?')):
                            new_sentence = sentence[:-1] + ref + sentence[-1]
                        else:
                            new_sentence = sentence + ref

                        cited_response = cited_response.replace(old_sentence, new_sentence, 1)
                        matched_citations.append(citation)
                        break

        # Calculate metrics
        overall_confidence = sum(c.relevance_score for c in matched_citations) / len(citations) if citations else 0
        citation_coverage = len(matched_citations) / len(sentences) if sentences else 0

        return CitedResponse(
            original_response=response,
            cited_response=cited_response,
            citations=citations,
            overall_confidence=overall_confidence,
            citation_coverage=min(1.0, citation_coverage)
        )

    def format_citation_footer(
        self,
        citations: List[Citation],
        include_snippets: bool = True,
        snippet_length: int = 150
    ) -> str:
        """
        Format citations as a footer/reference section.

        Args:
            citations: List of citations
            include_snippets: Whether to include content snippets
            snippet_length: Length of content snippets

        Returns:
            Formatted citation footer
        """
        if not citations:
            return ""

        lines = ["", "---", "**Sources:**"]

        for i, citation in enumerate(citations, 1):
            # Source line
            source_line = f"[{i}] **{citation.source_name}**"

            if citation.page_number:
                source_line += f" (Page {citation.page_number})"

            # Add confidence indicator
            confidence_emoji = {
                "high": "🟢",
                "medium": "🟡",
                "low": "🔴"
            }.get(citation.confidence, "⚪")

            source_line += f" {confidence_emoji}"
            lines.append(source_line)

            # Add snippet
            if include_snippets:
                content = citation.document.page_content[:snippet_length]
                if len(citation.document.page_content) > snippet_length:
                    content += "..."
                lines.append(f"   > {content}")
                lines.append("")

        return "\n".join(lines)


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG responses.
    """

    def __init__(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7
    ):
        """
        Initialize confidence scorer.

        Args:
            low_threshold: Score below this is low confidence
            high_threshold: Score above this is high confidence
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def score_retrieval(
        self,
        documents: List[Document],
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        Score the quality of retrieval results.

        Args:
            documents: Retrieved documents
            scores: Relevance scores

        Returns:
            Confidence metrics
        """
        if not documents or not scores:
            return {
                "overall_score": 0.0,
                "confidence_level": "none",
                "top_score": 0.0,
                "avg_score": 0.0,
                "num_high_confidence": 0,
                "warning": "No documents retrieved"
            }

        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        num_high = sum(1 for s in scores if s >= self.high_threshold)

        # Calculate overall score
        # Weight top score more heavily
        overall_score = (top_score * 0.6) + (avg_score * 0.4)

        # Determine confidence level
        if overall_score >= self.high_threshold:
            confidence_level = "high"
            warning = None
        elif overall_score >= self.low_threshold:
            confidence_level = "medium"
            warning = "Answer may be incomplete or partially relevant"
        else:
            confidence_level = "low"
            warning = "Low confidence - sources may not fully address the question"

        return {
            "overall_score": overall_score,
            "confidence_level": confidence_level,
            "top_score": top_score,
            "avg_score": avg_score,
            "num_high_confidence": num_high,
            "warning": warning
        }

    def get_confidence_indicator(self, score: float) -> str:
        """
        Get visual indicator for confidence score.

        Args:
            score: Confidence score (0-1)

        Returns:
            Confidence indicator string
        """
        if score >= self.high_threshold:
            return "🟢 High Confidence"
        elif score >= self.low_threshold:
            return "🟡 Medium Confidence"
        else:
            return "🔴 Low Confidence"

    def format_confidence_banner(
        self,
        confidence_metrics: Dict[str, Any]
    ) -> str:
        """
        Format confidence as a display banner.

        Args:
            confidence_metrics: Metrics from score_retrieval

        Returns:
            Formatted banner string
        """
        level = confidence_metrics.get("confidence_level", "unknown")
        score = confidence_metrics.get("overall_score", 0)
        warning = confidence_metrics.get("warning")

        # Create progress bar
        filled = int(score * 10)
        bar = "█" * filled + "░" * (10 - filled)

        emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"}.get(level, "⚪")

        banner = f"{emoji} **Confidence: {level.upper()}** [{bar}] {score:.0%}"

        if warning:
            banner += f"\n⚠️ {warning}"

        return banner
