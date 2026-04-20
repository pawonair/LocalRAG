"""
Query Expansion Module
LLM-based query rewriting, HyDE, and query decomposition.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class ExpandedQuery:
    """Represents an expanded query with metadata."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    hypothetical_document: Optional[str] = None
    expansion_method: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseQueryExpander(ABC):
    """Base class for query expansion strategies."""

    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query.

        Args:
            query: Original query

        Returns:
            Expanded query object
        """
        pass


class LLMQueryRewriter(BaseQueryExpander):
    """
    Query rewriter using LLM to generate alternative queries.
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        num_rewrites: int = 3
    ):
        """
        Initialize query rewriter.

        Args:
            llm_func: Function to call LLM
            num_rewrites: Number of query rewrites to generate
        """
        self.llm_func = llm_func
        self.num_rewrites = num_rewrites

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query by generating alternative phrasings.

        Args:
            query: Original query

        Returns:
            Expanded query with alternatives
        """
        if not self.llm_func:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none"
            )

        prompt = f"""Generate {self.num_rewrites} alternative ways to ask the following question.
Each alternative should capture the same intent but use different words or perspectives.
Return only the alternative questions, one per line.

Original question: {query}

Alternative questions:"""

        try:
            response = self.llm_func(prompt)

            # Parse alternatives from response
            alternatives = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present
                if line and len(line) > 2:
                    if line[0].isdigit() and line[1] in '.):':
                        line = line[2:].strip()
                    elif line[:2].replace('.', '').replace(')', '').isdigit():
                        line = line[3:].strip()

                    if line and line not in alternatives:
                        alternatives.append(line)

            # Always include original query first
            all_queries = [query] + alternatives[:self.num_rewrites]

            return ExpandedQuery(
                original_query=query,
                expanded_queries=all_queries,
                expansion_method="llm_rewrite"
            )

        except Exception as e:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none",
                metadata={"error": str(e)}
            )


class HyDEExpander(BaseQueryExpander):
    """
    Hypothetical Document Embeddings (HyDE) query expansion.
    Generates a hypothetical answer to use for retrieval.
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        document_type: str = "passage"
    ):
        """
        Initialize HyDE expander.

        Args:
            llm_func: Function to call LLM
            document_type: Type of document to generate ("passage", "article", "answer")
        """
        self.llm_func = llm_func
        self.document_type = document_type

    def expand(self, query: str) -> ExpandedQuery:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: Original query

        Returns:
            Expanded query with hypothetical document
        """
        if not self.llm_func:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none"
            )

        prompts = {
            "passage": f"""Write a short passage (2-3 sentences) that would perfectly answer the following question.
Write as if you're quoting from an authoritative source.

Question: {query}

Passage:""",

            "article": f"""Write a brief article excerpt (3-5 sentences) that would contain the answer to the following question.
Include relevant context and details.

Question: {query}

Article excerpt:""",

            "answer": f"""Provide a direct, factual answer to the following question.
Be concise but complete.

Question: {query}

Answer:"""
        }

        prompt = prompts.get(self.document_type, prompts["passage"])

        try:
            hypothetical_doc = self.llm_func(prompt).strip()

            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                hypothetical_document=hypothetical_doc,
                expansion_method="hyde",
                metadata={"document_type": self.document_type}
            )

        except Exception as e:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none",
                metadata={"error": str(e)}
            )


class QueryDecomposer(BaseQueryExpander):
    """
    Decomposes complex queries into simpler sub-queries.
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        max_sub_queries: int = 4
    ):
        """
        Initialize query decomposer.

        Args:
            llm_func: Function to call LLM
            max_sub_queries: Maximum number of sub-queries
        """
        self.llm_func = llm_func
        self.max_sub_queries = max_sub_queries

    def expand(self, query: str) -> ExpandedQuery:
        """
        Decompose query into sub-queries.

        Args:
            query: Original query

        Returns:
            Expanded query with sub-queries
        """
        if not self.llm_func:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none"
            )

        prompt = f"""Break down the following complex question into {self.max_sub_queries} simpler, focused sub-questions.
Each sub-question should address a specific aspect that helps answer the main question.
Return only the sub-questions, one per line.

Main question: {query}

Sub-questions:"""

        try:
            response = self.llm_func(prompt)

            # Parse sub-queries
            sub_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering
                if line and len(line) > 2:
                    if line[0].isdigit() and line[1] in '.):':
                        line = line[2:].strip()
                    elif line[:2].replace('.', '').replace(')', '').isdigit():
                        line = line[3:].strip()

                    if line and line not in sub_queries:
                        sub_queries.append(line)

            # Include original query
            all_queries = [query] + sub_queries[:self.max_sub_queries]

            return ExpandedQuery(
                original_query=query,
                expanded_queries=all_queries,
                expansion_method="decomposition",
                metadata={"num_sub_queries": len(sub_queries)}
            )

        except Exception as e:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none",
                metadata={"error": str(e)}
            )


class StepBackExpander(BaseQueryExpander):
    """
    Step-back prompting for query abstraction.
    Generates a more general query to retrieve broader context.
    """

    def __init__(self, llm_func: Optional[Callable[[str], str]] = None):
        """
        Initialize step-back expander.

        Args:
            llm_func: Function to call LLM
        """
        self.llm_func = llm_func

    def expand(self, query: str) -> ExpandedQuery:
        """
        Generate a step-back query (more general/abstract).

        Args:
            query: Original query

        Returns:
            Expanded query with step-back version
        """
        if not self.llm_func:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none"
            )

        prompt = f"""Given the following specific question, generate a more general "step-back" question that would help provide broader context.

Specific question: {query}

Step-back question (more general):"""

        try:
            step_back_query = self.llm_func(prompt).strip()

            # Clean up the response
            if step_back_query.startswith('"') and step_back_query.endswith('"'):
                step_back_query = step_back_query[1:-1]

            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query, step_back_query],
                expansion_method="step_back",
                metadata={"step_back_query": step_back_query}
            )

        except Exception as e:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_method="none",
                metadata={"error": str(e)}
            )


class QueryExpansionPipeline:
    """
    Pipeline for applying multiple query expansion strategies.
    """

    def __init__(self, expanders: Optional[List[BaseQueryExpander]] = None):
        """
        Initialize expansion pipeline.

        Args:
            expanders: List of expanders to apply
        """
        self.expanders = expanders or []

    def add_expander(self, expander: BaseQueryExpander) -> None:
        """Add an expander to the pipeline."""
        self.expanders.append(expander)

    def expand(self, query: str) -> ExpandedQuery:
        """
        Apply all expansion strategies.

        Args:
            query: Original query

        Returns:
            Combined expanded query
        """
        all_queries = set([query])
        hypothetical_doc = None
        methods = []
        all_metadata = {}

        for expander in self.expanders:
            result = expander.expand(query)

            # Collect all expanded queries
            all_queries.update(result.expanded_queries)

            # Keep first hypothetical document
            if result.hypothetical_document and not hypothetical_doc:
                hypothetical_doc = result.hypothetical_document

            methods.append(result.expansion_method)
            all_metadata.update(result.metadata)

        return ExpandedQuery(
            original_query=query,
            expanded_queries=list(all_queries),
            hypothetical_document=hypothetical_doc,
            expansion_method="+".join(methods),
            metadata=all_metadata
        )


def create_query_expander(
    expander_type: str = "rewrite",
    llm_func: Optional[Callable[[str], str]] = None,
    **kwargs
) -> BaseQueryExpander:
    """
    Factory function to create a query expander.

    Args:
        expander_type: Type of expander
        llm_func: LLM function for expansion
        **kwargs: Additional arguments

    Returns:
        Query expander instance
    """
    expanders = {
        "rewrite": LLMQueryRewriter,
        "hyde": HyDEExpander,
        "decompose": QueryDecomposer,
        "step_back": StepBackExpander,
    }

    if expander_type not in expanders:
        raise ValueError(f"Unknown expander type: {expander_type}")

    return expanders[expander_type](llm_func=llm_func, **kwargs)
