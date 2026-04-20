"""
Prompt Templates
Reusable prompt templates for different tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from string import Template


@dataclass
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    template: str
    description: str
    variables: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    category: str = "general"

    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.

        Args:
            **kwargs: Variables to substitute

        Returns:
            Formatted prompt string
        """
        t = Template(self.template)
        return t.safe_substitute(**kwargs)

    def format_with_system(self, **kwargs) -> tuple[Optional[str], str]:
        """
        Format both system prompt and template.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = self.format(**kwargs)

        if self.system_prompt:
            system = Template(self.system_prompt).safe_substitute(**kwargs)
            return system, user_prompt

        return None, user_prompt


# Default prompt templates
DEFAULT_PROMPTS: Dict[str, PromptTemplate] = {
    # RAG Prompts
    "rag_qa": PromptTemplate(
        name="rag_qa",
        template="""Use the following context to answer the question.

CONTEXT:
$context

QUESTION: $question

ANSWER:""",
        description="Basic RAG question-answering prompt",
        variables=["context", "question"],
        system_prompt="You are a helpful assistant that answers questions based on the provided context. Be accurate and cite sources when possible.",
        category="rag"
    ),

    "rag_qa_with_history": PromptTemplate(
        name="rag_qa_with_history",
        template="""Use the following context to answer the question.

CONTEXT:
$context

PREVIOUS CONVERSATION:
$chat_history

QUESTION: $question

ANSWER:""",
        description="RAG prompt with conversation history",
        variables=["context", "chat_history", "question"],
        system_prompt="You are a helpful assistant that answers questions based on the provided context. Consider the conversation history for context.",
        category="rag"
    ),

    "rag_strict": PromptTemplate(
        name="rag_strict",
        template="""Answer the question ONLY using the information in the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

CONTEXT:
$context

QUESTION: $question

ANSWER:""",
        description="Strict RAG prompt that only uses context",
        variables=["context", "question"],
        system_prompt="You are a precise assistant that only answers based on provided context. Never make up information.",
        category="rag"
    ),

    # Query Expansion Prompts
    "query_rewrite": PromptTemplate(
        name="query_rewrite",
        template="""Generate $num_rewrites alternative ways to ask the following question.
Each alternative should capture the same intent but use different words or perspectives.
Return only the alternative questions, one per line.

Original question: $question

Alternative questions:""",
        description="Rewrite query for better retrieval",
        variables=["question", "num_rewrites"],
        category="query_expansion"
    ),

    "query_decompose": PromptTemplate(
        name="query_decompose",
        template="""Break down the following complex question into simpler sub-questions.
Each sub-question should address a specific aspect that helps answer the main question.
Return only the sub-questions, one per line.

Main question: $question

Sub-questions:""",
        description="Decompose complex queries",
        variables=["question"],
        category="query_expansion"
    ),

    "hyde": PromptTemplate(
        name="hyde",
        template="""Write a short passage (2-3 sentences) that would perfectly answer the following question.
Write as if you're quoting from an authoritative source.

Question: $question

Passage:""",
        description="Hypothetical Document Embeddings",
        variables=["question"],
        category="query_expansion"
    ),

    "step_back": PromptTemplate(
        name="step_back",
        template="""Given the following specific question, generate a more general "step-back" question that would help provide broader context.

Specific question: $question

Step-back question:""",
        description="Step-back prompting for broader context",
        variables=["question"],
        category="query_expansion"
    ),

    # Summarization Prompts
    "summarize": PromptTemplate(
        name="summarize",
        template="""Summarize the following text concisely:

TEXT:
$text

SUMMARY:""",
        description="Basic text summarization",
        variables=["text"],
        system_prompt="You are a skilled summarizer. Create clear, concise summaries that capture key points.",
        category="summarization"
    ),

    "summarize_document": PromptTemplate(
        name="summarize_document",
        template="""Summarize the following document. Include:
1. Main topic and purpose
2. Key points and findings
3. Important conclusions

DOCUMENT:
$document

SUMMARY:""",
        description="Detailed document summarization",
        variables=["document"],
        category="summarization"
    ),

    # Analysis Prompts
    "analyze_code": PromptTemplate(
        name="analyze_code",
        template="""Analyze the following code and explain:
1. What the code does
2. Key functions and their purposes
3. Any potential issues or improvements

CODE:
```$language
$code
```

ANALYSIS:""",
        description="Code analysis and explanation",
        variables=["code", "language"],
        system_prompt="You are an expert programmer. Provide clear, technical explanations.",
        category="analysis"
    ),

    "extract_entities": PromptTemplate(
        name="extract_entities",
        template="""Extract key entities from the following text.
List each entity with its type (Person, Organization, Location, Date, etc.)

TEXT:
$text

ENTITIES:""",
        description="Named entity extraction",
        variables=["text"],
        category="analysis"
    ),

    # Chat Prompts
    "chat_friendly": PromptTemplate(
        name="chat_friendly",
        template="$message",
        description="Friendly conversational chat",
        variables=["message"],
        system_prompt="You are a friendly, helpful assistant. Be conversational and approachable.",
        category="chat"
    ),

    "chat_professional": PromptTemplate(
        name="chat_professional",
        template="$message",
        description="Professional business chat",
        variables=["message"],
        system_prompt="You are a professional assistant. Be clear, concise, and businesslike.",
        category="chat"
    ),

    # Re-ranking Prompt
    "rerank_relevance": PromptTemplate(
        name="rerank_relevance",
        template="""Rate the relevance of the following passage to the query on a scale of 0-10.
Only respond with a single number.

Query: $query

Passage: $passage

Relevance score (0-10):""",
        description="LLM-based relevance scoring",
        variables=["query", "passage"],
        category="reranking"
    ),
}


class PromptManager:
    """
    Manager for prompt templates.
    """

    def __init__(self):
        """Initialize prompt manager with default templates."""
        self._templates: Dict[str, PromptTemplate] = DEFAULT_PROMPTS.copy()

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def add(self, template: PromptTemplate) -> None:
        """Add a custom template."""
        self._templates[template.name] = template

    def remove(self, name: str) -> bool:
        """Remove a template."""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def list_templates(self, category: Optional[str] = None) -> List[PromptTemplate]:
        """List templates, optionally filtered by category."""
        templates = list(self._templates.values())

        if category:
            templates = [t for t in templates if t.category == category]

        return templates

    def list_categories(self) -> List[str]:
        """List all template categories."""
        return list(set(t.category for t in self._templates.values()))

    def format(self, name: str, **kwargs) -> str:
        """Format a template by name."""
        template = self.get(name)
        if template:
            return template.format(**kwargs)
        return ""

    def get_rag_prompt(
        self,
        context: str,
        question: str,
        chat_history: Optional[str] = None,
        strict: bool = False
    ) -> str:
        """
        Get formatted RAG prompt.

        Args:
            context: Document context
            question: User question
            chat_history: Optional chat history
            strict: Use strict mode (no external knowledge)

        Returns:
            Formatted prompt
        """
        if strict:
            return self.format("rag_strict", context=context, question=question)
        elif chat_history:
            return self.format(
                "rag_qa_with_history",
                context=context,
                question=question,
                chat_history=chat_history
            )
        else:
            return self.format("rag_qa", context=context, question=question)
