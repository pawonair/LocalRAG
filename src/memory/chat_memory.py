"""
Chat Memory Module
Manages conversation history with context window management.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Message:
    """Represents a single chat message."""
    role: str  # "user" or "assistant"
    content: str
    thinking: Optional[str] = None  # For Deepseek thinking tokens
    sources: Optional[List[dict]] = None  # Source documents
    timestamp: datetime = field(default_factory=datetime.now)


class ChatMemory:
    """
    Manages chat history in Streamlit session state.
    Provides context window management for token limiting.
    """

    SESSION_KEY = "chat_messages"
    MAX_MESSAGES = 50  # Maximum messages to keep in history

    @classmethod
    def initialize(cls) -> None:
        """Initialize chat memory in session state if not exists."""
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = []

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        thinking: Optional[str] = None,
        sources: Optional[List[dict]] = None
    ) -> None:
        """Add a message to chat history."""
        cls.initialize()

        message = Message(
            role=role,
            content=content,
            thinking=thinking,
            sources=sources
        )

        st.session_state[cls.SESSION_KEY].append(message)

        # Trim history if exceeds max
        if len(st.session_state[cls.SESSION_KEY]) > cls.MAX_MESSAGES:
            st.session_state[cls.SESSION_KEY] = st.session_state[cls.SESSION_KEY][-cls.MAX_MESSAGES:]

    @classmethod
    def get_messages(cls) -> List[Message]:
        """Get all messages from chat history."""
        cls.initialize()
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def get_context_messages(cls, max_pairs: int = 5) -> List[dict]:
        """
        Get recent message pairs for context.
        Returns list of dicts with 'role' and 'content' keys.
        """
        cls.initialize()
        messages = st.session_state[cls.SESSION_KEY]

        # Get last N message pairs (user + assistant)
        recent = messages[-(max_pairs * 2):] if len(messages) > max_pairs * 2 else messages

        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]

    @classmethod
    def get_context_string(cls, max_pairs: int = 3) -> str:
        """
        Get conversation context as a formatted string.
        Useful for including in prompts.
        """
        messages = cls.get_context_messages(max_pairs)

        if not messages:
            return ""

        context_parts = []
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg['content']}")

        return "\n".join(context_parts)

    @classmethod
    def clear(cls) -> None:
        """Clear all chat history."""
        st.session_state[cls.SESSION_KEY] = []

    @classmethod
    def is_empty(cls) -> bool:
        """Check if chat history is empty."""
        cls.initialize()
        return len(st.session_state[cls.SESSION_KEY]) == 0

    @classmethod
    def count(cls) -> int:
        """Get number of messages in history."""
        cls.initialize()
        return len(st.session_state[cls.SESSION_KEY])
