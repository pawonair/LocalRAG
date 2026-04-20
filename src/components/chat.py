"""
Chat UI Component
Renders chat messages with proper styling and auto-scroll.
"""

import streamlit as st
from typing import List, Optional
from memory.chat_memory import Message
from components.thinking_display import render_thinking_section


def render_chat_message(message: Message) -> None:
    """Render a single chat message with appropriate styling."""

    if message.role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Show thinking section if present
            if message.thinking:
                render_thinking_section(message.thinking)

            # Show main response
            st.markdown(message.content)

            # Show sources if present
            if message.sources:
                render_sources(message.sources)


def render_sources(sources: List[dict]) -> None:
    """Render source documents in an expandable section."""
    with st.expander("📚 Sources", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}:**")
            content = source.get("content", source.get("page_content", ""))
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            st.markdown(f"> {content}")
            if "source" in source:
                st.caption(f"From: {source['source']}")
            st.divider()


def render_chat_history(messages: List[Message]) -> None:
    """Render the full chat history."""
    for message in messages:
        render_chat_message(message)


def render_chat_input(disabled: bool = False) -> Optional[str]:
    """
    Render the chat input box.
    Returns the user's input if submitted, None otherwise.
    """
    return st.chat_input(
        placeholder="Ask a question about your documents...",
        disabled=disabled,
        key="chat_input"
    )


def render_streaming_message(placeholder, content: str, thinking: Optional[str] = None) -> None:
    """
    Render a message that's being streamed.
    Updates the placeholder with new content.
    """
    with placeholder.container():
        with st.chat_message("assistant", avatar="🤖"):
            if thinking:
                render_thinking_section(thinking, is_streaming=True)
            if content:
                st.markdown(content + "▌")  # Cursor indicator


def render_empty_state() -> None:
    """Render the empty state when no documents are loaded."""
    st.markdown(
        """
        <div style="text-align: center; padding: 50px 20px; color: #666;">
            <h3>Welcome to LocalRAG</h3>
            <p>Upload a PDF document to start chatting with your data.</p>
            <p style="font-size: 14px; margin-top: 20px;">
                Powered by Ollama + Deepseek R1
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_processing_status(status: str) -> None:
    """Render a processing status message."""
    st.info(f"⏳ {status}")
