"""
Thinking Display Component
Parses and displays Deepseek's thinking/reasoning process.
"""

import streamlit as st
import re
from typing import Tuple, Optional


def parse_thinking(response: str) -> Tuple[Optional[str], str]:
    """
    Parse Deepseek response to extract thinking and final answer.

    Deepseek R1 outputs thinking in <think>...</think> tags.

    Args:
        response: Raw response from Deepseek model

    Returns:
        Tuple of (thinking_content, final_answer)
    """
    # Pattern to match <think>...</think> tags
    think_pattern = r'<think>(.*?)</think>'

    # Find all thinking sections
    thinking_matches = re.findall(think_pattern, response, re.DOTALL)

    # Combine all thinking content
    thinking_content = "\n\n".join(thinking_matches).strip() if thinking_matches else None

    # Remove thinking tags from response to get final answer
    final_answer = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()

    return thinking_content, final_answer


def render_thinking_section(thinking: str, is_streaming: bool = False) -> None:
    """
    Render the thinking/reasoning section in a collapsible expander.

    Args:
        thinking: The thinking content to display
        is_streaming: Whether the content is still being streamed
    """
    if not thinking:
        return

    # Create expander for thinking section
    label = "💭 Thinking..." if is_streaming else "💭 Reasoning Process"

    with st.expander(label, expanded=is_streaming):
        # Style the thinking content
        st.markdown(
            f"""
            <div class="thinking-content">
                {format_thinking_content(thinking)}
            </div>
            """,
            unsafe_allow_html=True
        )


def format_thinking_content(thinking: str) -> str:
    """
    Format thinking content for better readability.
    Handles markdown-like formatting within thinking blocks.
    """
    # Convert newlines to proper HTML breaks
    formatted = thinking.replace('\n', '<br>')

    # Wrap in styled div
    return f"""
    <div style="
        font-size: 13px;
        color: #666;
        font-style: italic;
        line-height: 1.6;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 3px solid #007BFF;
        border-radius: 4px;
    ">
        {formatted}
    </div>
    """


def stream_thinking_parser(chunk: str, buffer: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse streaming chunks for thinking content.

    Maintains state to track if we're inside a thinking block.

    Args:
        chunk: New chunk of text
        buffer: Accumulated buffer of text

    Returns:
        Tuple of (updated_buffer, thinking_content, answer_content)
    """
    buffer += chunk

    # Check if we have a complete thinking block
    if '<think>' in buffer and '</think>' in buffer:
        thinking, answer = parse_thinking(buffer)
        return "", thinking, answer

    # Check if we're still in thinking mode
    if '<think>' in buffer and '</think>' not in buffer:
        # Extract partial thinking content (after <think> tag)
        partial_thinking = buffer.split('<think>', 1)[-1]
        return buffer, partial_thinking, None

    # No thinking tags, return as answer
    if '<think>' not in buffer:
        return "", None, buffer

    return buffer, None, None
