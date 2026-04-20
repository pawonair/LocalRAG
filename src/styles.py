"""
Styles Module
CSS styling for the LocalRAG application.
"""

import streamlit as st

# Color palette
PRIMARY_COLOR = "#007BFF"
SECONDARY_COLOR = "#FFC107"
BACKGROUND_COLOR = "#F8F9FA"
SIDEBAR_BACKGROUND = "#2C2F33"
TEXT_COLOR = "#212529"
SIDEBAR_TEXT_COLOR = "#FFFFFF"
HEADER_TEXT_COLOR = "#000000"

# Chat colors
USER_MESSAGE_BG = "#E3F2FD"
ASSISTANT_MESSAGE_BG = "#FFFFFF"
THINKING_BG = "#FFF8E1"


def apply_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Main Background */
        .stApp {
            background-color: #F8F9FA;
            color: #212529;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #2C2F33 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background-color: #007BFF;
            color: white;
            border: none;
            width: 100%;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #0056b3;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
            font-weight: bold;
        }

        /* Fix Text Visibility */
        /*p, span, div, label {
            color: #212529;
        }*/

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 8px;
        }

        /* User messages */
        [data-testid="stChatMessage"][data-testid*="user"] {
            background-color: #E3F2FD;
        }

        /* Assistant messages */
        [data-testid="stChatMessage"][data-testid*="assistant"] {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
        }

        /* Chat input styling */
        [data-testid="stChatInput"] {
            border-top: 1px solid #E0E0E0;
            padding-top: 16px;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 24px !important;
            border: 2px solid #E0E0E0 !important;
            padding: 12px 20px !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: #007BFF !important;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25) !important;
        }

        /* Thinking section */
        .thinking-content {
            background-color: #FFF8E1;
            border-left: 3px solid #FFC107;
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 4px;
            font-size: 13px;
            color: #666;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 14px;
            font-weight: 500;
        }

        /* Sources section */
        .sources-section {
            background-color: #F5F5F5;
            padding: 12px;
            border-radius: 8px;
            margin-top: 8px;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #007BFF;
            border-radius: 12px;
            padding: 20px;
            background-color: #FFFFFF;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #0056b3;
            background-color: #F8F9FA;
        }

        .stFileUploader > div > div > div > button {
            background-color: #007BFF;
            color: #FFFFFF;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Status messages */
        .stSuccess {
            background-color: #D4EDDA;
            border-color: #C3E6CB;
        }

        .stInfo {
            background-color: #E3F2FD;
            border-color: #BBDEFB;
        }

        .stWarning {
            background-color: #FFF8E1;
            border-color: #FFE082;
        }

        .stError {
            background-color: #FFEBEE;
            border-color: #FFCDD2;
        }

        /* Spinner */
        .stSpinner > div {
            border-color: #007BFF transparent transparent transparent;
        }

        /* Navigation Bar */
        header {
            background-color: #1E1E1E !important;
        }
        header * {
            color: #FFFFFF !important;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #F1F1F1;
        }

        ::-webkit-scrollbar-thumb {
            background: #C1C1C1;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #A1A1A1;
        }

        /* Document status badge */
        .doc-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 500;
        }

        .doc-status.ready {
            background-color: #D4EDDA;
            color: #155724;
        }

        .doc-status.processing {
            background-color: #FFF3CD;
            color: #856404;
        }

        /* Clear chat button */
        .clear-chat-btn {
            background-color: transparent;
            border: 1px solid #DC3545;
            color: #DC3545;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .clear-chat-btn:hover {
            background-color: #DC3545;
            color: white;
        }

        /* Welcome message */
        .welcome-container {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }

        .welcome-container h3 {
            color: #333 !important;
            margin-bottom: 16px;
        }

        .welcome-container p {
            color: #666 !important;
            font-size: 16px;
        }

        /* Model info badge */
        .model-badge {
            display: inline-block;
            padding: 4px 8px;
            background-color: #E3F2FD;
            color: #1976D2;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
