"""
LLM Module
Ollama integration and model management.
"""

from .ollama import OllamaClient, OllamaConfig, ModelStatus
from .models import (
    ModelInfo,
    ModelCapability,
    ModelRegistry,
    SUPPORTED_MODELS,
    get_model_info,
)
from .prompts import PromptTemplate, PromptManager, DEFAULT_PROMPTS

__all__ = [
    # Ollama Client
    "OllamaClient",
    "OllamaConfig",
    "ModelStatus",

    # Models
    "ModelInfo",
    "ModelCapability",
    "ModelRegistry",
    "SUPPORTED_MODELS",
    "get_model_info",

    # Prompts
    "PromptTemplate",
    "PromptManager",
    "DEFAULT_PROMPTS",
]
