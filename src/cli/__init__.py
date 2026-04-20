"""
CLI Module
Command-line interface for LocalRAG batch processing.
"""

from .batch import BatchProcessor, BatchConfig

__all__ = [
    "BatchProcessor",
    "BatchConfig",
]
