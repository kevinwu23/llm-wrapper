"""
LLM Wrapper - A unified Python wrapper for multiple Large Language Model APIs

This package provides a consistent interface for working with various LLM providers
including OpenAI, Anthropic Claude, Google Gemini, and Together AI.
"""

from .wrappers import (
    BaseWrapper,
    OpenAIWrapper,
    ClaudeWrapper, 
    GeminiWrapper,
    TogetherAIWrapper
)

__version__ = "0.1.0"
__author__ = "Kevin Wu"
__email__ = "kewu93@gmail.com"

__all__ = [
    "BaseWrapper",
    "OpenAIWrapper", 
    "ClaudeWrapper",
    "GeminiWrapper",
    "TogetherAIWrapper"
]