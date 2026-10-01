"""Concrete LLM backends.

``OpenAICompatibleProvider`` covers any endpoint speaking the OpenAI Chat
Completions dialect -- Groq, OpenRouter, Cerebras, Together, DeepSeek, Ollama,
vLLM, LM Studio and others. ``GoogleGenAIProvider`` is the worked example of a
backend with its own wire format.
"""

from pramana.generation.providers.google_genai import GoogleGenAIProvider
from pramana.generation.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["GoogleGenAIProvider", "OpenAICompatibleProvider"]
