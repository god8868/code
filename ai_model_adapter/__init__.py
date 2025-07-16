"""
AI模型适配器 - 统一的AI模型调用接口

提供统一的接口来调用不同的AI模型提供商（OpenAI, Claude, Gemini等）
"""

from .base import BaseAdapter, ModelResponse, ModelConfig, ModelType
from .openai_adapter import OpenAIAdapter
from .claude_adapter import ClaudeAdapter
from .gemini_adapter import GeminiAdapter
from .model_manager import ModelManager
from .config import Config

__version__ = "1.0.0"
__all__ = [
    "BaseAdapter",
    "ModelResponse", 
    "ModelConfig",
    "ModelType",
    "OpenAIAdapter",
    "ClaudeAdapter", 
    "GeminiAdapter",
    "ModelManager",
    "Config"
]