"""
基础适配器类和数据结构定义
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """支持的模型类型"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class ModelResponse:
    """统一的模型响应格式"""
    content: str
    model_name: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAdapter(ABC):
    """基础适配器抽象类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_type = config.model_type
        self.model_name = config.model_name
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """对话模式"""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs):
        """流式生成文本"""
        pass
    
    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """流式对话"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.model_name})"