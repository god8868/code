"""
模型管理器 - 统一管理和调用不同的AI模型适配器
"""

from typing import Dict, List, Optional, Any, Generator
from .base import BaseAdapter, ModelResponse, ModelConfig, ModelType
from .openai_adapter import OpenAIAdapter
from .claude_adapter import ClaudeAdapter
from .gemini_adapter import GeminiAdapter
from .config import Config


class ModelManager:
    """AI模型管理器"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._adapters: Dict[str, BaseAdapter] = {}
        self._adapter_classes = {
            ModelType.OPENAI: OpenAIAdapter,
            ModelType.CLAUDE: ClaudeAdapter,
            ModelType.GEMINI: GeminiAdapter,
        }
    
    def register_adapter(self, model_type: ModelType, adapter_class: type):
        """注册新的适配器类型"""
        if not issubclass(adapter_class, BaseAdapter):
            raise ValueError("Adapter class must inherit from BaseAdapter")
        self._adapter_classes[model_type] = adapter_class
    
    def get_adapter(self, model_name: str) -> BaseAdapter:
        """获取或创建适配器"""
        if model_name in self._adapters:
            return self._adapters[model_name]
        
        # 从配置获取模型配置
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model configuration not found for: {model_name}")
        
        # 创建适配器
        adapter_class = self._adapter_classes.get(model_config.model_type)
        if not adapter_class:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        
        adapter = adapter_class(model_config)
        self._adapters[model_name] = adapter
        return adapter
    
    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> ModelResponse:
        """生成文本"""
        model_name = model_name or self.config.get_default_model()
        if not model_name:
            raise ValueError("No model specified and no default model configured")
        
        adapter = self.get_adapter(model_name)
        return adapter.generate(prompt, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs) -> ModelResponse:
        """对话模式"""
        model_name = model_name or self.config.get_default_model()
        if not model_name:
            raise ValueError("No model specified and no default model configured")
        
        adapter = self.get_adapter(model_name)
        return adapter.chat(messages, **kwargs)
    
    def stream_generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """流式生成文本"""
        model_name = model_name or self.config.get_default_model()
        if not model_name:
            raise ValueError("No model specified and no default model configured")
        
        adapter = self.get_adapter(model_name)
        yield from adapter.stream_generate(prompt, **kwargs)
    
    def stream_chat(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        model_name = model_name or self.config.get_default_model()
        if not model_name:
            raise ValueError("No model specified and no default model configured")
        
        adapter = self.get_adapter(model_name)
        yield from adapter.stream_chat(messages, **kwargs)
    
    def compare_models(self, prompt: str, model_names: List[str], **kwargs) -> Dict[str, ModelResponse]:
        """比较多个模型的响应"""
        results = {}
        for model_name in model_names:
            try:
                response = self.generate(prompt, model_name=model_name, **kwargs)
                results[model_name] = response
            except Exception as e:
                results[model_name] = {
                    "error": str(e),
                    "model_name": model_name
                }
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        adapter = self.get_adapter(model_name)
        return adapter.get_model_info()
    
    def list_available_models(self) -> List[str]:
        """列出可用的模型"""
        return list(self.config.list_models().keys())
    
    def add_model(self, alias: str, config: ModelConfig):
        """添加新模型"""
        self.config.add_model_config(alias, config)
        # 清除缓存的适配器
        if alias in self._adapters:
            del self._adapters[alias]
    
    def remove_model(self, alias: str):
        """删除模型"""
        self.config.remove_model_config(alias)
        if alias in self._adapters:
            del self._adapters[alias]
    
    def set_default_model(self, model_name: str):
        """设置默认模型"""
        if model_name not in self.list_available_models():
            raise ValueError(f"Model {model_name} not found in configuration")
        self.config.set_default_model(model_name)
    
    def batch_generate(self, prompts: List[str], model_name: Optional[str] = None, **kwargs) -> List[ModelResponse]:
        """批量生成"""
        results = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, model_name=model_name, **kwargs)
                results.append(response)
            except Exception as e:
                results.append(ModelResponse(
                    content=f"Error: {str(e)}",
                    model_name=model_name or "unknown",
                    metadata={"error": True}
                ))
        return results
    
    def health_check(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """健康检查"""
        model_name = model_name or self.config.get_default_model()
        if not model_name:
            return {"status": "error", "message": "No model specified"}
        
        try:
            adapter = self.get_adapter(model_name)
            # 尝试一个简单的生成请求
            response = adapter.generate("Hello", max_tokens=10)
            return {
                "status": "healthy",
                "model_name": model_name,
                "latency": response.latency,
                "usage": response.usage
            }
        except Exception as e:
            return {
                "status": "error",
                "model_name": model_name,
                "error": str(e)
            }
    
    def clear_cache(self):
        """清除适配器缓存"""
        self._adapters.clear()
    
    def __repr__(self):
        return f"ModelManager(models={self.list_available_models()})"