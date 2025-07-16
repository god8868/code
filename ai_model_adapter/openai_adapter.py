"""
OpenAI适配器实现
"""

import time
from typing import Dict, Any, List, Generator
from .base import BaseAdapter, ModelResponse, ModelConfig, ModelType


class OpenAIAdapter(BaseAdapter):
    """OpenAI适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
        self._init_client()
    
    def _validate_config(self) -> None:
        """验证配置"""
        if self.config.model_type != ModelType.OPENAI:
            raise ValueError(f"Expected OpenAI model type, got {self.config.model_type}")
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """对话模式"""
        start_time = time.time()
        
        # 合并参数
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            **self.config.extra_params,
            **kwargs
        }
        
        try:
            response = self._client.chat.completions.create(**params)
            latency = time.time() - start_time
            
            # 提取响应内容
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else None
            finish_reason = response.choices[0].finish_reason
            
            return ModelResponse(
                content=content,
                model_name=self.model_name,
                usage=usage,
                finish_reason=finish_reason,
                latency=latency,
                metadata={"response_id": response.id}
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成文本"""
        messages = [{"role": "user", "content": prompt}]
        yield from self.stream_chat(messages, **kwargs)
    
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """流式对话"""
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
            **self.config.extra_params,
            **kwargs
        }
        
        try:
            stream = self._client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"OpenAI stream API call failed: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            models = self._client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise RuntimeError(f"Failed to get available models: {str(e)}")